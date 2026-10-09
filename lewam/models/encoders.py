"""Image encoders: a backbone producing one feature vector per image, then a projector to the latent space."""

import torch
from torch import nn
from stable_pretraining.backbone.utils import vit_hf

from lewam.models.module import MLP


class Encoder(nn.Module):
    """A backbone (features) followed by the projector; subclasses build both in __init__."""

    def build_projector(self, feature_dim: int, output_dim: int, proj_hidden: int) -> None:
        # BatchNorm in the projector: the backbones' final LayerNorm keeps SIGReg from being optimized effectively
        self.projector = MLP(
            input_dim=feature_dim, output_dim=output_dim, hidden_dim=proj_hidden,
            norm_fn=nn.BatchNorm1d, norm_first=False
        )

    def features(self, pixels: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """
        Encode images

        Args:
            pixels (torch.Tensor): (N, 3, H, W) normalized images

        Returns:
            latents (torch.Tensor): (N, output_dim) latents
        """
        return self.projector(self.features(pixels))


def batchnorm_to_groupnorm(module: nn.Module, num_groups: int = 16) -> nn.Module:
    """Replace every BatchNorm2d in `module` (in place, recursively) with GroupNorm(num_groups)."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(num_groups, child.num_features))
        else:
            batchnorm_to_groupnorm(child, num_groups)
    return module


class ResNetEncoder(Encoder):
    """ResNet18 trunk and a 32-keypoint spatial softmax and optional group_norm."""

    def __init__(self, output_dim: int, proj_hidden: int, group_norm: bool = True, crop_size: int | None = 202):
        """
        Args:
            output_dim (int): latent size
            proj_hidden (int): hidden width of the projector
            group_norm (bool): GroupNorm(16) in place of the trunk's BatchNorm
            crop_size (int | None): crop the images to this size (random offset in training, centered in eval)
        """
        super().__init__()
        import torchvision

        trunk = nn.Sequential(*list(torchvision.models.resnet18(weights=None).children())[:-2])
        self.trunk = batchnorm_to_groupnorm(trunk) if group_norm else trunk
        self.kp_conv = nn.Conv2d(512, 32, kernel_size=1)
        self.crop_size = crop_size
        self.build_projector(64, output_dim, proj_hidden)

    def crop(self, pixels: torch.Tensor) -> torch.Tensor:
        """Crop (N, 3, H, W) to crop_size: a random offset per image in training, the center in eval."""
        N, C, H, W = pixels.shape
        s = self.crop_size
        if self.training:
            top = torch.randint(0, H - s + 1, (N,), device=pixels.device)
            left = torch.randint(0, W - s + 1, (N,), device=pixels.device)
        else:
            top = pixels.new_full((N,), (H - s) // 2, dtype=torch.long)
            left = pixels.new_full((N,), (W - s) // 2, dtype=torch.long)
        # index into unfolded views: gathering with broadcast indices materialized ~2GB per batch
        windows = pixels.unfold(2, s, 1).unfold(3, s, 1)            # (N, C, H-s+1, W-s+1, s, s) view
        return windows[torch.arange(N, device=pixels.device), :, top, left]

    def features(self, pixels: torch.Tensor) -> torch.Tensor:
        """(N, 3, H, W) -> (N, 64): the expected (x, y) image coordinate of each of 32 keypoint maps."""
        if self.crop_size:
            pixels = self.crop(pixels)
        fmap = self.kp_conv(self.trunk(pixels))                      # (N, 32, h, w)
        N, K, H, W = fmap.shape
        attn = fmap.flatten(2).softmax(-1)                          # (N, 32, h*w)
        xs = torch.linspace(-1.0, 1.0, W, device=fmap.device)
        ys = torch.linspace(-1.0, 1.0, H, device=fmap.device)
        grid_x = xs.repeat(H)                                       # row-major flatten: column fastest
        grid_y = ys.repeat_interleave(W)
        return torch.cat([(attn * grid_x).sum(-1), (attn * grid_y).sum(-1)], dim=-1)


class ViTEncoder(Encoder):
    """ViT trained from scratch using CLS token as its latent."""

    def __init__(self, output_dim: int, proj_hidden: int, size: str = "tiny", img_size: int = 224):
        """
        Args:
            output_dim (int): latent size
            proj_hidden (int): hidden width of the projector
            size (str): tiny | small | base | large
            img_size (int): input image size
        """
        super().__init__()
        self.vit = vit_hf(size=size, patch_size=14, image_size=img_size, pretrained=False, use_mask_token=False)
        self.build_projector(self.vit.config.hidden_size, output_dim, proj_hidden)

    def features(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.vit(pixels, interpolate_pos_encoding=True).last_hidden_state[:, 0]


class DINOv3Encoder(Encoder):
    """Frozen pretrained DINOv3 ViT (timm); only the projector trains."""

    TIMM_MODELS = {
        "small": "vit_small_patch16_dinov3",
        "base": "vit_base_patch16_dinov3",
        "large": "vit_large_patch16_dinov3",
    }

    def __init__(self, output_dim: int, proj_hidden: int, size: str = "small", checkpoint: str | None = None):
        """
        Args:
            output_dim (int): latent size
            proj_hidden (int): hidden width of the projector
            size (str): small | base | large
            checkpoint (str | None): DINOv3 weights to load; None downloads timm's, "defer" builds the architecture
                only (the weights come with a full-model checkpoint)
        """
        super().__init__()
        import timm

        self.vit = timm.create_model(self.TIMM_MODELS[size], pretrained=checkpoint is None, num_classes=0)
        if checkpoint not in (None, "defer"):
            self.vit.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
        self.vit.requires_grad_(False)
        self.vit.eval()
        self.build_projector(self.vit.num_features, output_dim, proj_hidden)

    def train(self, mode: bool = True):
        super().train(mode)
        self.vit.eval()
        return self

    def features(self, pixels: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.vit(pixels)


def build_encoder(backbone: str, output_dim: int, proj_hidden: int, size: str = "tiny", img_size: int = 224,
                  checkpoint: str | None = None) -> Encoder:
    """
    The image encoder for a backbone name

    Args:
        backbone (str): resnet18dp (Diffusion Policy's ResNet18: GroupNorm, 202 crop) | resnet18 | vit | dinov3
        output_dim (int): latent size
        proj_hidden (int): hidden width of the projector
        size (str): ViT / DINOv3 size
        img_size (int): input image size (ViT)
        checkpoint (str | None): DINOv3 weights

    Returns:
        encoder (Encoder): the encoder
    """
    if backbone == "resnet18dp":
        return ResNetEncoder(output_dim, proj_hidden, group_norm=True, crop_size=202)
    if backbone == "resnet18":
        return ResNetEncoder(output_dim, proj_hidden, group_norm=False, crop_size=None)
    if backbone == "vit":
        return ViTEncoder(output_dim, proj_hidden, size=size, img_size=img_size)
    if backbone == "dinov3":
        return DINOv3Encoder(output_dim, proj_hidden, size=size, checkpoint=checkpoint)
    raise ValueError(f"unknown encoder backbone {backbone!r}")
