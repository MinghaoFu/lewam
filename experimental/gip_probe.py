"""Linear-probe the GIP encoder latent for robot proprio (11-D, visible) vs the
full privileged sim state (71-D, incl. object pose). If latent decodes proprio
well but state poorly, the encoder sees the robot but NOT the object -> the
object-perception bottleneck behind histbc << BC-RNN. Evidence experiment."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import torch
import stable_worldmodel as swm
import stable_worldmodel.data.formats.hdf5  # noqa
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from lewam.utils import get_img_preprocessor
import lewam.models.gip as gip

DSET = os.environ.get("PROBE_DSET", "can.h5")
RUN = os.environ.get("PROBE_RUN", "gip_robomimic_can")

model, adim = gip.load_gip_model(RUN)
model = model.to("cuda").eval()
model.requires_grad_(False)
model.interpolate_pos_encoding = True

ds = swm.data.load_dataset(DSET, transform=None,
                           keys_to_load=["pixels", "proprio", "state"],
                           num_steps=1, frameskip=5)
ds.transform = get_img_preprocessor(source="pixels", target="pixels", img_size=224)
dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True, num_workers=4)

Z, P, S = [], [], []
with torch.no_grad():
    for i, b in enumerate(dl):
        if i >= 50:
            break
        px = b["pixels"]
        px = (px if torch.is_tensor(px) else torch.as_tensor(px)).cuda().float()
        out = model.encode({"pixels": px})
        z = out["emb"][:, 0].cpu().numpy()            # (B, 192) CLS latent
        Z.append(z)
        P.append(np.asarray(b["proprio"])[:, 0])      # (B, 11)
        S.append(np.asarray(b["state"])[:, 0])        # (B, 71)
Z = np.concatenate(Z); P = np.concatenate(P); S = np.concatenate(S)
n = len(Z); tr = int(0.8 * n)
print(f"[probe] {RUN} on {DSET}: N={n} latent={Z.shape[1]} proprio={P.shape[1]} state={S.shape[1]}")

def probe(name, Y):
    Ytr, Yte = Y[:tr], Y[tr:]
    # standardize targets so R2 is comparable across dims
    mu, sd = Ytr.mean(0), Ytr.std(0) + 1e-6
    r = Ridge(alpha=10.0).fit(Z[:tr], (Ytr - mu) / sd)
    pred = r.predict(Z[tr:])
    r2 = r2_score((Yte - mu) / sd, pred, multioutput="variance_weighted")
    print(f"[probe] latent -> {name}: R2 = {r2:.3f}")
    return r2

probe("proprio(robot,11d)", P)
probe("state(full+object,71d)", S)
