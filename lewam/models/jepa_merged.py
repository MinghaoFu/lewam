"""Merged world-action model: one shared ARPredictor backbone, with a per-mode embedding.

Vanilla `JEPA` (lewam/models/jepa.py) runs the forward-dynamics head (`predict`) and the
action head (`predict_intention`) on two SEPARATE `ARPredictor` transformers. `MergedJEPA`
ties them to ONE shared trunk. To keep that shared trunk from having to *infer* which pass it
serves -- dynamics conditions on a_{<=t}, the action head on a_{<t}+goal+horizon -- each pass
adds a learned per-mode bias to the AdaLN conditioning. So there is one shared transformer
with two tiny mode vectors and the two readout heads (pred_proj -> z_{t+1}, action_decoder -> a_t).

Select with model=lewm_merged (or model._target_=lewam.models.jepa_merged.MergedJEPA); eval
rebuilds it automatically from config.json (the _target_ rides in the saved config).
"""

import torch
from torch import nn

from lewam.models.jepa import JEPA


class _ModedPredictor(nn.Module):
    """Wraps the shared ARPredictor trunk and adds a learned per-mode bias to the conditioning c.

    Both the dynamics wrapper and the action wrapper hold the SAME trunk object (shared weights);
    only the `mode` bias differs, so one backbone can tell 'dynamics' from 'action'.
    """

    def __init__(self, trunk, emb_dim):
        super().__init__()
        self.trunk = trunk                                    # shared across both wrappers
        self.mode = nn.Parameter(torch.zeros(1, 1, emb_dim))  # zero-init => identity at step 0

    def forward(self, x, c, proprio=None):
        return self.trunk(x, c + self.mode, proprio)          # c: (B,T,D); mode broadcasts

    @property
    def pos_embedding(self):
        return self.trunk.pos_embedding


class MergedJEPA(JEPA):
    """JEPA whose action head shares the dynamics backbone, with a per-mode conditioning bias."""

    def __setattr__(self, name, value):
        # The trainer (scripts/train.py) and eval (gip.load_gip_model) both attach the action head:
        #   model.action_predictor = instantiate(cfg.model.predictor)
        # Intercept it and wrap the SHARED trunk in two moded predictors (dynamics + action). Both
        # wrap the same trunk -- parameters() de-dups by identity, so it stays ONE backbone; only
        # +2*D params for the two mode biases. The freshly-instantiated ARPredictor is discarded.
        if name == "action_predictor" and getattr(self, "predictor", None) is not None:
            trunk = getattr(self.predictor, "trunk", self.predictor)   # unwrap if already wrapped
            emb_dim = trunk.pos_embedding.shape[-1]
            super().__setattr__("predictor", _ModedPredictor(trunk, emb_dim))  # dynamics mode
            value = _ModedPredictor(trunk, emb_dim)                            # action mode
        super().__setattr__(name, value)
