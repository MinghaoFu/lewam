"""Merged world-action model: one shared ARPredictor backbone for both heads.

Vanilla `JEPA` (lewam/models/jepa.py) runs the forward-dynamics head (`predict`) and the
action head (`predict_intention`) on two SEPARATE `ARPredictor` transformers --
`self.predictor` and `self.action_predictor` -- sharing only the encoder. `MergedJEPA` ties
the two: the action head runs on the SAME backbone as dynamics, so there is one shared
transformer with two readout heads (`pred_proj` -> z_{t+1}, `action_decoder` -> a_t) and the
two AdaLN conditioning modes (a_{<=t} for dynamics vs a_{<t}+goal+horizon for the action)
left unchanged.

Select it by pointing the model target at this class; no other code changes are needed:

    python scripts/train.py model._target_=lewam.models.jepa_merged.MergedJEPA \\
        data=<task> action_pred.enabled=true action_pred.goal_conditioned=true \\
        action_pred.horizon_conditioned=true

Eval is unchanged: config.json carries the _target_, so load_gip_model rebuilds MergedJEPA
and the tie re-applies automatically.
"""

from lewam.models.jepa import JEPA


class MergedJEPA(JEPA):
    """JEPA whose action predictor shares the forward-dynamics backbone (one ARPredictor)."""

    def __setattr__(self, name, value):
        # The trainer (scripts/train.py) and eval (gip.load_gip_model) both attach the action
        # head the same way:  model.action_predictor = instantiate(cfg.model.predictor).
        # Intercept that assignment and alias it to the dynamics backbone (self.predictor), so
        # both heads run on ONE ARPredictor. The freshly-instantiated instance is discarded;
        # parameters() de-dups by identity, so the shared backbone is optimized exactly once.
        if name == "action_predictor" and getattr(self, "predictor", None) is not None:
            value = self.predictor
        super().__setattr__(name, value)
