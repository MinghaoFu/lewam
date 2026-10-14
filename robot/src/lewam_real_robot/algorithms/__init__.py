def build_algorithm(cfg):
    if cfg.name == "generic_vit_predictor":
        from .generic_vit_predictor import GenericViTPredictor
        return GenericViTPredictor(cfg)
    if cfg.name == "lerobot_policy":
        from .lerobot_policy import LeRobotPolicy
        return LeRobotPolicy(cfg)
    raise ValueError(f"unknown algorithm '{cfg.name}'")
