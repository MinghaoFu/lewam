def build_eval(cfg):
    """Lazy eval-backend factory, mirroring datasets/. Same `backend=` dimension picks
    the matching eval environment, so training and eval always share one world.
    """
    backend = cfg.backend
    if backend == "offline":
        from .offline import OfflineEval
        return OfflineEval(cfg)
    if backend == "lerobot":
        from .lerobot import LeRobotEval
        return LeRobotEval(cfg)
    raise ValueError(
        f"eval backend '{backend}' not installed. "
        f"try: uv pip install 'lewam-real-robot[{backend}]'"
    )
