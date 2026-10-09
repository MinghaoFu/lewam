"""Utils to handle multi-camera views

The first view of a model is the scene camera the env's `render()` returns (info key `pixels`).
Every other view reaches the policy as info key `pixels.<camera>`, rendered into the env's observation dict.
"""

CAMERA_OF_COLUMN = {
    "pixels": "agentview",
    "pixels_agentview": "agentview",
    "pixels_r0eih": "robot0_eye_in_hand",
    "pixels_r1eih": "robot1_eye_in_hand",
}


def columns(spec):
    """The view columns of a checkpoint config: a list, a comma string, or nothing (one view, `pixels`)."""
    if not spec:
        return ["pixels"]
    if isinstance(spec, str):
        return [v for v in spec.split(",") if v]
    return list(spec)


def robosuite_camera_of(column: str) -> str:
    """The robosuite camera behind an h5 image column (`pixels_<name>` falls back to `<name>`)."""
    return CAMERA_OF_COLUMN.get(column, column[len("pixels_"):] if column.startswith("pixels_") else column)


def info_keys(views: list[str]) -> list[str]:
    """The info keys of a model's views: `pixels` for the first, `pixels.<camera>` for the others."""
    return ["pixels"] + [f"pixels.{robosuite_camera_of(view)}" for view in views[1:]]


def extra_cameras(views: list[str]) -> list[str]:
    """The cameras an env must add to its observation dict for these views."""
    return [robosuite_camera_of(view) for view in views[1:]]


def goal_info_keys(goal_views: list[str]):
    """The info keys of a model's goal views, as the World names the goal frame of each dataset column.
    `goal` for the scene column `pixels`, `goal_<column>` for every other camera column."""
    return ["goal" if view == "pixels" else f"goal_{view}" for view in goal_views]
