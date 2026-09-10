"""Camera views: the h5 image columns a multi-view model trains on, and the robosuite cameras the
eval envs render for them. The first view of a model is the scene camera the env's `render()`
returns (info key `pixels`); every other view reaches the policy as info key `pixels.<camera>`,
rendered into the env's observation dict."""

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


def camera_of(column):
    """The robosuite camera behind an h5 image column (`pixels_<name>` falls back to `<name>`)."""
    return CAMERA_OF_COLUMN.get(column, column[len("pixels_"):] if column.startswith("pixels_") else column)


def info_keys(views):
    """The info keys of a model's views: `pixels` for the first, `pixels.<camera>` for the others."""
    return ["pixels"] + [f"pixels.{camera_of(view)}" for view in views[1:]]


def extra_cameras(views):
    """The cameras an env must add to its observation dict for these views."""
    return [camera_of(view) for view in views[1:]]
