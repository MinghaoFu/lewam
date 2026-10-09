"""Evaluation data: the dataset and the seeded draw of (episode, start step) pairs."""

import h5py
import numpy as np
import torch
from pathlib import Path
import stable_pretraining as spt

from torchvision.transforms import v2 as transforms
from stable_worldmodel.data import LanceDataset, HDF5Dataset
from stable_worldmodel.data.utils import get_cache_dir


def img_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
        ]
    )


def episode_col(dataset: LanceDataset | HDF5Dataset) -> str:
    return "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"


def get_dataset(
    dataset_name: str, keys_to_cache: list[str], cache_dir: str | None = None
) -> LanceDataset | HDF5Dataset:
    """Returns `<dataset_name>.lance` in the cache dir or its datasets/ folder if present,
    else `<cache dir>/datasets/<dataset_name>.h5`."""
    dataset_path = Path(cache_dir or get_cache_dir())
    stem = str(dataset_name).removesuffix(".lance")
    lance_candidates = [dataset_path / f"{stem}.lance", dataset_path / "datasets" / f"{stem}.lance"]
    lance_path = next((path for path in lance_candidates if path.exists()), None)
    if lance_path is not None:
        column_names = getattr(LanceDataset(path=str(lance_path), keys_to_cache=keys_to_cache), "column_names", [])
        keys_to_load = [key for key in ("pixels", "action", "observation", "step_idx", "episode_idx", "ep_idx")
                        if key in column_names]
        return LanceDataset(path=str(lance_path), keys_to_load=keys_to_load, keys_to_cache=keys_to_cache)

    # load only per-step columns: a per-episode column such as model_xml (one row per episode) would be
    # step-indexed by get_row_data and raise IndexError
    with h5py.File(dataset_path / "datasets" / f"{dataset_name}.h5", "r") as h5:
        num_rows = len(h5["step_idx"])
        keys_to_load = [key for key in h5.keys()
                        if key not in ("ep_len", "ep_offset") and h5[key].shape[:1] == (num_rows,)]
    return HDF5Dataset(dataset_name, keys_to_load=keys_to_load, keys_to_cache=keys_to_cache, cache_dir=dataset_path)


def sample_eval_episodes(
    dataset: LanceDataset | HDF5Dataset, num_eval: int, seed: int, goal_offset_steps: int,
    full_traj: bool = False, random_start: bool = True, min_episode_len: int = 0
) -> tuple[list[int], list[int], list[int] | None]:
    """
    Draw num_eval (episode, start step) pairs, deterministically from seed. The goal is goal_offset_steps
    after the start, and episodes too short to reach it are excluded.

    Args:
        dataset (LanceDataset | HDF5Dataset): the eval dataset
        num_eval (int): pairs to draw
        seed (int): seed of the draw
        goal_offset_steps (int): steps from the start to the goal
        full_traj (bool): start at each episode's first frame with its last frame as the goal
        random_start (bool): random draw among eligible starts (episodes may repeat); False draws only
            from the first frame of eligible episodes
        min_episode_len (int): draw only from episodes at least this long, so one seed picks the
            same episodes for every goal offset

    Returns:
        episodes (list[int]): the episode of each pair
        starts (list[int]): the start step of each pair
        goal_offsets (list[int] | None): each pair's goal offset under full_traj, else None
    """
    episode_column = episode_col(dataset)
    episode_ids = np.unique(dataset.get_col_data(episode_column))
    row_steps = np.asarray(dataset.get_col_data("step_idx")).reshape(-1)    # step within its episode
    row_episodes = np.asarray(dataset.get_col_data(episode_column)).reshape(-1)
    lengths = np.array([np.max(row_steps[row_episodes == episode]) + 1 for episode in episode_ids])
    rng = np.random.default_rng(seed)

    if full_traj:
        first_frames = np.nonzero(row_steps == 0)[0]
        picks = np.sort(first_frames[rng.choice(len(first_frames), size=num_eval, replace=False)])
        episodes = dataset.get_row_data(picks)[episode_column]
        length_of = {episode: int(length) for episode, length in zip(episode_ids, lengths)}
        return episodes.tolist(), [0] * len(episodes), [length_of[episode] - 1 for episode in episodes]

    max_start = lengths - goal_offset_steps - 1
    max_start_by_episode = {episode: start for episode, start in zip(episode_ids, max_start)}
    row_max_start = np.array([max_start_by_episode[episode] for episode in row_episodes])
    row_length = lengths[np.searchsorted(episode_ids, row_episodes)]
    eligible = (row_max_start >= 0) & (row_length >= min_episode_len)

    if not random_start:
        first_frames = np.nonzero((row_steps == 0) & eligible)[0]
        picks = np.sort(first_frames[rng.choice(len(first_frames), size=num_eval, replace=False)])
        return dataset.get_row_data(picks)[episode_column].tolist(), [0] * len(picks), None

    valid = np.nonzero((row_steps <= row_max_start) & eligible)[0]
    if len(valid) - 1 < num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    # len(valid) - 1 matches lewm's eval.py draw
    picks = np.sort(valid[rng.choice(len(valid) - 1, size=num_eval, replace=False)])
    rows = dataset.get_row_data(picks)
    return rows[episode_column].tolist(), rows["step_idx"].tolist(), None
