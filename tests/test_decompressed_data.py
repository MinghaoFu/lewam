
import h5py
import numpy as np
import pytest
import torch

from scripts.decompress_h5 import decompress_column
from lewam.train.datasets import H5Frames
from lewam.train.datasets import TrajectoryDataset, get_start_indices, load_frames, read_actions


def prepare_decomp(path, root, views):
    folder = root / path.stem
    folder.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "r") as h5:
        for view in views:
            decompress_column(h5[view], H5Frames(str(path), view), str(folder / (view + ".npy")))


@pytest.fixture
def data(tmp_path):
    path = tmp_path / "toy.h5"
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f["pixels"] = rng.integers(0, 256, (40, 8, 8, 3), dtype=np.uint8)
        f["pixels_wrist"] = rng.integers(0, 256, (40, 8, 8, 3), dtype=np.uint8)
        actions = rng.normal(size=(40, 5)).astype(np.float32)
        actions[:, -1] = 2
        f["action"] = actions
        f["ep_offset"] = [0, 20]
        f["ep_len"] = [20, 20]
        success = np.zeros(40, dtype=bool)
        success[[7, 29]] = True
        f["success"] = success
    return path, tmp_path / "cache"


@pytest.mark.parametrize("terminal", ["last", "success"])
@pytest.mark.parametrize("ram", [True, False])
@pytest.mark.parametrize("views", [["pixels"], ["pixels", "pixels_wrist"]])
def test_samples_match_hdf5(data, terminal, ram, views):
    path, root = data
    prepare_decomp(path, root, views)
    frames = load_frames(str(path), views, "decomp", str(root), ram)
    actions, stats = read_actions(str(path))
    starts, bases, ttl = get_start_indices(str(path), 5, terminal)
    direct = load_frames(str(path), views, "h5")
    raw_actions, raw_stats = read_actions(str(path))
    expected_indices = get_start_indices(str(path), 5, terminal)
    assert stats == raw_stats
    assert torch.equal(actions, raw_actions)
    assert all(torch.equal(a, b) for a, b in zip((starts, bases, ttl), expected_indices))
    indices = torch.arange(len(starts))
    datasets = [TrajectoryDataset(f, actions, starts, bases, ttl, indices, 2, 5, 10, 1, 5, "sampled", 10) for f in (frames, direct)]
    for i in range(len(starts)):
        torch.manual_seed(i); cached = datasets[0][i]
        torch.manual_seed(i); original = datasets[1][i]
        assert all(torch.equal(a, b) for a, b in zip(cached, original))


def test_worker_batches_match_hdf5(data):
    from torch.utils.data import DataLoader

    path, root = data
    prepare_decomp(path, root, ["pixels"])
    frames = load_frames(str(path), ["pixels"], "decomp", str(root), True)
    actions, _ = read_actions(str(path))
    starts, bases, ttl = get_start_indices(str(path), 5, "last")
    batches = []
    for source in (frames, load_frames(str(path), ["pixels"], "h5")):
        dataset = TrajectoryDataset(source, actions, starts, bases, ttl, torch.arange(len(starts)), 2, 5, 10, 1, 5, "sampled", 10)
        loader = DataLoader(dataset, batch_size=8, num_workers=2, generator=torch.Generator().manual_seed(42))
        batches.append(list(loader))
    assert len(batches[0]) == len(batches[1])
    for cached, original in zip(*batches):
        assert all(torch.equal(a, b) for a, b in zip(cached, original))
