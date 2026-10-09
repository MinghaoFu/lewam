"""
We use hdf5 (h5) to store our data
Layout:
    dataset.h5
    ├── pixels      # (total_steps, H, W, C) uint8
    ├── action      # (total_steps, action_dim) float32
    ├── ep_len      # (num_episodes,) int32
    └── ep_offset   # (num_episodes,) int64

For multi-view/camera datasets, "pixels" is separated into one column per view
e.g. "pixels_scene0", "pixels_wrist0", etc.
"""

import os

import h5py
import hdf5plugin  # noqa: F401  the swm h5 files compress frames with plugin filters
import numpy as np
import torch
from torch.utils.data import Dataset


class H5Frames:
    """H5 image column returned as uint8 CHW."""

    def __init__(self, dataset_path: str, img_column: str):
        self.dataset_path, self.img_column = dataset_path, img_column
        with h5py.File(dataset_path, "r") as f:
            assert f[img_column].dtype == np.uint8, f"{img_column}: expected uint8 frames, got {f[img_column].dtype}"
            n, *frame = f[img_column].shape
        self.channels_last = frame[-1] == 3
        self.shape = torch.Size((n, 3, *frame[:2]) if self.channels_last else (n, *frame))
        self.dtype = torch.uint8
        self._file, self._column, self._pid = None, None, None

    def __len__(self) -> int:
        return self.shape[0]

    def __getstate__(self) -> dict:
        return {**self.__dict__, "_file": None, "_column": None, "_pid": None}

    def __getitem__(self, idx) -> torch.Tensor:
        if self._pid != os.getpid():  # an open h5 handle must not be shared with DataLoader workers
            self._file = h5py.File(
                self.dataset_path, "r", rdcc_nbytes=64 * 1024**2, rdcc_nslots=1009
            )
            self._column = self._file[self.img_column]
            self._pid = os.getpid()

        column = self._column
        if isinstance(idx, slice):
            frames = column[idx]
        elif np.ndim(idx) == 0:
            frames = column[int(idx)]
        else:
            unique, inverse = np.unique(np.asarray(idx), return_inverse=True)
            if len(unique):
                frames = np.stack([column[int(i)] for i in unique])[inverse]
            else:
                frames = np.empty((0, *column.shape[1:]), dtype=column.dtype)
        frames = torch.from_numpy(np.ascontiguousarray(frames))

        return frames.movedim(-1, -3).contiguous() if self.channels_last else frames


class MultiViewFrames:
    """
    Stack multiple columns from different cameras for multi-view observations 

    Indexing:
    - `frames[i]` -> (num_views, 3, H, W)
    - `frames[a:b]` -> (b - a, num_views, 3, H, W)
    """

    def __init__(self, views):
        self.views = views
        self.shape = (views[0].shape[0], len(views), *views[0].shape[1:])
        self.dtype = views[0].dtype
        assert all(v.shape == views[0].shape for v in views), "every camera's frames must have the same shape"

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        return torch.stack([v[idx] for v in self.views], dim=-4)
    

def read_episode_meta(dataset_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        dataset_path (str): path to h5 dataset
    
    Returns:
        ep_offsets (np.ndarray): (num_episodes,) global h5 row of each episode's first frame
        ep_len (np.ndarray): (num_episodes,) number of frames in each episode
    """
    with h5py.File(dataset_path, "r") as f:
        ep_offset = np.asarray(f["ep_offset"][:]).reshape(-1)
        ep_len = np.asarray(f["ep_len"][:]).reshape(-1)
    ep_offsets, first_ep_idx = np.unique(ep_offset, return_index=True)
    return ep_offsets.astype(np.int64), ep_len[first_ep_idx].astype(np.int64)


def read_actions(dataset_path: str) -> tuple[torch.Tensor, tuple[list[float], list[float]]]:
    """
    Args:
        dataset_path (str): path to h5 dataset

    Returns:
        actions (torch.Tensor): (total_steps, action_dim) one z-scored action per h5 row
        stats (tuple[list[float], list[float]]): per-dimension mean and std over the rows without NaN,
            with constant dims getting std 1
    """
    with h5py.File(dataset_path, "r") as f:
        raw = torch.from_numpy(np.asarray(f["action"][:]))
    no_nan = raw[~torch.isnan(raw).any(dim=1)]
    mean, std = no_nan.mean(0), no_nan.std(0)
    std[std == 0] = 1.0
    mean, std = mean.tolist(), std.tolist()
    actions = (raw.float() - torch.tensor(mean)) / torch.tensor(std)
    return actions.half(), (mean, std)


def get_start_indices(dataset_path: str, frameskip: int, terminal_state: str = "last") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return every frame that can be a training sample

    Args:
        dataset_path (str): path to h5 dataset
        frameskip (int): number of frames between observations. Starts are at least `frameskip`
            frames before their episode's terminal frame
        terminal_state (str): criteria for determining an episode's terminal frame.
            "last" ends each episode at its last frame,
            "success" at the first frame where column="success" is set in the h5 (the last frame if none)

    Returns:
        frame_starts (torch.Tensor): (S,) h5 row of each start point
        ep_starts (torch.Tensor): (S,) h5 row of the frame's episode's first frame
        frames_to_terminal (torch.Tensor): (S,) frames til the episode's terminal frame, >= frameskip
    """
    starts, lengths = read_episode_meta(dataset_path)
    terminal = lengths - 1

    if terminal_state == "success":
        with h5py.File(dataset_path, "r") as f:
            success = np.asarray(f["success"][:]).reshape(-1).astype(bool)
        for i, (start, length) in enumerate(zip(starts, lengths)):
            hits = np.flatnonzero(success[start:start + length])
            if hits.size:
                terminal[i] = hits[0]
        print(f"[data] terminal_state=success: {int((terminal == lengths - 1).sum())}/{len(starts)} "
              f"episodes end at their last frame", flush=True)
        
    frame_starts, ep_starts, frames_to_terminal = [], [], []
    for start, end in zip(starts, terminal):
        t = np.arange(max(0, end - frameskip + 1))
        frame_starts.append(start + t)
        ep_starts.append(np.full(t.size, start))
        frames_to_terminal.append(end - t)

    return tuple(
        torch.from_numpy(np.concatenate(a).astype(np.int64)) for a in (frame_starts, ep_starts, frames_to_terminal)
    )

def load_frames(
    dataset_path: str, views: list[str], data_source: str = "h5", decomp_dir: str = "", load_in_ram: bool = False
):
    """
    "Loads" frames from an h5 dataset or decompressed copy as an indexable tensor or interface class

    Args:
        dataset_path (str): path to h5 dataset
        views (list[str]): h5 image columns, one per camera
        data_source (str): "h5" decompresses each frame from the h5 when a sample reads it; "decomp"
            reads the uncompressed copy scripts/decompress_h5.py wrote under `decomp_dir`
        decomp_dir (str): the --decomp_dir given to scripts/decompress_h5.py
        load_in_ram (bool): decomp only: read the whole copy into memory at startup, instead of
            reading pieces from disk as samples need them

    Returns:
        frames: a tensor or interface class (H5Frames | MultiViewFrames) which can logically be thought of
            as a (total_steps, 3, H, W) tensor of frames for one camera, (total_steps, num_views, 3, H, W) for multi-view
    """
    assert data_source in ("h5", "decomp"), f"unknown data_source: {data_source}"
    if data_source == "h5":
        assert not load_in_ram, "--load_in_ram applies to --data_source decomp"
        cameras = [H5Frames(dataset_path, v) for v in views]
    else:
        assert decomp_dir, "--data_source decomp needs --decomp_dir"
        stem = os.path.splitext(os.path.basename(dataset_path))[0]
        cameras = []
        for v in views:
            path = os.path.join(decomp_dir, stem, f"{v}.npy")
            assert os.path.isfile(path), (f"{path} is missing: run scripts/decompress_h5.py --dataset_path "
                                          f"{dataset_path} --decomp_dir {decomp_dir} --views {','.join(views)}")
            
            frames = torch.from_numpy(np.load(path, mmap_mode=None if load_in_ram else "c"))
            assert frames.shape == H5Frames(dataset_path, v).shape, \
                f"{path} {tuple(frames.shape)} does not match the h5 and must be rebuilt"
            cameras.append(frames)

    return cameras[0] if len(cameras) == 1 else MultiViewFrames(cameras)


##################
#    Datasets    #
##################
class TrajectoryDataset(Dataset):
    """
    Each sample from this dataset contains
    - a start frame
    - its history + mask indicating which frames (if any) are padded
    - an action chunk + valid mask
    - the next state(s) every `frameskip` frames + valid mask
    - a goal frame + normalized horizon if any

    Args:
        frames: frames from `load_frames`
        actions (torch.Tensor): (N, action_dim) z-scored actions from `read_actions`
        frame_starts (torch.Tensor): (S,) h5 row of each start point, from `get_start_indices`
        ep_starts (torch.Tensor): (S,) h5 row of each start point's episode's first frame
        frames_to_terminal (torch.Tensor): (S,) frames from each start point to its terminal frame
        possible_starts (torch.Tensor): the start points this split samples from, as positions in `frame_starts`
        history_len (int): frames in the history, the newest being frame_start
        history_stride (int): frames between consecutive history frames
        num_actions (int): actions in the target chunk
        num_states (int): state targets, one every `frameskip` frames after frame_start
        frameskip (int): raw frames per chunk
        goal_type (str): "none"; "terminal" (the terminal frame, h_norm 0); or "sampled" (h ~ U[1, h_max]
            chunks ahead, clamped to the terminal frame, h_norm = h / h_max)
        h_max (int): the largest sampled goal horizon, in frameskip units
    """

    def __init__(self, frames, actions: torch.Tensor, frame_starts: torch.Tensor, ep_starts: torch.Tensor,
                 frames_to_terminal: torch.Tensor, possible_starts: torch.Tensor, history_len: int, history_stride: int,
                 num_actions: int, num_states: int, frameskip: int, goal_type: str, h_max: int):
        assert goal_type in ("none", "terminal", "sampled"), goal_type
        self.frames = frames
        self.actions = actions
        self.frame_starts = frame_starts
        self.ep_starts = ep_starts
        self.frames_to_terminal = frames_to_terminal
        self.possible_starts = possible_starts
        self.history_len = history_len
        self.history_stride = history_stride
        self.num_actions = num_actions
        self.num_states = num_states
        self.frameskip = frameskip
        self.goal_type = goal_type
        self.h_max = h_max

    def __len__(self) -> int:
        return self.possible_starts.numel()

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        """
        Returns:
            history (torch.Tensor): (history_len, [views,] 3, H, W) frames, oldest first
            history_pad (torch.Tensor): (history_len,) True where the history is padding
            action_target (torch.Tensor): (num_actions, action_dim)
            action_valid (torch.Tensor): (num_actions,) 0 past the terminal frame
            next_state_frames (torch.Tensor): (num_states, [views,] 3, H, W)
            next_state_valid (torch.Tensor): (num_states,) 0 past the terminal frame
            goal_frame (torch.Tensor): ([views,] 3, H, W), empty for goal_type "none"
            h_norm (torch.Tensor): () normalized goal horizon, 0 unless goal_type is "sampled"
        """
        start_idx = int(self.possible_starts[idx])
        frame_start = int(self.frame_starts[start_idx])
        ep_start = int(self.ep_starts[start_idx])
        frames_to_terminal = int(self.frames_to_terminal[start_idx])
        return (
            *self._history(frame_start, ep_start), *self._action_target(frame_start, frames_to_terminal),
            *self._next_state_targets(frame_start, frames_to_terminal), *self._goal(frame_start, frames_to_terminal)
        )

    def _history(self, frame_start: int, ep_start: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the history frames (oldest first, repeated if the history extends past the episode's start),
           and a mask indicating which frames are padding."""
        rows = [frame_start - k * self.history_stride for k in range(self.history_len - 1, -1, -1)]
        hist_raw = self.frames[torch.tensor([r for r in rows if r >= ep_start])]
        n_pad = self.history_len - hist_raw.shape[0]
        # repeat + left-pad the first frame if the history extends past the episode's start
        hist_padded = torch.cat([hist_raw[:1].expand(n_pad, *hist_raw.shape[1:]), hist_raw])
        return hist_padded, torch.arange(self.history_len) < n_pad

    def _action_target(self, frame_start: int, frames_to_terminal: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the next `num_actions` actions from `frame_start`, padded with zeros if the episode ends first,
           and a mask indicating which actions are valid."""
        n_valid = min(self.num_actions, frames_to_terminal)
        target = torch.zeros((self.num_actions, self.actions.shape[1]), dtype=torch.float32)
        target[:n_valid] = self.actions[frame_start:frame_start + n_valid].float()
        valid = torch.zeros(self.num_actions, dtype=torch.float32)
        valid[:n_valid] = 1.0
        return target, valid

    def _next_state_targets(self, frame_start: int, frames_to_terminal: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the next `num_states` frames from `frame_start` at intervals of `self.frameskip`,
           and a mask indicating which frames are valid (not past the episode's terminal frame)"""
        offsets = [q * self.frameskip for q in range(1, self.num_states + 1)]
        if not offsets:  # num_states=0 is possible e.g., a pure policy
            return torch.zeros((0, *self.frames.shape[1:]), dtype=self.frames.dtype), torch.zeros(0)
        frames = self.frames[torch.tensor([frame_start + min(o, frames_to_terminal) for o in offsets])]
        return frames, torch.tensor([1.0 if o <= frames_to_terminal else 0.0 for o in offsets])

    def _goal(self, frame_start: int, frames_to_terminal: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.goal_type == "none":
            return torch.zeros(0, dtype=torch.uint8), torch.tensor(0.0)
        if self.goal_type == "terminal":
            return self.frames[frame_start + frames_to_terminal], torch.tensor(0.0)
        h = min(
            int(torch.randint(1, self.h_max + 1, (1,)).item()), 
            max(1, frames_to_terminal // self.frameskip)
        )
        return self.frames[frame_start + min(h * self.frameskip, frames_to_terminal)], torch.tensor(h / self.h_max)
