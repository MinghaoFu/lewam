from torch.utils.data import Dataset


class BaseSource(Dataset):
    """Backend adapter base. Every backend emits ONE common batch/obs-dict format so
    the model never sees a backend-specific dataset type.

    Batch fields are keyed by the modality spec's `from` values (entry name if `from`
    is omitted). A field read by several spec entries carries the sorted union of
    their index windows, ascending in time; the algorithm slices each entry's window
    back out. Float columns are float32.

    `provided_modalities` advertises what this source supplies, so an experiment can
    fail fast at build time if the algorithm needs a field the data lacks.
    """

    provided_modalities: set[str] = set()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
