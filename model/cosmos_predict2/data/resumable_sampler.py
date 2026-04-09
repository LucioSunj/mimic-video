import torch
from torch.utils.data import Dataset, DistributedSampler


class ResumableDistributedSampler(DistributedSampler):
    def __init__(self, dataset: Dataset, num_replicas: int, rank: int, shuffle: bool, seed: int = 0):
        super().__init__(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=True,
        )

        self.start_iter = 0

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.seed)
            indices = torch.randperm(len(self.dataset), generator=g)[: self.total_size][self.rank :: self.num_replicas]
        else:
            indices = torch.arange(
                self.rank * self.num_samples,
                (self.rank + 1) * self.num_samples,
                dtype=torch.int64,
            )

        if not len(indices) == self.num_samples:
            raise AssertionError

        indices = indices[self.start_iter :]
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples - self.start_iter

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_start_iter(self, start_iter: int) -> None:
        assert start_iter < self.num_samples
        self.start_iter = start_iter
