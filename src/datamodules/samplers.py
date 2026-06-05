import math
from typing import Iterator, Optional

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


class RandomSubsetPerEpochSampler(Sampler[int]):
    def __init__(
        self,
        data_source,
        num_samples: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        self.data_source = data_source
        self.requested_num_samples = num_samples
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        if num_replicas is None:
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1
        if rank is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0

        self.num_replicas = num_replicas
        self.rank = rank

    @property
    def sample_count(self) -> int:
        return min(self.requested_num_samples, len(self.data_source))

    def __iter__(self) -> Iterator[int]:
        sample_count = self.sample_count
        if sample_count <= 0:
            return iter([])

        epoch = self.epoch
        self.epoch += 1

        generator = torch.Generator()
        generator.manual_seed(self.seed + epoch)

        if self.shuffle:
            indices = torch.randperm(len(self.data_source), generator=generator).tolist()[:sample_count]
        else:
            indices = list(range(sample_count))

        if self.num_replicas == 1:
            return iter(indices)

        if self.drop_last and sample_count % self.num_replicas != 0:
            total_size = math.ceil((sample_count - self.num_replicas) / self.num_replicas) * self.num_replicas
            indices = indices[:total_size]
        else:
            total_size = math.ceil(sample_count / self.num_replicas) * self.num_replicas
            padding_size = total_size - sample_count
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]

        return iter(indices[self.rank:total_size:self.num_replicas])

    def __len__(self) -> int:
        sample_count = self.sample_count
        if self.num_replicas == 1:
            return sample_count
        if self.drop_last:
            return sample_count // self.num_replicas
        return math.ceil(sample_count / self.num_replicas)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
