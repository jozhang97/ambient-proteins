from pathlib import Path
from typing import Optional
from itertools import cycle
import random
import math
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler


class TimeSampler(Sampler):

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        window_size: float = 0.5,
        diffusion_config: Optional[dict] = None,
        avg_plddt: Optional[dict] = None,
        training_config: Optional[dict] = None,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)

        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size
        self.found_times = {}  # the found timestep for each protein
        self.sampled_times = {}  # the sampled timestep for each protein
        self.high_plddt_mask = {}  # whether this is an absolutely great protein
        self.diffusion_config = diffusion_config
        self.avg_plddt = avg_plddt
        self.training_config = training_config
        print(f"{self.diffusion_config['plddt_to_timestep']=}")

    def plddt_to_timestep(self, found_plddt):
        for threshold, timestep in self.diffusion_config["plddt_to_timestep"].items():
            if found_plddt > threshold:
                return timestep
        return 1000

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            # this ensures consistent shuffle among the processes.
            rnd = np.random.RandomState(self.seed + self.epoch)
            rnd.shuffle(order)
            # create a local window for dynamic shuffling within the window.
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        timestep = np.random.randint(1000, size=(1,)) + 1

        indices = []
        print("Running sampler....")
        start_time = time.time()
        while len(indices) < self.num_samples:
            i = idx % order.size
            # check the timestep for this protein
            file_key = str(self.dataset.filepaths[order[i]]).split("/")[-1]  # type: ignore
            if idx % self.num_replicas == self.rank:
                if file_key not in self.avg_plddt:
                    print(f"[WARNING]: Node: {self.rank}/{self.num_replicas} Recomputing plddt for file ({idx}): {file_key}, this should not be happening")
                    print(f"[WARNING]: Node: {self.rank}/{self.num_replicas}  Number of keys in avg_pldd dict: {len(self.avg_plddt.keys())}.")
                    print(f"[WARNING]: Node: {self.rank}/{self.num_replicas}  size of dataset.filepaths: {len(self.dataset.filepaths)}.")
                    dataset_item = self.dataset[order[i]]
                    found_plddt = np.sum(
                        dataset_item["plddt_scores"] * dataset_item["residue_mask"]
                    ) / np.sum(dataset_item["residue_mask"])
                    self.avg_plddt[file_key] = found_plddt
                else:
                    found_plddt = self.avg_plddt[file_key]

                found_timestep = self.plddt_to_timestep(found_plddt)
                self.found_times[file_key] = found_timestep

            if (idx % self.num_replicas == self.rank) and (
                timestep >= found_timestep + self.diffusion_config["dt_buffer"]
            ):
                self.sampled_times[file_key] = timestep
                self.high_plddt_mask[file_key] = found_timestep <= 0
                yield order[i]
                indices.append(order[i])
                timestep = np.random.randint(1000, size=(1,)) + 1

            if window >= 2:
                # if dynamic shuffling is enabled, then global order can change and the indices that are
                # getting assigned to each GPU can change as well.
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]

            idx += 1

        print(f"[{self.rank}]: Completed sampling")
        print(f"Sampling Runtime: {time.time() - start_time:.2f}s")
        self.epoch += 1
        print(f"Length of set of indices: {len(set(indices))}")

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
