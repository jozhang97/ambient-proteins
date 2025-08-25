import torch
from torch.utils.data import Dataset, DataLoader
from genie.data.ambient_dataset import AmbientDataset
from genie.data.data_module import GenieDataModule
from genie.data.sampler import TimeSampler

class AmbientDataModule(GenieDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.training_config = kwargs.get("training_config", {})

    def train_dataloader(
        self, ddp: bool = False, shuffle: bool = True
    ) -> DataLoader:
        """
        Set up dataloader for training.

        Returns:
                An instance of torch.utils.data.DataLoader.
        """
        # Create dataset information
        assert self.train_dataset_built, "run setup method first"

        names = self._load_st_filenames(self.train_dataset)
        dataset_info = {
            "datadir": self.datadir,
            "names": names,
        }

        # Create dataset
        self.dataset = AmbientDataset(
            dataset_info,
            self.min_n_res,
            self.max_n_res,
            self.max_n_chain,
            self.motif_prob,
            self.motif_min_pct_res,
            self.motif_max_pct_res,
            self.motif_min_n_seg,
            self.motif_max_n_seg,
        )

        # Need to sample time uniformly
        avg_plddt = self.avg_plddt if hasattr(self, 'avg_plddt') else {}
        sampler = TimeSampler(
            self.dataset,
            shuffle=shuffle,
            diffusion_config=self.diffusion_config,
            avg_plddt=avg_plddt,
            training_config=self.training_config,
        )

        dataloader = DataLoader(
            self.dataset,
            shuffle=False,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=0,
        )

        self.dataloader = dataloader
        self.sampler = sampler
        return dataloader


if __name__ == "__main__":
    pass
