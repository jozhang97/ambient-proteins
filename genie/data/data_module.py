from pathlib import Path
from typing import Union, Optional
from datetime import datetime
from time import time
from traceback import format_exc

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from tqdm import tqdm

from genie.data.dataset import GenieDataset
from genie.utils.feat_utils import summarize_pdb

class GenieDataModule(LightningDataModule):
    """
    Pytorch Lightning data module for Genie 2 training.
    Derived from pytorch_lightning.LightningDataModule.

    Key functions:
        -   filter input dataset based on the number of residues
        -   randomly split input dataset into a training and a validation dataset
        -   create a training dataset and a validation dataset by writing the pdb 
            names into train.txt and validation.txt respectively
        -   initialize datasets and dataloaders based on pdb names.
    """

    def __init__(
        self,
        name: str,
        rootdir: Path,
        datadir: Path,
        min_n_res: int,
        max_n_res: int,
        max_n_chain: int,
        validation_split: Union[int, float],
        batch_size: int,
        motif_prob: float,
        motif_min_pct_res: float,
        motif_max_pct_res: float,
        motif_min_n_seg: int,
        motif_max_n_seg: int,
        min_plddt: int,
        max_plddt: int = 101,
        max_samples: Optional[int] = None,
        wandb_group=None,
        dataset_base_name: str = "",
        train_dataset: Optional[Path] = None,
        val_dataset: Optional[Path] = None,
        cached_data: Optional[Path] = None,
        overwrite: bool = False,
        use_cache: bool = True,
        diffusion_config: Optional[dict] = None,
        **kwargs,
    ):
        """
        Initialize data module.

        Args:
            name:
                Name of the training run.
            rootdir:
                Root directory.
            datadir:
                Data directory.
            min_n_res:
                Minimum number of residues in a structure.
            max_n_res:
                Maximum number of residues in a structure.
            max_n_chain:
                Maximum number of chains in a structure.
            validation_split:
                Either the number of structures in the validation dataset (by
                specifying a positive integer), or the percentage of structures
                from the dataset to be used for validation (by specifying a
                float between 0 and 1).
            batch_size:
                Number of structures in a training batch.
            motif_prob:
                Percentage of motif-conditional training tasks.
            motif_min_pct_res:
                Minimum percentage of residues (out of the total sequence length
                of the input structure) to be defined as motif residues.
            motif_max_pct_res:
                Maximum percentage of residues (out of the total sequence length
                of the input structure) to be defined as motif residues.
            motif_min_n_seg:
                Minimum number of motif segments.
            motif_max_n_seg:
                Maximum number of motif segments.
            min_plddt:
                Minimum average pLDDT score.
            max_plddt:
                Maximum average pLDDT score.
            max_samples:
                Maximum number of samples to use.
            diffusion_config:
                Configuration for the diffusion model.
            training_config:
                Configuration for the training process.
        """
        super(GenieDataModule, self).__init__()

        # Base parameters
        self.name = name
        self.rootdir = Path(rootdir)
        self.min_n_res = min_n_res
        self.max_n_res = max_n_res
        self.max_n_chain = max_n_chain
        self.validation_split = validation_split
        self.batch_size = batch_size

        # Dataset parameters
        self.datadir = Path(datadir)

        # Additional parameters related to motif-conditional training task
        self.motif_prob = motif_prob
        self.motif_min_pct_res = motif_min_pct_res
        self.motif_max_pct_res = motif_max_pct_res
        self.motif_min_n_seg = motif_min_n_seg
        self.motif_max_n_seg = motif_max_n_seg

        self.min_plddt = min_plddt
        self.max_plddt = max_plddt
        self.max_samples = max_samples

        self.train_dataset: Optional[Path] = None
        self.val_dataset: Optional[Path] = None
        self.dataset_base_name: Optional[str] = None
        self.cached_data = cached_data
        self.train_dataset_built = False
        self.val_dataset_built = False
        self.overwrite = overwrite or not use_cache
        self.use_cache = use_cache

        if any([train_dataset, val_dataset, dataset_base_name]):
            self._process_external_datasets(
                train_dataset, val_dataset, dataset_base_name
            )

        self.diffusion_config = diffusion_config

    def setup(
        self,
        stage=None,
        dataset_base_name: str = "",
        threads: int = 72,
        cached_data: Optional[Path] = None,
        train_dataset: Optional[Path] = None,
        use_cache: bool = True,
        overwrite: Optional[bool] = None,  # None behaves different than False
        **kwargs,
    ) -> None:
        """
        Set up data module before training.

        Args:
                stage:
                        Stage name required by LightningDataModule. Default to None.
        """

        self.use_cache = use_cache

        if not isinstance(cached_data, Path) and not isinstance(self.cached_data, Path):
            if isinstance(train_dataset, Path):
                self._process_external_datasets(
                    train_dataset=train_dataset, overwrite=overwrite, **kwargs
                )
            elif isinstance(self.train_dataset, Path):
                self._process_external_datasets(
                    train_dataset=self.train_dataset, overwrite=overwrite, **kwargs
                )

            elif isinstance(dataset_base_name, str) and len(dataset_base_name) > 0:
                self._process_external_datasets(
                    dataset_base_name=dataset_base_name, overwrite=overwrite, **kwargs
                )
            elif (
                isinstance(self.dataset_base_name, str)
                and len(self.dataset_base_name) > 0
            ):
                self._process_external_datasets(
                    dataset_base_name=dataset_base_name, overwrite=overwrite, **kwargs
                )

            if self.train_dataset_built:
                # # Check if the training dataset and validation dataset are created
                # # This ensures that subsequent training runs utilize the same training-
                # # validation split on the dataset.
                print(f"Using pregenerated train_dataset: {self.train_dataset}")
                return

        # TODO should I move this into _process_external_datasets by removing > 0 str filter?
        if self.train_dataset is None:
            self.dataset_base_name = dataset_base_name if dataset_base_name else f""
            self.train_dataset = (
                self.rootdir / f"{self.name}/{self.dataset_base_name}train.txt"
            )
            self.val_dataset = (
                self.rootdir / f"{self.name}/{self.dataset_base_name}validation.txt"
            )
            self.train_dataset_built = False
            self.val_dataset_built = False

        if isinstance(cached_data, Path) and not cached_data.is_file():
            cached_data = (
                cached_data if cached_data.is_file() else self.datadir / cached_data
            )

        elif isinstance(self.cached_data, Path):
            cached_data = (
                self.cached_data
                if self.cached_data.is_file()
                else self.datadir / self.cached_data
            )

        if cached_data is None or not cached_data.is_file():
            prefix = self.dataset_base_name if self.dataset_base_name else f"cached_"
            max_samples = "none" if self.max_samples is None else self.max_samples
            cached_data = (
                self.datadir
                / f"{prefix}{max_samples}_plddt{self.min_plddt}-{self.max_plddt}_nres{self.min_n_res}-{self.max_n_res}.csv"
            )

        print(f"Starting name extraction...")
        print(f"Using pLLDT threshold: {self.min_plddt} {self.max_plddt}")
        print(f"Using n_res threshold: {self.min_n_res} {self.max_n_res}")

        if self.use_cache and cached_data.is_file():
            # Load structure paths by filtering a cached file of path, plddt, n_res with pandas
            print(f"Producing dataset from cached data: {cached_data.name}...")
            names = self._load_cached_st_files(cached_data)

        else:
            # filter structures based on max_samples, min_plddt, max_plddt, min_n_res, max_n_res
            # Filtering for input structures is customizable via the function _validate.
            if cached_data.is_file():
                print(f"Ignoring cached data file: {cached_data.resolve()}")

            print(f"looping through structures in {self.datadir}...")
            names = self._fetch_st_files(
                self.datadir, overwrite, threads
            )  # TODO refactor

        if self.validation_split is not None:
            # Split structures into a training set and a validation set
            # Splitting of input structures is customizable via the function _split.
            train_names, validation_names = self._split(names)
            self.train_dataset = (
                self.train_dataset.parent
                / f"{self.train_dataset.stem}_{len(train_names)}.txt"
            )
            self.val_dataset = (
                self.val_dataset.parent
                / f"{self.val_dataset.stem}_{len(validation_names)}.txt"
            )

            # Save structure names into files
            self._save_st_files(train_names, self.train_dataset)
            self._save_st_files(validation_names, self.val_dataset)

            self.train_dataset_built = True
            self.test_dataset_built = True

        else:
            # Save structure names into files
            train_names = names
            self.train_dataset = (
                self.train_dataset.parent
                / f"{self.train_dataset.stem}_{len(train_names)}.txt"
            )

            self._save_st_files(train_names, self.train_dataset)

            self.train_dataset_built = True

    def train_dataloader(self, ddp: bool = False, shuffle: bool = True) -> DataLoader:
        """
        Set up dataloader for training.

        Returns:
                An instance of torch.utils.data.DataLoader.
        """
        # Create dataset information
        assert self.train_dataset_built, "run setup method first"
        dataset_info = {
            "datadir": self.datadir,
            "names": self._load_st_filenames(self.train_dataset),
        }

        # Create dataset
        dataset = GenieDataset(
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

        # Create dataloader
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle if ddp is False else False,
            sampler=(
                torch.utils.data.distributed.DistributedSampler(
                    dataset, shuffle=shuffle
                )
                if ddp is not False
                else None
            ),
        )

    ############################
    ###   Helper Functions   ###
    ############################

    def _process_external_datasets(
        self,
        train_dataset: Optional[Path] = None,
        val_dataset: Optional[Path] = None,
        dataset_base_name: str = Optional[None],
        train_suffix: str = "train.txt",
        val_suffix: str = "validation.txt",
        use_default: bool = False,
        overwrite: Optional[bool] = None,  # None behaves different than False
        **kwargs,
    ) -> None:

        overwrite = self.overwrite if overwrite is None else overwrite
        if not self.use_cache:
            overwrite = True

        # Define filepaths for training and validation dataset
        # These files keep track of the set of structures used in the training
        # and validation process respectively, by recording their names.
        self.train_dataset: Optional[Path] = None
        self.val_dataset: Optional[Path] = None
        self.train_dataset_built = False
        self.val_dataset_built = False

        if train_dataset:
            assert (
                dataset_base_name is None
            ), "train_dataset and dataset_base_name are mutually exclusive"

        elif dataset_base_name:
            assert (
                train_dataset is None
            ), "train_dataset and dataset_base_name are mutually exclusive"

        if isinstance(train_dataset, Path):
            if not train_dataset.is_file():
                train_dataset = self.rootdir / f"{self.name}/{train_dataset.name}"
            self.train_dataset = train_dataset

        elif not train_dataset is None:
            raise ValueError(
                f"train_dataset must be a Path object: {type(train_dataset)}"
            )

        if isinstance(val_dataset, Path):
            if not val_dataset.is_file():
                val_dataset = self.rootdir / f"{self.name}/{val_dataset.name}"
            self.val_dataset = val_dataset

        elif not val_dataset is None:
            raise ValueError(f"val_dataset must be a Path object: {type(val_dataset)}")

        if isinstance(dataset_base_name, str) or use_default:
            self.dataset_base_name = dataset_base_name if dataset_base_name else f""
            self.train_dataset = (
                self.rootdir / f"{self.name}/{self.dataset_base_name}{train_suffix}"
            )
            self.val_dataset = (
                self.rootdir / f"{self.name}/{self.dataset_base_name}{val_suffix}"
            )

        # elif dataset_base_name is None:
        #     raise ValueError(f"dataset_base_name must be a string: {type(dataset_base_name)} {dataset_base_name}")

        if isinstance(self.train_dataset, Path):
            if self.train_dataset.is_file() and self.train_dataset.stat().st_size > 0:
                self.train_dataset_built = True
            else:
                self.train_dataset_built = False
                self.train_dataset = (
                    self.rootdir / f"{self.name}/{self.dataset_base_name}{train_suffix}"
                )
                print(
                    f"train_dataset not found or empty: {self.train_dataset.resolve()}"
                )
        else:
            print(f"train_dataset not provided")

        if isinstance(self.val_dataset, Path):
            if self.val_dataset.is_file() and self.val_dataset.stat().st_size > 0:
                self.val_dataset_built = True
            else:
                self.val_dataset_built = False
                print(
                    f"val_dataset not found or empty: {self.val_dataset.resolve()}"
                )
        else:
            print(f"val_dataset not provided")

        # Overwrites the final structure dataset used for train/val not the cached data file
        if overwrite:
            self.train_dataset_built = False
            self.val_dataset_built = False

        if self.train_dataset_built:
            print(
                f"Using precomputed train_dataset: {self.train_dataset.resolve()}"
            )
        else:
            print(f"Not using precomputed train_dataset")

        if self.val_dataset_built:
            print(f"Using precomputed val_dataset: {self.val_dataset.resolve()}")
        else:
            print(f"Not using precomputed val_dataset")

    def _load_st_filenames(self, dataset: Path) -> list[Path]:
        """
        Load structure names.

        Args:
                filepath:
                        Path to the file containing a list of structure names.

        Returns:
                names:
                        A list of structure names.
        """
        assert isinstance(dataset, Path), type(dataset)
        assert dataset.is_file(), dataset.resolve()

        with dataset.open("rt") as file:
            st_files = [Path(line.strip()) for line in file]

        return st_files

    def _save_st_files(self, st_files: list[Path], out_file: Path) -> None:
        """
        Save structure names.

        Args:
                names:
                        A list of structure file names.
                filepath:
                        Path to output file that stores this list of structure names.
        """
        assert len(st_files) > 0, len(st_files)
        assert isinstance(out_file, Path), type(out_file)

        out_file.parent.mkdir(0o774, parents=True, exist_ok=True)

        with out_file.open("wt") as f:
            f.write("\n".join(map(str, st_files)))  # save relative paths

        print(f"Saved {len(st_files)} structures to {out_file.resolve()}")

    def _load_cached_st_files(self, dataset: Path) -> list[Path]:
        import pandas as pd

        df = pd.read_csv(dataset)
        # save average plddt to access it later
        self.avg_plddt = df.set_index('name')['plddt'].to_dict()
        self.avg_plddt = {k.split('/')[-1]: v for k, v in self.avg_plddt.items()}

        names = df[
            (df["plddt"] >= self.min_plddt)
            & (df["plddt"] <= self.max_plddt)
            & (df["n_res"] >= self.min_n_res)
            & (df["n_res"] <= self.max_n_res)
        ]["name"].tolist()

        st_paths = [Path(st) for st in names]

        print(f"Loaded {len(st_paths)} structures from {dataset.resolve()}")

        return st_paths

    def _fetch_st_files(
        self,
        datadir: Optional[Path] = None,
        overwrite: bool = False,
        threads: int = 72,
        cached_dataset: Optional[str] = None,
    ) -> list[Path]:
        """
        Fetch names for structures in the data directory, that pass the set of
        filters defined in the function _validate.

        Args:
                datadir:
                        Data directory.
                cached_dataset:
                        just the name of the file and not the full path

        Returns:
                A list of Paths for all passed structures.
        """
        if datadir is None:
            datadir = self.datadir

        assert isinstance(datadir, Path)

        if cached_dataset is None:
            prefix = self.dataset_base_name if self.dataset_base_name else f"cached_"
            max_samples = "none" if self.max_samples is None else self.max_samples
            cached_dataset = f"{prefix}{max_samples}_plddt{self.min_plddt}-{self.max_plddt}_nres{self.min_n_res}-{self.max_n_res}.csv"

        out_file = datadir / cached_dataset

        if not overwrite and (out_file.is_file() and out_file.stat().st_size > 0):
            st_files = self._load_cached_st_files(out_file)
            print(
                f"Loaded dataset ({len(st_files)} structures) from {out_file.resolve()}"
            )

            return st_files

        print(
            f"creating dataset (threads: {threads}) then saving to {out_file.resolve()}..."
        )

        if not "pymp" in dir():
            import pymp

        sample_count = 0
        proteins = list(datadir.rglob("*.pdb*"))
        total_files = len(proteins)
        metadata = pymp._shared.list()
        st_files = pymp._shared.list()
        sample_count = pymp._shared._get_manager().Value(int, 0)

        t_start = time()
        with pymp.Parallel(threads) as p:
            for idx in p.xrange(total_files):
                fp = proteins[idx].resolve()
                if (
                    self.max_samples is not None
                    and sample_count.value >= self.max_samples
                ):
                    break
                try:
                    print(
                        f"Thread {p.thread_num} - Validating ({idx}): {fp.name}..."
                    )
                    is_valid, plddt, n_res = self._validate(
                        fp
                    )  # checks L > 20 and L < 256

                except Exception as e:
                    is_valid = False
                    print(
                        f"Thread {p.thread_num} - Error in {fp} ({e})\n{format_exc()}"
                    )

                else:
                    if is_valid:
                        print(
                            f"Thread {p.thread_num} - is valid:\n({sample_count.value:6d}/{self.max_samples}) plddt: {plddt:0.2f} n_res: {n_res} - {fp.resolve()}"
                        )
                        st_files.append(fp)
                        sample_count.value += 1
                        metadata.append((fp, plddt, n_res))
                        # pbar.update(1)

        with out_file.open("wt") as f:
            f.write("name,plddt,n_res\n")
            f.write(
                "\n".join(
                    [f"{pdb},{plddt},{n_res}" for pdb, plddt, n_res in list(metadata)]
                )
            )

        print(f"Wrote dataset: {out_file.resolve()}")
        runtime = time() - t_start

        print(f"Generated dataset {self.name} in {runtime:.2f} seconds")

        return list(st_files)

    def _split(self, st_files: list[Path]):
        """
        Split structures into a training dataset and a validation dataset.

        By default, the splitting is based on random selection and the parameter
        validation_split specifies either the number of validation data points or
        the percentage of the total dataset used for validation.

        Args:
                names:
                        A list of structure names to be split.

        Returns:
                train_names:
                        A list of names for structures used at the training stage.
                validation_names:
                        A list of names for structures used at the valiadtion stage.
        """
        split_idx = (
            int(len(st_files) * self.validation_split)
            if self.validation_split < 1
            else int(self.validation_split)
        )
        train_names = st_files[:-split_idx]
        validation_names = st_files[-split_idx:]
        return train_names, validation_names

    def _validate(self, pdb_f: Path, chain_idx: int = 0) -> tuple[bool, float, int]:
        """
        Filter input structure based on the minimum and maximum number of residues.

        Args:
                filepath:
                        Path to the PDB file of a structure.

        Returns:
                A boolean indicating whether the structure passes the set of filters.
                plddt float
                n_res of structure
        """
        summary = summarize_pdb(pdb_f)
        plddt_avg = sum(summary["plddt_scores"][chain_idx]) / len(
            summary["plddt_scores"][chain_idx]
        )
        return (
            (
                summary["num_residues"] >= self.min_n_res
                and summary["num_residues"] <= self.max_n_res
                and plddt_avg >= self.min_plddt
                and (self.max_plddt is None or plddt_avg <= self.max_plddt)
            ),
            round(plddt_avg, 3),
            summary["num_residues"],
        )
