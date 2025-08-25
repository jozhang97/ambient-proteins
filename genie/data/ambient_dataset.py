from pathlib import Path
import random

import pandas as pd
import numpy as np

from genie.data.dataset import GenieDataset
from genie.utils.feat_utils import (
    create_np_features_from_pdb,
    pad_np_features,
)

class AmbientDataset(GenieDataset):
    """
    Dataset for ambient training.
    Derived from torch.utils.data.Dataset.
    """

    def __init__(
        self,
        dataset_info,
        min_n_res,
        max_n_res,
        max_n_chain,
        motif_prob,
        motif_min_pct_res,
        motif_max_pct_res,
        motif_min_n_seg,
        motif_max_n_seg,
    ):
        """
        Initialize dataset.

        Args:
            dataset_info:
                A dictionary on dataset information containing
                -	a key named 'datadir', which stores the data directory
                -	a key named 'names', which store a list of structure names
            min_n_res:
                Minimum number of residues in a structure.
            max_n_res:
                Maximum number of residues in a structure.
            max_n_chain:
                Maximum number of chains in a structure.
            motif_prob:
                Percentage of motif-conditional training tasks.
            motif_min_pct_res:
                Minimum percentage of residues (out of the total sequence length of
                the input structure) to be defined as motif residues.
            motif_max_pct_res:
                Maximum percentage of residues (out of the total sequence length of
                the input structure) to be defined as motif residues.
            motif_min_n_seg:
                Minimum number of motif segments.
            motif_max_n_seg:
                Maximum number of motif segments.
        """
        print(f"Initializing AmbientDataset with min_n_res={min_n_res}, max_n_res={max_n_res}, max_n_chain={max_n_chain}")
        super(GenieDataset, self).__init__()
        self.min_n_res = min_n_res
        self.max_n_res = max_n_res
        self.max_n_chain = max_n_chain

        # Motif-specific parameters
        self.motif_prob = motif_prob
        self.motif_min_pct_res = motif_min_pct_res
        self.motif_max_pct_res = motif_max_pct_res
        self.motif_min_n_seg = motif_min_n_seg
        self.motif_max_n_seg = motif_max_n_seg

        print(f"Motif parameters: prob={motif_prob}, min_pct_res={motif_min_pct_res}, max_pct_res={motif_max_pct_res}, min_n_seg={motif_min_n_seg}, max_n_seg={motif_max_n_seg}")

        # Create filepaths
        self.filepaths = self._get_filepaths(dataset_info)
        print(f"Structures in the dataset: {len(self.filepaths)}")

        print(f"Generating random noise for {len(self)} structures to make ambient proteins")
        self.random_noise = np.random.randn(
            len(self), self.max_n_res, 3
        )

    def __getitem__(self, idx):
        """
        Returns a feature dictionary for a structure with the given index in the
        training dataset. Each value in the feature dictionary is padded accordingly
        (based on the maximum number of residues and the maximum number of chains) to
        ensure the successful construction of a batched feature dictionary.

        Args:
            idx:
                Index of the structure in the training dataset, that is, index in
                the 'filepaths' parameter.

        Returns:
            np_features:
                A feature dictionary containing information on an input structure
                (padded to a total sequence length of N), including
                -	aatype:
                        [N, 20] one-hot encoding on amino acid types
                -	num_chains:
                        [1] number of chains in the structure
                -	num_residues:
                        [1] number of residues in the structure
                -	num_residues_per_chain:
                        [1] an array of number of residues by chain
                -	atom_positions:
                        [N, 3] an array of Ca atom positions
                -	residue_mask:
                        [N] residue mask to indicate which residue position is masked
                -	residue_index:
                        [N] residue index (started from 0)
                -	chain_index:
                        [N] chain index (started from 0)
                -	fixed_sequence_mask:
                        [N] mask to indicate which residue contains conditional
                        sequence information
                -	fixed_structure_mask:
                        [N, N] mask to indicate which pair of residues contains
                        conditional structural information
                -	fixed_group:
                        [N] group index to indicate which group the residue belongs to
                        (useful for specifying multiple functional motifs)
                -	interface_mask:
                        [N] deprecated and set to all zeros.
        """
        # Load filepath
        filepath = self.filepaths[idx]

        # Load features
        np_features = create_np_features_from_pdb(filepath)

        # Make sure that ambient noise is fixed for each structure
        np_features["noise"] = self.random_noise[idx][: np_features["num_residues"]]

        # Update masks
        if np.random.random() <= self.motif_prob:
            np_features = self._update_motif_masks(np_features)

        # Pad
        np_features = pad_np_features(np_features, self.max_n_chain, self.max_n_res)
        return np_features

    def __len__(self):
        return super().__len__()
