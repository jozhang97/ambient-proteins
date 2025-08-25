import torch

from genie.diffusion.ddpm import DDPM
from genie.utils.affine_utils import T
from genie.utils.geo_utils import compute_frenet_frames
from genie.utils.loss import mse
from genie.utils.feat_utils import prepare_tensor_features

from genie.diffusion.ambient_utils import add_extra_noise_from_vp_to_vp


class Genie(DDPM):
    """
    An instantiation of DDPM for unconditional diffusion.
    """

    def training_step(self, batch, batch_idx):
        """
        Training iteration.

        Args:
            batch:
                A batched feature dictionary with a batch size B, where each 
                structure is padded to the maximum sequence length N. It contains 
                the following information
                    -   aatype: 
                            [B, N, 20] one-hot encoding on amino acid types
                    -   num_chains: 
                            [B, 1] number of chains in the structure
                    -   num_residues: 
                            [B, 1] number of residues in the structure
                    -   num_residues_per_chain: 
                            [B, 1] an array of number of residues by chain
                    -   atom_positions: 
                            [B, N, 3] an array of Ca atom positions
                    -   residue_mask: 
                            [B, N] residue mask to indicate which residue position is masked
                    -   residue_index: 
                            [B, N] residue index (started from 0)
                    -   chain_index: 
                            [B, N] chain index (started from 0)
                    -   fixed_sequence_mask: 
                            [B, N] mask to indicate which residue contains conditional
                            sequence information
                    -   fixed_structure_mask: 
                            [B, N, N] mask to indicate which pair of residues contains
                            conditional structural information
                    -   fixed_group:
                            [B, N] group index to indicate which group the residue belongs to
                            (useful for specifying multiple functional motifs)
                    -   interface_mask:
                            [B, N] deprecated and set to all zeros.
                    -   noise
                            [B, N, 3] pre-computed noise for ambient diffusion
            batch_idx:
                [1] Index of this training batch.

        Returns:
            loss:
                [1] Motif-weighted mean of per-residue mean squared error between the predicted 
                noise and the groundtruth noise, averaged across all structures in the batch
        """

        # Perform setup in the first run
        if not self.setup:
            self.setup_schedule()
            self.setup = True

        # Define features
        features = prepare_tensor_features(batch)

        if self.config.diffusion['ambient']:
            # use pre-sampled times
            sampler = self.trainer.datamodule.sampler
            file_names = [x.split("/")[-1] for x in batch["filepath"]]
            s = torch.tensor([sampler.sampled_times[x] for x in file_names], device=self.device).view(-1)
            high_plddt_mask = torch.tensor([sampler.high_plddt_mask[x] for x in file_names], device=self.device)
            found_times = torch.tensor([sampler.found_times[x] for x in file_names], device=self.device)
            found_times = torch.clamp(found_times, min=0)
        else:
            s = torch.randint(
                self.config.diffusion['n_timestep'],
                size=(features['atom_positions'].shape[0],)
            ).to(self.device) + 1

        # Sample noise
        z = torch.randn_like(features['atom_positions']) * features['residue_mask'].unsqueeze(-1)

        # Apply noise
        trans_s = self.sqrt_alphas_cumprod[s].view(-1, 1, 1) * features['atom_positions'] + \
            self.sqrt_one_minus_alphas_cumprod[s].view(-1, 1, 1) * z

        if self.config.diffusion['ambient']:
            # apply ambient noise on low-quality proteins
            ambient_x_t, ambient_z, ambient_loss_ratios, x_tnature, desired_std, trust_level_std = \
                self._apply_ambient_noise(features, s, trans_s, z, found_times)
            trans_s = torch.where(high_plddt_mask.view(-1, 1, 1), trans_s, ambient_x_t)
            z = torch.where(high_plddt_mask.view(-1, 1, 1), z, ambient_z)

        rots_s = compute_frenet_frames(
            trans_s,
            features['chain_index'],
            features['residue_mask']
        )
        ts = T(rots_s, trans_s)

        # Predict noise
        output = self.model(ts, s, features)

        pred = output["z"]
        tgt = z
        if self.config.diffusion['ambient']:
            # rescale pred, tgt for ambient loss
            ambient_pred, ambient_tgt = self._ambient_loss_scaling(pred, trans_s, x_tnature, desired_std, trust_level_std)

            if self.config.diffusion["always_ambient_for_high_noises"]:
                # train clean proteins with ambient loss
                mask = (s <= self.config.diffusion['t_nature']).view(-1, 1, 1)
            else:
                # train clean proteins with normal loss
                mask = high_plddt_mask.view(-1, 1, 1)

            pred = torch.where(mask, pred, ambient_pred)
            tgt = torch.where(mask, tgt, ambient_tgt)

        # Compute masks
        condition_mask = features['residue_mask'] * features['fixed_sequence_mask']
        infill_mask = features['residue_mask'] * ~features['fixed_sequence_mask']

        # Compute condition and infill losses
        condition_losses = mse(pred, tgt, condition_mask, aggregate='sum')
        infill_losses = mse(pred, tgt, infill_mask, aggregate='sum')

        # Compute weighted losses
        unweighted_losses = (condition_losses + infill_losses) / features['num_residues']
        weighted_losses = (self.config.training['condition_loss_weight'] * condition_losses + infill_losses) / \
            (self.config.training['condition_loss_weight'] * torch.sum(condition_mask, dim=-1) + torch.sum(infill_mask, dim=-1))

        if self.config.diffusion['ambient']:
            # We only use the ambient loss ratio on the bad protein (low_plddt)
            scaling = torch.where(
                high_plddt_mask,
                torch.ones_like(ambient_loss_ratios),
                ambient_loss_ratios,
            )
            weighted_losses *= scaling
            unweighted_losses *= scaling

        # Aggregate
        unweighted_loss = torch.mean(unweighted_losses)
        weighted_loss = torch.mean(weighted_losses)
        self.log('unweighted_loss', unweighted_loss, on_step=True, on_epoch=True)
        self.log('weighted_loss', weighted_loss, on_step=True, on_epoch=True)

        # Log
        batch_mask = torch.sum(condition_mask, dim=-1) > 0
        condition_losses = condition_losses / torch.sum(condition_mask, dim=-1)
        infill_losses = infill_losses / torch.sum(infill_mask, dim=-1)
        for i in range(batch_mask.shape[0]):
            if batch_mask[i]:
                self.log('motif_mse_loss', condition_losses[i], on_step=True, on_epoch=True)
                self.log('scaffold_mse_loss', infill_losses[i], on_step=True, on_epoch=True)
            else:
                self.log('unconditional_mse_loss', infill_losses[i], on_step=True, on_epoch=True)

        return weighted_loss

    def _apply_ambient_noise(self, features, s, trans_s, z, found_times):

        trust_level_std = self.sqrt_one_minus_alphas_cumprod[found_times]  # sigma t_nature
        desired_std = self.sqrt_one_minus_alphas_cumprod[s]  # sigma t

        # compute loss scaling factor
        ambient_loss_ratios = (
            2
            * (desired_std**2 - trust_level_std**2)
            / torch.sqrt(desired_std**2 * (1 - trust_level_std**2))
        )
        ambient_loss_ratios = 1 / ambient_loss_ratios

        # x_0 -> x_t_nature
        x_tnature = (
            self.sqrt_alphas_cumprod[found_times].view(-1, 1, 1) * features['atom_positions'] +
            trust_level_std.view(-1, 1, 1) * features['noise']
        )

        # x_t_nature -> x_t
        x_tnature_expanded = x_tnature[:, None]  # ambient_utils expects [B, 1, N, 3]
        x_t, ambient_z, _ = add_extra_noise_from_vp_to_vp(
            x_tnature_expanded, trust_level_std, desired_std
        )
        x_t = x_t[:, 0]
        ambient_z = ambient_z[:, 0]

        return x_t, ambient_z, ambient_loss_ratios, x_tnature, desired_std, trust_level_std

    def _ambient_loss_scaling(self, z_pred, trans_s, x_tnature, desired_std, trust_level_std):
        # trans_s is x_t
        a_t = (desired_std**2 - trust_level_std**2) / (desired_std**2)
        a_t /= torch.sqrt(1 - trust_level_std**2)
        b_t = (trust_level_std / desired_std) ** 2
        b_t *= torch.sqrt((1 - desired_std**2) / (1 - trust_level_std**2))
        c_t = 1 - desired_std**2

        output_scaling_factor = -a_t * torch.sqrt(1 - c_t)
        noise_scaling_factor = a_t + b_t * torch.sqrt(c_t)
        pred = (
            noise_scaling_factor[:, None, None] * trans_s
            + output_scaling_factor[:, None, None] * z_pred
        )
        tgt = x_tnature * torch.sqrt(c_t)[:, None, None]

        MAX_LOSS_SCALE = 2
        loss_scaling = torch.clamp(torch.sqrt(1 / c_t), 0, MAX_LOSS_SCALE)[
            :, None, None
        ]
        pred *= loss_scaling
        tgt *= loss_scaling
        return pred, tgt
