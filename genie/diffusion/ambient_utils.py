import torch

def broadcast_batch_tensor(batch_tensor):
    """ Takes a tensor of potential shape (batch_size) and returns a tensor of shape (batch_size, 1, 1, 1).
    """
    return batch_tensor[:, None, None, None]
    
def ambient_sqrt(x):
    """
        Computes the square root of x if x is positive, and 1 otherwise.
    """
    return torch.where(x < 0, torch.ones_like(x), x.sqrt())

def add_extra_noise_from_vp_to_vp(noisy_input, current_sigma, desired_sigma):
    """
        Adds extra noise to the input to move from current_sigma to desired_sigma.
        Args:
            noisy_input: input to add noise to
            current_sigma: current noise level
            desired_sigma: desired noise level
        Returns:
            extra_noisy: noisy input with extra noise
            noise_realization: the noise realization that was added
            done: True if the desired noise level was reached, False otherwise
    """
    scaling_coeff = ambient_sqrt((1 - desired_sigma**2) / (1 - current_sigma ** 2))
    noise_coeff = ambient_sqrt(desired_sigma ** 2 - (scaling_coeff ** 2) * current_sigma ** 2)
    noise_realization = torch.randn_like(noisy_input)
    scaling_coeff, noise_coeff, current_sigma, desired_sigma = [broadcast_batch_tensor(x) for x in [scaling_coeff, noise_coeff, current_sigma, desired_sigma]]
    extra_noisy = scaling_coeff * noisy_input + noise_coeff * noise_realization
    # when we are trying to move to a lower noise level, just do nothing
    extra_noisy = torch.where(current_sigma > desired_sigma, noisy_input, extra_noisy)
    return extra_noisy, noise_realization, (current_sigma <= desired_sigma)[:, 0, 0, 0]
