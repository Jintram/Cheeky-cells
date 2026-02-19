
# %% ###########################################################################

# Functions to help apply a model to input images
# 



# %% ###########################################################################


def resolve_device(requested_device: str) -> str:
    # Keep mps default but fall back to cpu when needed
    if requested_device == 'mps' and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'