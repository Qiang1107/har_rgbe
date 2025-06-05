import torch


def load_vitpose_pretrained(model: torch.nn.Module, checkpoint_path: str):
    """Load ViTPose pre-trained weights into the model's backbone.

    This function expects the checkpoint to be a dictionary containing either the
    raw state_dict or under the key ``state_dict``. Parameters that do not match
    will be ignored so that classification heads can be re-initialised.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        # remove leading 'module.' used by DataParallel
        if k.startswith("module."):
            k = k[len("module."):]
        # ensure keys are prefixed with 'backbone.' so that they match our model
        if not k.startswith("backbone."):
            new_state_dict[f"backbone.{k}"] = v
        else:
            new_state_dict[k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"Loaded pretrained weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    return missing, unexpected