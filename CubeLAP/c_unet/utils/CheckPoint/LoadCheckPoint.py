import torch

def LoadCheckPoint(model, checkpoint_path, old_encoder_prefix="unet.encoder."):
    """
    Loads encoder weights from a checkpoint into the model.

    Args:
        model (torch.nn.Module): The model into which weights should be loaded.
        checkpoint_path (str): Path to the checkpoint file.
        old_encoder_prefix (str): Prefix used in the checkpoint for encoder weights.

    Returns:
        dict: A summary of missing and unexpected keys.
    """
    print(f"LOAD_FROM_CHECKPOINTS=True detected. Loading weights from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    full_state_dict = checkpoint.get('state_dict', checkpoint)

    encoder_state_dict = {}
    for key, value in full_state_dict.items():
        if key.startswith(old_encoder_prefix):
            new_key = key[len("unet."):]  # Keep "encoder." prefix
            encoder_state_dict[new_key] = value

    if not encoder_state_dict:
        print(f"Warning: No weights found in the checkpoint with the prefix '{old_encoder_prefix}'.")
        return {"missing_keys": [], "unexpected_keys": []}

    missing_keys, unexpected_keys = model.load_state_dict(encoder_state_dict, strict=False)

    print("Weights loaded successfully.")
    print("\n--- Weight Loading Analysis ---")

    if missing_keys:
        print(
            f"ℹInfo: {len(missing_keys)} keys were found in the new model but not in the checkpoint (expected for new/extended layers):")
        for k in missing_keys[:5]: print(f"     - {k}")

    if unexpected_keys:
        print(
            f"⚠Warning: {len(unexpected_keys)} unexpected keys were found. Please double-check the key mapping logic:")
        for k in unexpected_keys[:5]: print(f"     - {k}")

    if not unexpected_keys and missing_keys:
        print("Great! No unexpected keys were loaded, and missing keys are likely due to model extensions.")
    elif not unexpected_keys and not missing_keys:
        print("Perfect! All keys matched exactly.")

    return {
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys
    }
