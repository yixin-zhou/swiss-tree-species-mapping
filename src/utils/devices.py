"""Resolve command-line GPU indices within the devices visible to this process."""
import os


def select_gpus(value):
    """Return requested GPU indices and their CUDA visibility string, without CUDA imports."""
    if value is None:
        return None, None
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part.isdecimal() for part in parts):
        raise ValueError("--gpus expects GPU indices such as 1 or 0,2.")
    indices = [int(part) for part in parts]
    if len(indices) != len(set(indices)):
        raise ValueError("--gpus must not contain duplicate indices.")
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    if inherited is None:
        selected = [str(index) for index in indices]
    else:
        visible = [part.strip() for part in inherited.split(",") if part.strip()]
        if not visible or visible == ["-1"] or max(indices) >= len(visible):
            raise ValueError("--gpus indices must refer to the GPUs visible in the current environment.")
        selected = [visible[index] for index in indices]
    return indices, ",".join(selected)
