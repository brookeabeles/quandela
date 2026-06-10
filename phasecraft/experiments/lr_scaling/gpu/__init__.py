"""GPU-accelerated LR-QAOA training and benchmark pipeline."""

from .backend import DeviceInfo, detect_device

__all__ = ["DeviceInfo", "detect_device"]
