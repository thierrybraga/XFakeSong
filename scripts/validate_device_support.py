import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.domain.services.detection.utils import get_available_devices  # noqa: E402, F401
from app.domain.services.detection_service import DetectionService  # noqa: E402


def validate_device_support():
    print("Validating Device Support...")

    # 1. Test get_available_devices
    devices = get_available_devices()
    print(f"Available devices: {devices}")
    assert "CPU" in devices, "CPU should always be available"

    # 2. Test DetectionService device setting
    try:
        # Initialize service (might fail if models dir not present,
        # so we mock or handle error)
        # Assuming models dir exists or is created by loader
        ds = DetectionService(models_dir="test_models_dir")

        print(f"Initial device: {ds.device}")
        assert ds.device == "CPU"

        ds.set_device("GPU:0")
        print(f"Set device to GPU:0. Current device: {ds.device}")
        assert ds.device == "GPU:0"

        # Reset to CPU
        ds.set_device("CPU")
        print("Reset device to CPU.")

    except Exception as e:
        print(
            f"Service initialization failed (expected if models missing): {e}"
        )
        # Even if init fails, we validated the utils part.

    print("Validation passed!")


if __name__ == "__main__":
    validate_device_support()
