from functools import lru_cache
from pathlib import Path

import segmentation_models_pytorch as smp
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "HP_DeepLabV3Plus" / "best_model.pt"


def build_model():
    return smp.DeepLabV3Plus(
        encoder_name="efficientnet-b3",
        encoder_weights=None,
        in_channels=1,
        classes=1,
        activation="sigmoid",
    )


@lru_cache(maxsize=2)
def load_model(checkpoint_path=str(DEFAULT_CHECKPOINT), device_name="cpu"):
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint}")

    device = torch.device(device_name)
    model = build_model().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


def inference_model(input_array, checkpoint_path=DEFAULT_CHECKPOINT, device="cpu", model=None):
    """Run heatmap inference for one displacement-frequency matrix."""
    torch_model = model or load_model(str(checkpoint_path), str(device))
    torch_device = next(torch_model.parameters()).device

    tensor = torch.from_numpy(input_array).unsqueeze(0).unsqueeze(0).float().to(torch_device)
    with torch.inference_mode():
        pred = torch_model(tensor)
    return pred.squeeze().cpu().numpy()
