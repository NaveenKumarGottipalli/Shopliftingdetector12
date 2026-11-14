"""Frozen CLIP and SigLIP feature extraction with exact OpenAI CLIP normalization constants."""
from pathlib import Path
import numpy as np
import torch

# Exact OpenAI CLIP normalization constants (green channel mean = 0.4578275)
OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]
SIGLIP_MEAN      = [0.5, 0.5, 0.5]
SIGLIP_STD       = [0.5, 0.5, 0.5]

def unwrap_features(output, model, kind: str) -> torch.Tensor:
    """Unwrap tensor or ModelOutput across transformers versions."""
    if torch.is_tensor(output):
        return output
    t = getattr(output, "pooler_output", None)
    if t is None:
        t = output.last_hidden_state[:, 0]
    dim = getattr(model.config, "projection_dim", None)
    proj = getattr(model, "visual_projection" if kind == "image" else "text_projection", None)
    if dim and t.shape[-1] != dim and proj is not None:
        t = proj(t)
    return t

class FrameDataset(torch.utils.data.Dataset):
    def __init__(self, clip_ids, frames_dir: Path):
        self.ids = list(clip_ids)
        self.frames_dir = Path(frames_dir)
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, i):
        return torch.from_numpy(np.load(self.frames_dir / f"{self.ids[i]}.npy"))
