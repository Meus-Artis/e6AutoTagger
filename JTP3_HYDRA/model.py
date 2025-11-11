from math import ceil

import torch
from torch import Tensor
from torch.nn import Identity

import timm
from timm.models import NaFlexVit

from PIL import Image

from safetensors import safe_open

from image import process_srgb, put_srgb_patch

def sdpa_attn_mask(
    patch_valid: Tensor,
    num_prefix_tokens: int = 0,
    symmetric: bool = True,
    q_len: int | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    mask = patch_valid.unflatten(-1, (1, 1, -1))

    if num_prefix_tokens:
        mask = torch.cat((
            torch.ones(
                *mask.shape[:-1], num_prefix_tokens,
                device=patch_valid.device, dtype=torch.bool
            ), mask
        ), dim=-1)

    return mask

timm.models.naflexvit.create_attention_mask = sdpa_attn_mask

def get_image_size_for_seq(
    image_hw: tuple[int, int],
    patch_size: int = 16,
    max_seq_len: int = 1024,
    max_ratio: float = 1.0,
    eps: float = 1e-5,
) -> tuple[int, int]:
    """Determine image size for sequence length constraint."""

    assert max_ratio >= 1.0
    assert eps * 2 < max_ratio

    h, w = image_hw
    max_py = int(max((h * max_ratio) // patch_size, 1))
    max_px = int(max((w * max_ratio) // patch_size, 1))

    if (max_py * max_px) <= max_seq_len:
        return max_py * patch_size, max_px * patch_size

    def patchify(ratio: float) -> tuple[int, int]:
        return (
            min(int(ceil((h * ratio) / patch_size)), max_py),
            min(int(ceil((w * ratio) / patch_size)), max_px)
        )

    py, px = patchify(eps)
    if (py * px) > max_seq_len:
        raise ValueError(f"Image of size {w}x{h} is too large.")

    ratio = eps
    while (max_ratio - ratio) >= eps:
        mid = (ratio + max_ratio) / 2.0

        mpy, mpx = patchify(mid)
        seq_len = mpy * mpx

        if seq_len > max_seq_len:
            max_ratio = mid
            continue

        ratio = mid
        py = mpy
        px = mpx

        if seq_len == max_seq_len:
            break

    assert py >= 1 and px >= 1
    return py * patch_size, px * patch_size

def process_image(img: Image.Image, patch_size: int, max_seq_len: int) -> Image.Image:
    def compute_resize(wh: tuple[int, int]) -> tuple[int, int]:
        h, w = get_image_size_for_seq((wh[1], wh[0]), patch_size, max_seq_len)
        return w, h

    return process_srgb(img, resize=compute_resize)

def patchify_image(img: Image.Image, patch_size: int, max_seq_len: int, share_memory: bool = False) -> tuple[Tensor, Tensor, Tensor]:
    patches = torch.zeros(max_seq_len, patch_size * patch_size * 3, device="cpu", dtype=torch.uint8)
    patch_coords = torch.zeros(max_seq_len, 2, device="cpu", dtype=torch.int16)
    patch_valid = torch.zeros(max_seq_len, device="cpu", dtype=torch.bool)

    if share_memory:
        patches.share_memory_()
        patch_coords.share_memory_()
        patch_valid.share_memory_()

    put_srgb_patch(img, patches, patch_coords, patch_valid, patch_size)
    return patches, patch_coords, patch_valid

def load_image(
    path: str,
    patch_size: int = 16,
    max_seq_len: int = 1024,
    share_memory: bool = False
) -> tuple[Tensor, Tensor, Tensor]:
    with open(path, "rb", buffering=(1024 * 1024)) as file:
        img: Image.Image = Image.open(file)

        try:
            processed = process_image(img, patch_size, max_seq_len)
        except:
            img.close()
            raise

    if img is not processed:
        img.close()

    return patchify_image(processed, patch_size, max_seq_len, share_memory)

def load_model(path: str, device: torch.device | str | None = None) -> tuple[NaFlexVit, list[str]]:
    with safe_open(path, framework="pt", device="cpu") as file:
        metadata = file.metadata()

        state_dict = {
            key: file.get_tensor(key)
            for key in file.keys()
        }

    arch = metadata["modelspec.architecture"]
    if not arch.startswith("naflexvit_so400m_patch16_siglip"):
        raise ValueError(f"Unrecognized model architecture: {arch}")

    tags = metadata["classifier.labels"].split("\n")

    model = timm.create_model(
        'naflexvit_so400m_patch16_siglip',
        pretrained=False, num_classes=0,
        pos_embed_interp_mode="bilinear",
        weight_init="skip", fix_init=False,
        device="cpu", dtype=torch.bfloat16,
    )

    match arch[31:]:
        case "": # vanilla
            model.reset_classifier(len(tags))

        case "+rr_slim":
            model.reset_classifier(len(tags))

            if "attn_pool.q.weight" not in state_dict:
                model.attn_pool.q = Identity()

            if "head.bias" not in state_dict:
                model.head.bias = None

        case "+rr_chonker":
            from chonker_pool import ChonkerPool

            model.attn_pool = ChonkerPool(
                2, 1152, 72,
                device=device, dtype=torch.bfloat16
            )
            model.head = model.attn_pool.create_head(len(tags))
            model.num_classes = len(tags)

        case "+rr_hydra":
            from hydra_pool import HydraPool

            model.attn_pool = HydraPool.for_state(
                state_dict, "attn_pool.",
                device=device, dtype=torch.bfloat16
            )
            model.head = model.attn_pool.create_head()
            model.num_classes = len(tags)

            state_dict["attn_pool._extra_state"] = { "q_normed": True }

        case _:
            raise ValueError(f"Unrecognized model architecture: {arch}")

    model.eval().to(dtype=torch.bfloat16)
    model.load_state_dict(state_dict, strict=True)
    model.to(device=device)

    return model, tags
