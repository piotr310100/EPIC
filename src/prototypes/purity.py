import torch
from tqdm import tqdm

from .utils import pixelwise_multiply


def purity_argmax(feature_map, c):
    if feature_map.ndim != 4:
        raise ValueError(
            f"Input tensor must have 4 dimensions (B, C, H, W), but got {feature_map.ndim} dimensions."
        )

    batch_indices = torch.arange(feature_map.size(0), device=feature_map.device)

    channel_activation = feature_map[batch_indices, c]

    max_idx = torch.argmax(
        channel_activation.view(channel_activation.size(0), -1), dim=-1
    )

    w = feature_map.size(-1)
    max_y, max_x = max_idx // w, max_idx % w

    all_channel_values = feature_map[batch_indices, :, max_y, max_x]

    l2_norm = torch.linalg.norm(all_channel_values, dim=-1).clamp_min(1e-8)

    if torch.is_tensor(c):
        purity = all_channel_values.gather(1, c.unsqueeze(1)).squeeze(1)
    else:
        purity = all_channel_values[:, c]

    normalized_purity = purity / l2_norm

    return normalized_purity


@torch.no_grad()
def get_prototypes_purity(
    feature_model,
    prototypes_dataloader,
    device="cpu",
    U=None,
    purity_fn=purity_argmax,
):
    total_purity_normed = 0.0
    total_count = 0

    for batch, channels in tqdm(prototypes_dataloader, desc="Average Purity Calculation"):
        batch = batch.to(device, non_blocking=True)
        channels = channels.to(device, non_blocking=True)

        output = feature_model(batch)

        if U is not None:
            output = pixelwise_multiply(output, U)

        purity = purity_fn(output, channels)

        total_purity_normed += purity.sum().item()
        total_count += purity.numel()

    return total_purity_normed / total_count


PURITY_FUNCTIONS = {"argmax": purity_argmax}


def get_purity_fn(type):
    if type in PURITY_FUNCTIONS:
        return PURITY_FUNCTIONS[type]
    else:
        raise ValueError(f"Unknown purity function type: {type}")
