import torch
from tqdm import tqdm

from .utils import compute_channel_activation, pixelwise_multiply


@torch.no_grad
def generate_prototypes(
    feature_model, dataloader, num_channels=512, N=5, device="cpu", U=None
):
    feature_model.eval()
    activations_pos = torch.full((N, num_channels), float("-inf")).to(
        device, non_blocking=True
    )
    images_list_pos = {k: [] for k in range(num_channels)}

    for batch_idx, (images, i) in tqdm(enumerate(dataloader), total=len(dataloader)):
        images = images.to(device, non_blocking=True)
        feature_map = feature_model(images)
        if U is not None:
            feature_map = pixelwise_multiply(feature_map, U)

        batch_activations = compute_channel_activation(feature_map)

        if batch_idx == 0:
            combined_activations_pos = batch_activations
        else:
            combined_activations_pos = torch.cat((batch_activations, activations_pos))

        activations_pos, top_pos_indices = torch.topk(
            combined_activations_pos,
            min(N, len(combined_activations_pos)),
            largest=True,
            dim=0,
            sorted=True,
        )

        for k in range(num_channels):
            indices = torch.tensor([k], device=images.device)
            top_pos_images = [
                i[idx].item()
                if idx < feature_map.size(0)
                else images_list_pos[k][idx % feature_map.size(0)]
                for idx in torch.index_select(top_pos_indices, 1, indices).cpu()
            ]
            images_list_pos[k] = top_pos_images

    return images_list_pos


@torch.no_grad
def topk_active_channels(model, classification_head, image, k=4, device="cpu"):
    if image.ndim == 3:
        image = image.unsqueeze(0)
    elif image.ndim != 4:
        raise ValueError(
            f"Input tensor must have 3 or 4 dimensions (C, H, W) or (B, C, H, W), but got {image.ndim} dimensions."
        )
    model.eval()
    linear_head = classification_head.A
    image = image.cuda()
    feature_map = model(image)
    logits = classification_head(feature_map)
    predicted_class = logits.argmax(dim=-1).item()
    class_weights = linear_head[predicted_class]
    feature_map = classification_head.before_linear(feature_map)
    _, channels = torch.topk(
        class_weights * torch.nn.functional.relu(feature_map).squeeze(0),
        k,
        dim=0,
        largest=True,
        sorted=True,
    )
    return channels.tolist()
