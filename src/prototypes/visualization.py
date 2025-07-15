import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
from PIL import ImageDraw
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import resize

from .utils import pixelwise_multiply, unnormalize


def compute_activation_bbox(channel_activation, orig_size):
    max_idx = torch.argmax(
        channel_activation.view(channel_activation.size(0), -1), dim=-1
    )
    h, w = channel_activation.size(-2), channel_activation.size(-1)
    max_y, max_x = max_idx // w, max_idx % w
    scale_h, scale_w = orig_size[0] / h, orig_size[1] / w
    y_start = (max_y.float() * scale_h).long()
    y_end = ((max_y + 1).float() * scale_h).long()
    x_start = (max_x.float() * scale_w).long()
    x_end = ((max_x + 1).float() * scale_w).long()
    return x_start, y_start, x_end, y_end


@torch.no_grad
def get_visualized_prototypes(
    feature_model,
    positive_prototypes,
    orig_dataloader,
    channels,
    U=None,
    device="cpu",
    transform=unnormalize,
    img_fn="none",
):
    img_transform = img_transforms[img_fn]
    result = {}

    for c in channels:
        result[c] = []
        prototypes_idx = positive_prototypes[c]
        batch = torch.stack(
            [orig_dataloader.dataset[idx][0] for idx in prototypes_idx]
        ).to(device=device, non_blocking=True)
        feature_map = feature_model(batch)
        if U is not None:
            feature_map = pixelwise_multiply(feature_map, U)
        channel_activation = feature_map[:, c]
        x_start, y_start, x_end, y_end = compute_activation_bbox(
            channel_activation, batch.size()[2:]
        )
        R_k = resize(channel_activation.unsqueeze(1), (batch.size(-2), batch.size(-1)))
        S_k = torch.nn.functional.relu(R_k)
        P_k = S_k / torch.max(S_k).clamp_min(1e-8)
        Z = transform(batch)
        for i in range(batch.size(0)):
            prototype_image = img_transform(P_k[i], Z[i])
            prototype_image = ToPILImage()(prototype_image.cpu())
            draw = ImageDraw.Draw(prototype_image)
            draw.rectangle(
                [
                    x_start[i].item(),
                    y_start[i].item(),
                    x_end[i].item() - 1,
                    y_end[i].item() - 1,
                ],
                outline=(255, 255, 0),
                width=2,
            )
            result[c].append(prototype_image)

    return result


@torch.no_grad
def get_image_prototypes(
    feature_model,
    image,
    channels,
    U=None,
    device="cpu",
    transform=unnormalize,
    img_fn="none",
):
    img_transform = img_transforms[img_fn]

    if image.ndim == 3:
        image = image.unsqueeze(0)
    elif image.ndim != 4 or image.size(0) != 1:
        raise ValueError(
            f"Input tensor must have 3 or 4 dimensions (C, H, W) or (1, C, H, W), but got {image.ndim} dimensions."
        )
    image = image.to(device=device, non_blocking=True)

    result = {}
    batch = image
    feature_map = feature_model(batch)
    if U is not None:
        feature_map = pixelwise_multiply(feature_map, U)
    Z = transform(batch)
    i = 0

    for c in channels:
        channel_activation = feature_map[:, c]
        x_start, y_start, x_end, y_end = compute_activation_bbox(
            channel_activation, batch.size()[2:]
        )
        R_k = resize(channel_activation.unsqueeze(1), (batch.size(-2), batch.size(-1)))
        S_k = torch.nn.functional.relu(R_k)
        P_k = S_k / torch.max(S_k).clamp_min(1e-8)
        prototype_image = img_transform(P_k[i], Z[i])
        prototype_image = ToPILImage()(prototype_image.cpu())
        draw = ImageDraw.Draw(prototype_image)
        draw.rectangle(
            [
                x_start[i].item(),
                y_start[i].item(),
                x_end[i].item() - 1,
                y_end[i].item() - 1,
            ],
            outline=(255, 255, 0),
            width=2,
        )
        result[c] = prototype_image

    return result


def visualize_explanations(
    result,
    image_prototypes,
    num_cols=5,
    output_path="./output",
    name="selected_prototypes",
):
    os.makedirs(output_path, exist_ok=True)

    channels = list(image_prototypes.keys())
    num_rows = len(channels)

    fig = plt.figure(figsize=((num_cols + 3) * 3, num_rows * 3))

    outer_gs = gridspec.GridSpec(
        num_rows,
        4,
        width_ratios=[1, 1, num_cols, 0.3],
        height_ratios=[1] * num_rows,
        hspace=0.01,
    )

    for row_idx, c in enumerate(channels):
        if c not in result:
            continue

        ax_highlight = fig.add_subplot(outer_gs[row_idx, 1])
        ax_highlight.imshow(image_prototypes[c])
        ax_highlight.axis("off")

        inner_gs = gridspec.GridSpecFromSubplotSpec(
            1,
            num_cols,
            subplot_spec=outer_gs[row_idx, 2],
            wspace=0.01,
        )

        for col_idx, img_pil in enumerate(result[c][:num_cols]):
            ax_proto = fig.add_subplot(inner_gs[0, col_idx])
            ax_proto.imshow(img_pil)
            ax_proto.axis("off")

    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0.01, wspace=0.15)

    output_file = os.path.join(output_path, f"{name}.jpg")
    plt.savefig(output_file, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved explanations visualization to: {output_file}")


def visualize_prototypes(
    result, num_cols=5, output_path="./output", name="selected_prototypes"
):
    output_path = os.path.join(output_path, name)
    os.makedirs(output_path, exist_ok=True)

    for c, images in result.items():
        fig, axes = plt.subplots(
            1,
            num_cols + 1,
            figsize=(num_cols * 3 + 0.3, 3),
            gridspec_kw={"width_ratios": [1] * num_cols + [0.3]},
        )

        for col_idx in range(num_cols):
            ax = axes[col_idx] if num_cols > 1 else axes
            if col_idx < len(images):
                img_pil = images[col_idx]  # PIL image
                ax.imshow(img_pil)
                ax.axis("off")
            else:
                ax.axis("off")

        ax_label = axes[-1]
        ax_label.axis("off")
        ax_label.text(
            0.2, 0.5, str(c), ha="center", va="center", rotation=90, fontsize=24
        )

        plt.subplots_adjust(wspace=0.01)

        output_file = os.path.join(output_path, f"channel_{c}.jpg")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved prototypes visualization for Channel {c} to {output_file}")


def visualize_combined_prototypes(
    result_before,
    result_after,
    num_cols=5,
    output_path="./output",
    name="selected_prototypes",
):
    output_path = os.path.join(output_path, name)
    os.makedirs(output_path, exist_ok=True)

    for (c, images_before), (_, images_after) in zip(
        result_before.items(), result_after.items()
    ):
        fig, axes = plt.subplots(2, num_cols, figsize=(num_cols * 3, 6))

        for col_idx in range(num_cols):
            ax = axes[0][col_idx] if num_cols > 1 else axes[0]
            if col_idx < len(images_before):
                img_pil = images_before[col_idx]
                ax.imshow(img_pil)
                ax.axis("off")
            else:
                ax.axis("off")

        axes[0][0].annotate(
            "Prototypes before train",
            xy=(0, 1),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=12,
            color="black",
            xycoords="axes fraction",
        )

        for col_idx in range(num_cols):
            ax = axes[1][col_idx] if num_cols > 1 else axes[1]
            if col_idx < len(images_after):
                img_pil = images_after[col_idx]
                ax.imshow(img_pil)
                ax.axis("off")
            else:
                ax.axis("off")

        axes[1][0].annotate(
            "Prototypes after train",
            xy=(0, 1),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=12,
            color="black",
            xycoords="axes fraction",
        )

        plt.suptitle(f"Channel {c}", fontsize=16, y=1.02)
        plt.tight_layout()

        output_file = os.path.join(output_path, f"channel_{c}_prototypes.jpg")
        plt.savefig(output_file, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved prototypes visualization for Channel {c} to {output_file}")


def gray(P, Z):
    return 0.5 + P * (Z - 0.5)


def none(P, Z):
    return Z


img_transforms = {"gray": gray, "none": none}
