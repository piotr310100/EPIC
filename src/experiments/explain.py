import json
import os
import random

import numpy as np
import torch
from omegaconf import DictConfig

from data import create_indexed_dataloader, create_prototype_dataloader
from matrix import create_matrix
from models import create_backbone_model, create_modified_head
from prototypes import (
    generate_prototypes,
    get_image_prototypes,
    get_prototypes_purity,
    get_purity_fn,
    get_visualized_prototypes,
    topk_active_channels,
    visualize_explanations,
)


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def explain_predictions(config: DictConfig):
    set_seeds(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_model, feature_model, transform, num_channels = create_backbone_model(
        config.model.name,
        device,
        config.model.custom_weights_path,
        config.model.num_classes,
    )
    base_model = base_model.to(device).eval()
    feature_model = feature_model.to(device).eval()
    print(f"Feature model {config.model.name} initialized without classification head.")

    dataloader_train = create_indexed_dataloader(
        config.dataset.datapath_train,
        transform,
        config.dataloader.batch_size,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
        shuffle=config.dataloader.shuffle,
    )

    disentanglement_matrix = create_matrix(config.matrix.type, num_channels, device)
    disentanglement_matrix.load_state(
        os.path.join(config.output_path, f"{config.matrix.type}.pt"),
        map_location=device,
    )

    positive_prototypes = generate_prototypes(
        feature_model,
        dataloader_train,
        num_channels,
        config.visualization.num_prototypes,
        device,
        disentanglement_matrix(),
    )
    print("Prototypes created.")

    prototype_dataloader = create_prototype_dataloader(
        positive_prototypes,
        dataloader_train,
        config.dataloader.batch_size,
        config.dataloader.num_workers,
        config.dataloader.pin_memory,
        config.dataloader.shuffle,
    )
    purity_fn = get_purity_fn(config.prototypes.purity_fn)
    purity = get_prototypes_purity(
        feature_model,
        prototype_dataloader,
        device=device,
        U=disentanglement_matrix(),
        purity_fn=purity_fn,
    )
    print(f"Avg purity: {purity}")

    classification_head = create_modified_head(
        base_model, config.model.name, disentanglement_matrix
    )

    dataloader_val = create_indexed_dataloader(
        config.dataset.datapath_val,
        transform,
        config.dataloader.batch_size,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
        shuffle=config.dataloader.shuffle,
    )

    for repeat in range(config.visualization.num_explanations):
        idx = np.random.randint(0, len(dataloader_val.dataset))
        image, *_ = dataloader_val.dataset[idx]
        channels = topk_active_channels(
            feature_model,
            classification_head,
            image,
            k=config.visualization.k,
            device=device,
        )
        image_prototypes = get_image_prototypes(
            feature_model,
            image,
            channels,
            U=disentanglement_matrix(),
            device=device,
            img_fn=config.visualization.img_fn,
        )
        visualized_prototypes = get_visualized_prototypes(
            feature_model,
            positive_prototypes,
            dataloader_val,
            channels,
            U=disentanglement_matrix(),
            device=device,
            img_fn=config.visualization.img_fn,
        )
        explanation_path = os.path.join(config.output_path, "Explanations")
        visualize_explanations(
            visualized_prototypes,
            image_prototypes,
            config.visualization.num_prototypes,
            explanation_path,
            str(repeat),
        )
