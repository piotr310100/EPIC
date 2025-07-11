import json
import os
import random

import numpy as np
import torch
from omegaconf import DictConfig

from data import create_indexed_dataloader, create_prototype_dataloader
from matrix import create_matrix
from models import create_backbone_model
from prototypes import (
    generate_prototypes,
    get_prototypes_purity,
    get_purity_fn,
    get_visualized_prototypes,
    visualize_combined_prototypes,
    visualize_prototypes,
)
from schedulers import create_scheduler
from trainers import Trainer


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_purity(config: DictConfig):
    set_seeds(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(config.output_path, exist_ok=True)

    _, feature_model, transform, num_channels = create_backbone_model(
        config.model.name,
        device,
        config.model.custom_weights_path,
        config.model.num_classes,
    )
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

    positive_prototypes = generate_prototypes(
        feature_model,
        dataloader_train,
        num_channels,
        config.visualization.num_prototypes,
        device,
    )
    print("Prototypes created.")

    selected_channels = torch.randint(low=0, high=num_channels, size=(10,)).tolist()
    prototypes_before_train = get_visualized_prototypes(
        feature_model,
        positive_prototypes,
        dataloader_train,
        channels=selected_channels,
        device=device,
        img_fn=config.visualization.img_fn,
    )
    visualize_prototypes(
        prototypes_before_train,
        config.visualization.num_prototypes,
        config.output_path,
        "Base_prototypes",
    )

    prototype_dataloader = create_prototype_dataloader(
        positive_prototypes,
        dataloader_train,
        config.dataloader.batch_size,
        config.dataloader.num_workers,
        config.dataloader.pin_memory,
        config.dataloader.shuffle,
    )

    purity_fn = get_purity_fn(config.prototypes.purity_fn)

    purity_before_train = get_prototypes_purity(
        feature_model, prototype_dataloader, purity_fn=purity_fn, device=device
    )

    num_prototypes_scheduler = create_scheduler(**config.prototypes.scheduler)

    disentanglement_matrix = create_matrix(config.matrix.type, num_channels, device)

    trainer = Trainer(
        feature_model,
        dataloader_train,
        disentanglement_matrix,
        device=device,
        num_channels=num_channels,
        num_epochs_per_prototypes=config.training.num_epochs_per_prototypes,
        batch_size=config.dataloader.batch_size,
        lr=config.training.lr,
        purity_fn=purity_fn,
        num_prototypes_scheduler=num_prototypes_scheduler,
    )
    trainer.train()
    disentanglement_matrix.save_state(config.output_path, f"{config.matrix.type}.pt")

    positive_prototypes = generate_prototypes(
        feature_model,
        dataloader_train,
        num_channels,
        config.visualization.num_prototypes,
        device,
        disentanglement_matrix(),
    )
    plot_path = os.path.join(config.output_path, f"{config.matrix.type}_train")
    trainer.plot_purity_over_epochs(plot_path)
    prototypes_after_train = get_visualized_prototypes(
        feature_model,
        positive_prototypes,
        dataloader_train,
        selected_channels,
        U=disentanglement_matrix(),
        device=device,
        img_fn=config.visualization.img_fn,
    )
    visualize_combined_prototypes(
        prototypes_before_train,
        prototypes_after_train,
        config.visualization.num_prototypes,
        config.output_path,
        "Trained_prototypes",
    )
    prototype_dataloader = create_prototype_dataloader(
        positive_prototypes,
        dataloader_train,
        config.dataloader.batch_size,
        config.dataloader.num_workers,
        config.dataloader.pin_memory,
        config.dataloader.shuffle,
    )
    purity_after_train = get_prototypes_purity(
        feature_model,
        prototype_dataloader,
        device=device,
        U=disentanglement_matrix(),
        purity_fn=purity_fn,
    )
    with open(
        os.path.join(
            config.output_path, f"purity_after_{config.matrix.type}_train.json"
        ),
        "w",
    ) as outfile:
        json.dump(
            {
                "purity_before_train": purity_before_train,
                "purity_after_train": purity_after_train,
            },
            outfile,
            indent=4,
        )
    print(f"Purity before training: {purity_before_train}")
    print(f"Purity after training: {purity_after_train}")