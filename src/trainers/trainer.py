import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from data import create_prototype_dataloader
from prototypes import (
    generate_prototypes,
    get_prototypes_purity,
    pixelwise_multiply,
    purity_argmax,
)


class Trainer:
    def __init__(
        self,
        model,
        dataloader,
        disentanglement_matrix,
        num_channels=512,
        num_epochs_per_prototypes=10,
        batch_size=512,
        lr=0.001,
        device="cpu",
        purity_fn=purity_argmax,
        num_prototypes_scheduler=None,
    ):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.purity_fn = purity_fn
        self.num_epochs_per_prototypes = num_epochs_per_prototypes
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.disentanglement_matrix = disentanglement_matrix
        self.optimizer = torch.optim.Adam(
            self.disentanglement_matrix.parameters(), lr=lr, weight_decay=1e-5
        )
        self.avg_purities = []
        self.num_prototypes_scheduler = num_prototypes_scheduler

    def loss_fn(self, P, k):
        transformed_P = pixelwise_multiply(P, self.disentanglement_matrix())
        normalized_purity = self.purity_fn(transformed_P, k)
        return -normalized_purity

    def train_prototypes(self, prototype_dataloader):
        for epoch in range(self.num_epochs_per_prototypes):
            avg_purity = 0.0
            for batch_prototypes, channels in tqdm(
                prototype_dataloader,
                total=len(prototype_dataloader),
                desc=f"Epoch {epoch}",
            ):
                batch_prototypes = batch_prototypes.to(self.device)
                channels = channels.to(self.device)

                with torch.no_grad():
                    feature_map = self.model(batch_prototypes)

                loss = self.loss_fn(feature_map, channels).mean()
                avg_purity += -loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            avg_purity = avg_purity / len(prototype_dataloader)
            self.avg_purities.append(avg_purity)

            if (epoch + 1) % 10 == 0 or (epoch + 1) == self.num_epochs_per_prototypes:
                print(
                    f"Epoch [{epoch + 1}/{self.num_epochs_per_prototypes}], Average purity: {avg_purity:.4f}"
                )

    def train(self):
        for n in self.num_prototypes_scheduler:
            print(f"Starting training for n={n}")
            positive_prototypes = generate_prototypes(
                self.model,
                self.dataloader,
                self.num_channels,
                n,
                self.device,
                self.disentanglement_matrix(),
            )
            prototype_dataloader = create_prototype_dataloader(
                positive_prototypes,
                self.dataloader,
                batch_size=self.batch_size,
                shuffle=True,
            )
            prototypes_purity = get_prototypes_purity(
                self.model,
                prototype_dataloader,
                device=self.device,
                purity_fn=self.purity_fn,
                U=self.disentanglement_matrix(),
            )
            self.avg_purities.append(prototypes_purity)
            print(f"Dataloader created. Starting training for n={n} prototypes.")
            self.train_prototypes(prototype_dataloader)

    def plot_purity_over_epochs(self, save_file):
        epochs = np.arange(0, len(self.avg_purities))

        plt.figure(figsize=(10, 6))
        plt.plot(
            epochs, self.avg_purities, label="Purity during training", color="blue"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Average Purity")
        plt.title("Change in Average Purity Over Training")
        plt.legend()
        plt.grid(True)

        plt.savefig(save_file, bbox_inches="tight")
        plt.close()
        print(f"Purity plot saved to {save_file}")
