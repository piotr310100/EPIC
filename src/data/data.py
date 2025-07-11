import torchvision
from torch.utils.data import DataLoader, Dataset


class PrototypesDataset(Dataset):
    def __init__(
        self,
        positive_prototypes,
        orig_dataloader,
        transform=None,
    ):
        self.prototypes_all = []
        self.orig_dataloader = orig_dataloader

        for c, prototypes in positive_prototypes.items():
            for i in prototypes:
                self.prototypes_all.append((i, c))

        self.transform = transform

    def __len__(self):
        return len(self.prototypes_all)

    def __getitem__(self, idx):
        i, c = self.prototypes_all[idx]
        prototype = self.orig_dataloader.dataset[i][0]

        if self.transform:
            prototype = self.transform(prototype)

        return prototype, c


class IndexedImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)

    def __getitem__(self, idx):
        image, _ = super().__getitem__(idx)
        return image, idx


def create_prototype_dataloader(
    positive_prototypes,
    orig_dataloader,
    batch_size,
    num_workers=0,
    pin_memory=True,
    shuffle=True,
):
    dataset = PrototypesDataset(
        positive_prototypes,
        orig_dataloader,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return dataloader


def create_indexed_dataloader(
    datapath,
    transform,
    batch_size,
    num_workers,
    pin_memory,
    shuffle=False,
):
    dataset = IndexedImageFolder(
        root=datapath,
        transform=transform,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return dataloader
