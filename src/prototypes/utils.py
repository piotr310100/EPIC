import torch


def compute_channel_activation(feature_map):
    channel_activation = feature_map.view(feature_map.size(0), feature_map.size(1), -1)
    return channel_activation.sum(dim=2)


def pixelwise_multiply(P, U):
    return torch.einsum("ij,bjhw->bihw", U, P)


def unnormalize(batch):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).to(batch.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).to(batch.device)
    return batch * std + mean