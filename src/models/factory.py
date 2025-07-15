from dataclasses import astuple, dataclass

import torch
from torchvision.transforms import InterpolationMode

from models.registry import model_list

from .backbones import create_backbone
from .transforms import ClassificationPresetEval


@dataclass
class ModelBundle:
    base_model: torch.nn.Module
    backbone: torch.nn.Module
    transform: callable
    num_channels: int

    def __iter__(self):
        return iter(astuple(self))


def replace_classifier(base_model, model_name, num_classes):
    if "resnet" in model_name:
        base_model.fc = torch.nn.Linear(base_model.fc.in_features, num_classes)
    elif "convnext" in model_name:
        base_model.classifier[2] = torch.nn.Linear(
            base_model.classifier[2].in_features, num_classes
        )
    elif "densenet" in model_name:
        base_model.classifier = torch.nn.Linear(
            base_model.classifier.in_features, num_classes
        )
    elif "swin" in model_name:
        base_model.head = torch.nn.Linear(base_model.head.in_features, num_classes)
    else:
        raise ValueError(
            f"Custom classifier replacement not implemented for model '{model_name}'"
        )
    return base_model


def create_backbone_model(
    model_name, device="cuda", custom_weights_path=None, num_classes=1000
):
    if model_name not in model_list:
        raise ValueError(
            f"Invalid model name '{model_name}'. Supported models: {', '.join(model_list.keys())}"
        )

    model_fn, weights, num_channels = model_list[model_name]

    if custom_weights_path:
        print(f"Loading custom weights from {custom_weights_path}")
        base_model = model_fn(weights=None).to(device)
        base_model = replace_classifier(base_model, model_name, num_classes)
        state_dict = torch.load(custom_weights_path, map_location=device)
        base_model.load_state_dict(state_dict["model"])
        transform = ClassificationPresetEval(
            crop_size=224,
            resize_size=256,
            interpolation=InterpolationMode.BILINEAR,
        )
    else:
        base_model = model_fn(weights=weights).to(device)
        transform = weights.transforms()

    backbone = create_backbone(model_name, base_model)
    return ModelBundle(base_model, backbone, transform, num_channels)
