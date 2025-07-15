from torchvision.models import (
    resnet50, ResNet50_Weights,
    resnet34, ResNet34_Weights,
    resnet18, ResNet18_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
    convnext_base, ConvNeXt_Base_Weights,
    convnext_small, ConvNeXt_Small_Weights,
    convnext_large, ConvNeXt_Large_Weights,
    densenet121, DenseNet121_Weights,
    swin_v2_s, Swin_V2_S_Weights,
)

model_list = {
    "resnet18": (resnet18, ResNet18_Weights.DEFAULT, 512),
    "resnet34": (resnet34, ResNet34_Weights.DEFAULT, 512),
    "resnet50": (resnet50, ResNet50_Weights.DEFAULT, 2048),
    "convnext_tiny": (convnext_tiny, ConvNeXt_Tiny_Weights.DEFAULT, 768),
    "convnext_small": (convnext_small, ConvNeXt_Small_Weights.DEFAULT, 768),
    "convnext_base": (convnext_base, ConvNeXt_Base_Weights.DEFAULT, 1024),
    "convnext_large": (convnext_large, ConvNeXt_Large_Weights.DEFAULT, 1536),
    "densenet121": (densenet121, DenseNet121_Weights.DEFAULT, 1024),
    "swin_v2_s": (swin_v2_s, Swin_V2_S_Weights.DEFAULT, 768),
}