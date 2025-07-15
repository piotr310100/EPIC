import torch


class ResNetBackbone(torch.nn.Module):
    def __init__(self, model):
        super(ResNetBackbone, self).__init__()
        self.feature = torch.nn.Sequential(*list(model.children())[:-2])

    def forward(self, x):
        x = self.feature(x)
        return x


class ConvNeXtBackbone(torch.nn.Module):
    def __init__(self, model):
        super(ConvNeXtBackbone, self).__init__()
        self.feature = torch.nn.Sequential(*list(model.children())[:-2])
        self.eps = model.classifier[0].eps
        self.gamma = model.classifier[0].weight
        self.beta = model.classifier[0].bias

    def _layernorm(self, x):
        mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)
        var = torch.var(
            x.mean(dim=(2, 3), keepdim=True), dim=(1), keepdim=True, unbiased=False
        )
        x = (
            torch.div(x - mean, torch.sqrt(var + self.eps))
            .mul(self.gamma.view(1, -1, 1, 1))
            .add(self.beta.view(1, -1, 1, 1))
        )
        return x

    def forward(self, x):
        x = self.feature(x)
        x = self._layernorm(x)
        return x


class DenseNetBackbone(torch.nn.Module):
    def __init__(self, model):
        super(DenseNetBackbone, self).__init__()
        self.feature = torch.nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        x = self.feature(x)
        x = torch.nn.functional.relu(x, inplace=True)
        return x


class SwinTransformerBackbone(torch.nn.Module):
    def __init__(self, model):
        super(SwinTransformerBackbone, self).__init__()
        self.feature = torch.nn.Sequential(*list(model.children())[:-3])

    def forward(self, x):
        x = self.feature(x)
        return x


BACKBONE_CLASSES = {
    "resnet": ResNetBackbone,
    "convnext": ConvNeXtBackbone,
    "densenet": DenseNetBackbone,
    "swin": SwinTransformerBackbone,
}


def create_backbone(model_name, base_model):
    for prefix, cls in BACKBONE_CLASSES.items():
        if prefix in model_name:
            return cls(base_model)
    raise ValueError(f"No backbone implementation for model '{model_name}'")
