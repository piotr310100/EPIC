import torch

from prototypes import pixelwise_multiply


class ModifiedHeadResnet(torch.nn.Module):
    def __init__(self, model, U=None, device="cuda"):
        super(ModifiedHeadResnet, self).__init__()
        self.avgpool = list(model.children())[-2].to(device)
        fc = list(model.children())[-1].to(device)
        self.U = U
        if U is None:
            self.A = fc.weight
        else:
            self.A = fc.weight @ self.U.inverse()
        self.b = fc.bias

    def _preprocess_input(self, x):
        return pixelwise_multiply(x, self.U())

    def forward(self, x):
        x = self._preprocess_input(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return torch.nn.functional.linear(x, self.A, self.b)

    def before_linear(self, x):
        x = self._preprocess_input(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class ModifiedHeadConvNeXt(torch.nn.Module):
    def __init__(self, model, U=None, device="cuda"):
        super(ModifiedHeadConvNeXt, self).__init__()
        self.avgpool = list(model.children())[-2].to(device)
        classifier = list(model.children())[-1].to(device)
        self.eps = model.classifier[0].eps
        self.gamma = model.classifier[0].weight
        self.beta = model.classifier[0].bias
        self.flatten = classifier[1]
        fc = classifier[2]
        self.U = U
        if U is None:
            self.A = fc.weight
        else:
            self.A = fc.weight @ self.U.inverse()
        self.b = fc.bias

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

    def _preprocess_input(self, x):
        return pixelwise_multiply(x, self.U())

    def forward(self, x):
        x = self._preprocess_input(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return torch.nn.functional.linear(x, self.A, self.b)

    def before_linear(self, x):
        x = self._preprocess_input(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return x


class ModifiedHeadDenseNet(torch.nn.Module):
    def __init__(self, model, U=None, device="cuda"):
        super(ModifiedHeadDenseNet, self).__init__()
        fc = model.classifier.to(device)
        self.U = U
        if U is None:
            self.A = fc.weight
        else:
            self.A = fc.weight @ self.U.inverse()
        self.b = fc.bias

    def _preprocess_input(self, x):
        return pixelwise_multiply(x, self.U())

    def forward(self, x):
        x = self._preprocess_input(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return torch.nn.functional.linear(x, self.A, self.b)

    def before_linear(self, x):
        x = self._preprocess_input(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x


class ModifiedHeadSwinTransformer(torch.nn.Module):
    def __init__(self, model, U=None, device="cuda"):
        super(ModifiedHeadSwinTransformer, self).__init__()
        self.avgpool = list(model.children())[-3].to(device)
        self.flatten = list(model.children())[-2].to(device)
        fc = list(model.children())[-1].to(device)
        self.U = U
        if U is None:
            self.A = fc.weight
        else:
            self.A = fc.weight @ self.U.inverse()
        self.b = fc.bias

    def _preprocess_input(self, x):
        return pixelwise_multiply(x, self.U())

    def forward(self, x):
        x = self._preprocess_input(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return torch.nn.functional.linear(x, self.A, self.b)

    def before_linear(self, x):
        x = self._preprocess_input(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return x


HEAD_CLASSES = {
    "resnet": ModifiedHeadResnet,
    "convnext": ModifiedHeadConvNeXt,
    "densenet": ModifiedHeadDenseNet,
    "swin": ModifiedHeadSwinTransformer,
}


def create_modified_head(model, model_name, U=None):
    for prefix, cls in HEAD_CLASSES.items():
        if prefix in model_name:
            return cls(model, U)
    raise ValueError(f"No modified head implementation for '{model_name}'")
