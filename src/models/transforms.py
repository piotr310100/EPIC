# Based on the implementation from https://github.com/pytorch/vision/blob/main/references/classification/presets.py

import torch
from torchvision.transforms.functional import InterpolationMode


def get_module(use_v2):
    # We need a protected import to avoid the V2 warning in case just V1 is used
    if use_v2:
        import torchvision.transforms.v2

        return torchvision.transforms.v2
    else:
        import torchvision.transforms

        return torchvision.transforms


class ClassificationPresetEval:
    def __init__(
        self,
        *,
        crop_size,
        resize_size=256,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        backend="pil",
        use_v2=False,
    ):
        T = get_module(use_v2)
        transforms = []
        backend = backend.lower()
        if backend == "tensor":
            transforms.append(T.PILToTensor())
        elif backend != "pil":
            raise ValueError(f"backend can be 'tensor' or 'pil', but got {backend}")

        transforms += [
            T.Resize(resize_size, interpolation=interpolation, antialias=True),
            T.CenterCrop(crop_size),
        ]

        if backend == "pil":
            transforms.append(T.PILToTensor())

        transforms += [
            (
                T.ToDtype(torch.float, scale=True)
                if use_v2
                else T.ConvertImageDtype(torch.float)
            ),
            T.Normalize(mean=mean, std=std),
        ]

        if use_v2:
            transforms.append(T.ToPureTensor())

        self.transforms = T.Compose(transforms)

    def __call__(self, img):
        return self.transforms(img)


class ClassificationCropped:
    def __init__(
        self,
        *,
        resize_size=(224, 224),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        use_v2=False,
    ):
        T = get_module(use_v2)

        if isinstance(resize_size, int):
            resize_size = (resize_size, resize_size)
        elif isinstance(resize_size, tuple) and len(resize_size) != 2:
            raise ValueError(f"{resize_size=} should be a tuple of size 2")
        else:
            raise ValueError("resize_size should be a tuple of size 2 or int")

        self.transforms = T.Compose(
            [
                T.Resize(resize_size, interpolation=interpolation, antialias=True),
                T.PILToTensor(),
                T.ToDtype(torch.float, scale=True),
                T.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        return self.transforms(img)


class ClassificationFull:
    def __init__(
        self,
        *,
        resize_size=(224, 224),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        train=True,
        use_v2=False,
    ):
        T = get_module(use_v2)

        if isinstance(resize_size, int):
            resize_size = (resize_size, resize_size)
        elif isinstance(resize_size, tuple) and len(resize_size) != 2:
            raise ValueError(f"{resize_size=} should be a tuple of size 2")
        else:
            raise ValueError("resize_size should be a tuple of size 2 or int")

        transforms = []
        if train:
            transforms.extend(
                [T.RandomAffine(degrees=(-25, 25), shear=15), T.RandomHorizontalFlip()]
            )
        transforms.extend(
            [
                T.Resize(resize_size, interpolation=interpolation, antialias=True),
                T.PILToTensor(),
                T.ToDtype(torch.float, scale=True),
                T.Normalize(mean=mean, std=std),
            ]
        )

        self.transforms = T.Compose(transforms)

    def __call__(self, img):
        return self.transforms(img)
