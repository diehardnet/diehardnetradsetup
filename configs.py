# Error threshold for the test
import enum
import time

import torch

CLASSIFICATION_ABS_THRESHOLD = 1e-5
DETECTION_BOXES_ABS_THRESHOLD = 1e-5
DETECTION_SCORES_ABS_THRESHOLD = 1e-5

MAXIMUM_ERRORS_PER_ITERATION = 4096

CLASSIFICATION_CRITICAL_TOP_K = 1

# FORCE the gpu to be present
DEVICE = "cuda:0"
assert torch.cuda.is_available() is True, RuntimeError("GPU is not available")


# Classification
# noinspection PyArgumentList
class DNNName(enum.Enum):
    # CIFAR 10
    MobileNetV2Cifar10 = enum.auto()
    ResNet44Cifar10 = enum.auto()
    DiehardNetRe6Cifar10 = enum.auto()
    DiehardNetTrainWCifar10 = enum.auto()
    DiehardNetOrderICifar10 = enum.auto()
    DiehardNetNanFilCifar10 = enum.auto()
    # CIFAR 100
    MobileNetV2Cifar100 = enum.auto()
    ResNet44Cifar100 = enum.auto()
    DiehardNetRe6Cifar100 = enum.auto()
    DiehardNetTrainWCifar100 = enum.auto()
    DiehardNetOrderICifar100 = enum.auto()
    DiehardNetNanFilCifar100 = enum.auto()

    def __str__(self): return str(self.name)

    def __repr__(self): return str(self)


MODEL_LINKS = {
    # CIFAR 10
    DNNName.MobileNetV2Cifar10: "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/mobilenetv2/cifar10_mobilenetv2_x1_4-3bbbd6e2.pt",
    DNNName.ResNet44Cifar10: None,
    DNNName.DiehardNetRe6Cifar10: None,
    DNNName.DiehardNetTrainWCifar10: None,
    DNNName.DiehardNetOrderICifar10: None,
    DNNName.DiehardNetNanFilCifar10: None,
    # CIFAR 100
    DNNName.MobileNetV2Cifar100: "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/mobilenetv2/cifar100_mobilenetv2_x1_4-8a269f5e.pt",
    DNNName.ResNet44Cifar100: None,
    DNNName.DiehardNetRe6Cifar100: None,
    DNNName.DiehardNetTrainWCifar100: None,
    DNNName.DiehardNetOrderICifar100: None,
    DNNName.DiehardNetNanFilCifar100: None,
}

ALL_DNNS = list(MODEL_LINKS.keys())


class Timer:
    time_measure = 0

    def tic(self):
        self.time_measure = time.time()

    def toc(self):
        self.time_measure = time.time() - self.time_measure

    @property
    def diff_time(self):
        return self.time_measure

    def __str__(self):
        return f"{self.time_measure:.4f}s"

    def __repr__(self):
        return str(self)
