# Error threshold for the test
import time

CLASSIFICATION_ABS_THRESHOLD = 1e-5
DETECTION_BOXES_ABS_THRESHOLD = 1e-5
DETECTION_SCORES_ABS_THRESHOLD = 1e-5

MAXIMUM_ERRORS_PER_ITERATION = 4096

CLASSIFICATION_CRITICAL_TOP_K = 1
# FORCE the gpu to be present
DEVICE = "cuda:0"

ALL_DNNS = {'c100_res44_test_01_bn-relu_base_',
            'c100_res44_test_02_bn-relu6',
            'c100_res44_test_02_bn-relu6_base_',
            'c100_res44_test_02_relu6-bn_200_epochs',
            'c10_res44_test_01_bn-relu_base_',
            'c10_res44_test_02_bn-relu6',
            'c10_res44_test_02_bn-relu6_base_',
            'c10_res44_test_02_relu6-bn_200_epochs'}

CIFAR10 = "cifar10"
CIFAR100 = "cifar100"

CLASSES = {
    CIFAR10: 10,
    CIFAR100: 100
}


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

# Classification
# noinspection PyArgumentList
# class DNNName(enum.Enum):
#     # CIFAR 10
#     RepVGGA2Cifar10 = enum.auto()
#     ResNet44Cifar10 = enum.auto()
#     DiehardNetRe6Cifar10 = enum.auto()
#     DiehardNetTrainWCifar10 = enum.auto()
#     DiehardNetOrderICifar10 = enum.auto()
#     DiehardNetNanFilCifar10 = enum.auto()
#     # CIFAR 100
#     RepVGGA2Cifar100 = enum.auto()
#     ResNet44Cifar100 = enum.auto()
#     DiehardNetRe6Cifar100 = enum.auto()
#     DiehardNetTrainWCifar100 = enum.auto()
#     DiehardNetOrderICifar100 = enum.auto()
#     DiehardNetNanFilCifar100 = enum.auto()
#
#     def __str__(self): return str(self.name)
#
#     def __repr__(self): return str(self)
#
#     def equals(self, other: str) -> bool: return other == self.name
#
#     @classmethod
#     def is_str_valid(cls, model: str) -> bool:
#         return any([i.name == model for i in cls])
#
#
# MODEL_LINKS = {
#     # CIFAR 10
#     DNNName.RepVGGA2Cifar10: "cifar10_repvgg_a2",
#     DNNName.ResNet44Cifar10: None,
#     DNNName.DiehardNetRe6Cifar10: None,
#     DNNName.DiehardNetTrainWCifar10: None,
#     DNNName.DiehardNetOrderICifar10: None,
#     DNNName.DiehardNetNanFilCifar10: None,
#     # CIFAR 100
#     DNNName.RepVGGA2Cifar100: "cifar100_repvgg_a2",
#     DNNName.ResNet44Cifar100: None,
#     DNNName.DiehardNetRe6Cifar100: None,
#     DNNName.DiehardNetTrainWCifar100: None,
#     DNNName.DiehardNetOrderICifar100: None,
#     DNNName.DiehardNetNanFilCifar100: None,
# }
