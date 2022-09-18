# Error threshold for the test
CLASSIFICATION_ABS_THRESHOLD = 1e-5
DETECTION_BOXES_ABS_THRESHOLD = 1e-5
DETECTION_SCORES_ABS_THRESHOLD = 1e-5

MAXIMUM_ERRORS_PER_ITERATION = 4096
MAXIMUM_INFOS_PER_ITERATION = 256

CLASSIFICATION_CRITICAL_TOP_K = 1
# FORCE the gpu to be present
DEVICE = "cuda:0"

# CIFAR 10
MobileNetV2x14Cifar10 = "cifar10_mobilenetv2_x1_4"
ResNet44Cifar10 = 'c10_res44_test_01_bn-relu_base_'
DiehardNetRe6Cifar10 = 'c10_res44_test_02_bn-relu6'
DiehardNetTrainWCifar10 = 'c10_res44_test_02_bn-relu6_base_'
DiehardNetOrderICifar10 = 'c10_res44_test_02_relu6-bn_200_epochs'
# Order inversion + nan filter + Relu6
DiehardNetOINanCifar10 = 'c10_res44_test_02_relu6-bn_200_epochs_nanfilter'
# Gelu6
DiehardNetNanFilCifar10 = 'c10_res44_test_02_gelu6_nans'
# CIFAR 100
MobileNetV2x14Cifar100 = "cifar100_mobilenetv2_x1_4"
ResNet44Cifar100 = 'c100_res44_test_01_bn-relu_base_'
DiehardNetRe6Cifar100 = 'c100_res44_test_02_bn-relu6'
DiehardNetTrainWCifar100 = 'c100_res44_test_02_bn-relu6_base_'
DiehardNetOrderICifar100 = 'c100_res44_test_02_relu6-bn_200_epochs'
# Order inversion + nan filter + Relu6
DiehardNetOINanCifar100 = 'c100_res44_test_02_relu6-bn_200_epochs_nanfilter'
# Gelu6
DiehardNetNanFilCifar100 = 'c100_res44_test_02_gelu6_nans'

ALL_DNNS = {
    # CIFAR 10
    MobileNetV2x14Cifar10, ResNet44Cifar10,
    DiehardNetRe6Cifar10, DiehardNetTrainWCifar10,
    DiehardNetOrderICifar10, DiehardNetNanFilCifar10,
    DiehardNetOINanCifar10,
    # CIFAR 100
    MobileNetV2x14Cifar100, ResNet44Cifar100,
    DiehardNetRe6Cifar100, DiehardNetTrainWCifar100,
    DiehardNetOrderICifar100, DiehardNetNanFilCifar100,
    DiehardNetOINanCifar100
}

CIFAR10 = "cifar10"
CIFAR100 = "cifar100"

CLASSES = {
    CIFAR10: 10,
    CIFAR100: 100
}
