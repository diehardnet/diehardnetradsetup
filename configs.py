# Error threshold for the test
CLASSIFICATION_ABS_THRESHOLD = 1e-5
DETECTION_BOXES_ABS_THRESHOLD = 1e-5
DETECTION_SCORES_ABS_THRESHOLD = 1e-5

MAXIMUM_ERRORS_PER_ITERATION = 4096

CLASSIFICATION_CRITICAL_TOP_K = 1
# FORCE the gpu to be present
DEVICE = "cuda:0"

# CIFAR 10
MobileNetV2x14Cifar10 = "c10_mobilenetv2_x1_4"
ResNet44Cifar10 = 'c10_res44_test_01_bn-relu_base_'
DiehardNetRe6Cifar10 = 'c10_res44_test_02_bn-relu6'
DiehardNetTrainWCifar10 = 'c10_res44_test_02_bn-relu6_base_'
DiehardNetOrderICifar10 = 'c10_res44_test_02_relu6-bn_200_epochs'
DiehardNetNanFilCifar10 = ""
# CIFAR 100
MobileNetV2x14Cifar100 = "c10_mobilenetv2_x1_4"
ResNet44Cifar100 = 'c100_res44_test_01_bn-relu_base_'
DiehardNetRe6Cifar100 = 'c100_res44_test_02_bn-relu6'
DiehardNetTrainWCifar100 = 'c100_res44_test_02_bn-relu6_base_'
DiehardNetOrderICifar100 = 'c100_res44_test_02_relu6-bn_200_epochs'
DiehardNetNanFilCifar100 = ""

MODEL_LINKS = {
    # CIFAR 10
    MobileNetV2x14Cifar10: "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/mobilenetv2/"
                           "cifar10_mobilenetv2_x1_4-3bbbd6e2.pt",
    ResNet44Cifar10: "c10_res44_test_01_bn-relu_base_sgd-epoch=99-val_acc=0.92.ckpt",
    DiehardNetRe6Cifar10: "'c10_res44_test_02_bn-relu6_base_sgd-epoch=99-val_acc=0.93.ckpt",
    DiehardNetTrainWCifar10: "c10_res44_test_02_bn-relu6_sgd-epoch=99-val_acc=0.93.ckpt",
    DiehardNetOrderICifar10: "c10_res44_test_02_relu6-bn_sgd-epoch=99-val_acc=0.91.ckpt",
    DiehardNetNanFilCifar10: None,
    # CIFAR 100
    MobileNetV2x14Cifar100: "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/mobilenetv2/"
                            "cifar100_mobilenetv2_x1_4-8a269f5e.pt",
    ResNet44Cifar100: "c100_res44_test_01_bn-relu_base_sgd_9-epoch=99-val_acc=0.70.ckpt",
    DiehardNetRe6Cifar100: "c100_res44_test_02_bn-relu6_base_sgd-epoch=99-val_acc=0.70.ckpt",
    DiehardNetTrainWCifar100: "c100_res44_test_02_bn-relu6_sgd-epoch=99-val_acc=0.70.ckpt",
    DiehardNetOrderICifar100: "c100_res44_test_02_relu6-bn_sgd-epoch=99-val_acc=0.69.ckpt",
    DiehardNetNanFilCifar100: None,
}

ALL_DNNS = list(MODEL_LINKS.keys())

CIFAR10 = "cifar10"
CIFAR100 = "cifar100"

CLASSES = {
    CIFAR10: 10,
    CIFAR100: 100
}
