#!/usr/bin/python3

import os

from configs import *

mobilenetv2_repo = "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/mobilenetv2/"

MODEL_LINKS = {
    # CIFAR 10
    MobileNetV2x14Cifar10: f"{mobilenetv2_repo}/cifar10_mobilenetv2_x1_4-3bbbd6e2.pt",
    # ResNet44Cifar10: "c10_res44_test_01_bn-relu_base_sgd-epoch=99-val_acc=0.92.ckpt",
    # DiehardNetRe6Cifar10: "'c10_res44_test_02_bn-relu6_base_sgd-epoch=99-val_acc=0.93.ckpt",
    # DiehardNetTrainWCifar10: "c10_res44_test_02_bn-relu6_sgd-epoch=99-val_acc=0.93.ckpt",
    # DiehardNetOrderICifar10: "c10_res44_test_02_relu6-bn_sgd-epoch=99-val_acc=0.91.ckpt",
    # DiehardNetNanFilCifar10: None,
    # CIFAR 100
    MobileNetV2x14Cifar100: f"{mobilenetv2_repo}/cifar100_mobilenetv2_x1_4-8a269f5e.pt",
    # ResNet44Cifar100: "c100_res44_test_01_bn-relu_base_sgd_9-epoch=99-val_acc=0.70.ckpt",
    # DiehardNetRe6Cifar100: "c100_res44_test_02_bn-relu6_base_sgd-epoch=99-val_acc=0.70.ckpt",
    # DiehardNetTrainWCifar100: "c100_res44_test_02_bn-relu6_sgd-epoch=99-val_acc=0.70.ckpt",
    # DiehardNetOrderICifar100: "c100_res44_test_02_relu6-bn_sgd-epoch=99-val_acc=0.69.ckpt",
    # DiehardNetNanFilCifar100: None,
}
check_points = "data/checkpoints"
if os.path.isdir(check_points) is False:
    os.mkdir(check_points)

for m, link in MODEL_LINKS.items():
    final_path = f"{check_points}/{m}"
    if os.path.isfile(final_path) is False:
        assert os.system(f"wget {link} -O {final_path}"), "Command not successful"
