#!/usr/bin/python3

import os

from configs import *

MOBILE_NET = {
    # CIFAR 10
    MobileNetV2x14Cifar10: "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/mobilenetv2"
                           "/cifar10_mobilenetv2_x1_4-3bbbd6e2.pt",
    # CIFAR 100
    MobileNetV2x14Cifar100: "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/mobilenetv2"
                            "/cifar100_mobilenetv2_x1_4-8a269f5e.pt",
}
check_points = "data/checkpoints"
if os.path.isdir(check_points) is False:
    os.mkdir(check_points)

for link in MOBILE_NET:
    final_path = f"{check_points}/{os.path.basename(link)}"

    if os.path.isfile(final_path) is False:
        assert os.system(f"wget {link} -P {check_points}") == 0, "Command not successful"

