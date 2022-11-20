#!/usr/bin/python3

import os

import torch

print(torch.__version__)
links = [
    # Mobile net
    # CIFAR 10
    # "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/mobilenetv2/cifar10_mobilenetv2_x1_4-3bbbd6e2.pt",
    # # CIFAR 100
    # "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/mobilenetv2/cifar100_mobilenetv2_x1_4-8a269f5e.pt",

    # Diehardnet all
    "https://www.dropbox.com/s/4497lt4a72l9yw3/chipir_2022.tar.gz",

    # Diehardnet transfer learning
    "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",

]
check_points = "data/checkpoints"
if os.path.isdir(check_points) is False:
    os.mkdir(check_points)

for link in links:
    file_name = os.path.basename(link)
    print(f"Downloading {file_name}")
    final_path = f"{check_points}/{file_name}"

    if os.path.isfile(final_path) is False:
        assert os.system(f"wget {link} -P {check_points}") == 0, "Download mobile net weights not successful"

    if ".tar.gz" in file_name:
        assert os.system(f"tar xzf {final_path} -C {check_points}") == 0, "Extracting the checkpoints not successful"
