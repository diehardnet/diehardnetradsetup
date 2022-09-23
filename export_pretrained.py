#!/usr/bin/python3

import torch

# To export
data_dir = "data/checkpoints"
models = {
    "cifar10_mobilenetv2_x1_4": "cifar10_mobilenetv2_x1_4-3bbbd6e2.pt",
    "cifar100_mobilenetv2_x1_4": "cifar100_mobilenetv2_x1_4-8a269f5e.pt"
}

for name, weights in models.items():
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", name, pretrained=False).eval()
    traced_graph = torch.jit.trace(model, torch.randn(128, 3, 32, 32))
    ckpt = f"{data_dir}/{weights}"
    traced_graph.save(ckpt)
