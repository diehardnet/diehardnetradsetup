import argparse
import logging
from typing import List

import torch
import torchvision

import common
import configs
import dnn_log_helper


def load_model(args: argparse.Namespace) -> torch.Module:
    checkpoint_path = f"{args.checkpointdir}/{args.ckpt}"
    model = object()
    if args.name in configs.DEEPLABV3_RESNET50:
        # Better for offline approach
        model = torchvision.models.segmentation.deeplabv3_resnet50(weights=None)
    else:
        dnn_log_helper.log_and_crash(f"Incorrect DNN configuration {args.name}")

    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    model = model.to(configs.DEVICE)
    return model


def compare_segmentation(output_tensor: torch.tensor,
                         golden_tensor: torch.tensor,
                         golden_top_k_labels: torch.tensor,
                         ground_truth_labels: torch.tensor,
                         batch_id: int,
                         top_k: int,
                         output_logger: logging.Logger = None) -> int:
    # Make sure that they are on CPU
    out_is_cuda, golden_is_cuda = output_tensor.is_cuda, golden_tensor.is_cuda
    if out_is_cuda or golden_is_cuda:
        dnn_log_helper.log_and_crash(
            fatal_string=f"Tensors are not on CPU. OUT IS CUDA:{out_is_cuda} GOLDEN IS CUDA:{golden_is_cuda}")

    if ground_truth_labels.is_cuda:
        dnn_log_helper.log_and_crash(fatal_string=f"Ground truth is on cuda.")

    output_errors = 0

    # First check if the tensors are equal or not
    if common.equal(rhs=output_tensor, lhs=golden_tensor, threshold=configs.CLASSIFICATION_ABS_THRESHOLD) is True:
        return output_errors

    # ------------ Check the size of the tensors
    if output_tensor.shape != golden_tensor.shape:
        info_detail = f"shape-diff g:{golden_tensor.shape} o:{output_tensor.shape}"
        if output_logger:
            output_logger.error(info_detail)
        dnn_log_helper.log_info_detail(info_detail)

    if output_errors != 0:
        dnn_log_helper.log_error_count(error_count=output_errors)

    return output_errors


def check_dnn_accuracy(predicted: List[torch.tensor], ground_truth: List[torch.tensor],
                       output_logger: logging.Logger = None) -> None:
    raise NotImplementedError()
