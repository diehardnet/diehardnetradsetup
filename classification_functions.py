import logging
from typing import List

import torch

import common
import configs
import dnn_log_helper


def compare_classification(output_tensor: torch.tensor,
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
    # global TEST
    # TEST += 1
    # if TEST == 100:
    #     # # FIXME: FI debug
    #     # # Simulate a non-critical error
    #     output_tensor[34, 0] *= 0.9
    #     # Simulate a critical error
    #     # output_tensor[55, 0] = 39304
    #     # # Shape SDC
    #     # output_tensor = torch.reshape(output_tensor, (4, 3200))

    # First check if the tensors are equal or not
    if common.equal(rhs=output_tensor, lhs=golden_tensor, threshold=configs.CLASSIFICATION_ABS_THRESHOLD) is True:
        return output_errors

    # ------------ Check the size of the tensors
    if output_tensor.shape != golden_tensor.shape:
        info_detail = f"shape-diff g:{golden_tensor.shape} o:{output_tensor.shape}"
        if output_logger:
            output_logger.error(info_detail)
        dnn_log_helper.log_info_detail(info_detail)

    # Iterate over the batches
    for img_id, (output_batch, golden_batch, golden_batch_label, ground_truth_label) in enumerate(
            zip(output_tensor, golden_tensor, golden_top_k_labels, ground_truth_labels)):
        # using the same approach as the detection, compare only the positions that differ
        if common.equal(rhs=golden_batch, lhs=output_batch, threshold=configs.CLASSIFICATION_ABS_THRESHOLD) is False:
            # ------------ Critical error checking ---------------------------------------------------------------------
            if output_logger:
                output_logger.error(f"batch:{batch_id} Not equal output tensors")

            # Check if there is a Critical error
            top_k_batch_label_flatten = torch.topk(output_batch, k=top_k).indices.squeeze(0).flatten()
            golden_batch_label_flatten = golden_batch_label.flatten()
            for i, (tpk_found, tpk_gold) in enumerate(zip(golden_batch_label_flatten, top_k_batch_label_flatten)):
                # Both are integers, and log only if it is feasible
                if tpk_found != tpk_gold and output_errors < configs.MAXIMUM_ERRORS_PER_ITERATION:
                    output_errors += 1
                    error_detail_ctr = f"batch:{batch_id} critical-img:{img_id} i:{i} g:{tpk_gold} o:{tpk_found}"
                    error_detail_ctr += f" gt:{ground_truth_label}"
                    if output_logger:
                        output_logger.error(error_detail_ctr)
                    dnn_log_helper.log_error_detail(error_detail_ctr)

            # ------------ Check error on the whole output -------------------------------------------------------------
            for i, (gold, found) in enumerate(zip(golden_batch, output_batch)):
                diff = abs(gold - found)
                if diff > configs.CLASSIFICATION_ABS_THRESHOLD and output_errors < configs.MAXIMUM_ERRORS_PER_ITERATION:
                    output_errors += 1
                    error_detail_out = f"batch:{batch_id} img:{img_id} i:{i} g:{gold:.6e} o:{found:.6e}"
                    if output_logger:
                        output_logger.error(error_detail_out)
                    dnn_log_helper.log_error_detail(error_detail_out)

    if output_errors != 0:
        dnn_log_helper.log_error_count(error_count=output_errors)

    return output_errors


def check_dnn_accuracy(predicted: List[torch.tensor], ground_truth: List[torch.tensor],
                       output_logger: logging.Logger = None) -> None:
    correct, gt_count = 0, 0
    for pred, gt in zip(predicted, ground_truth):
        gt_count += gt.shape[0]
        correct += torch.sum(torch.eq(pred, gt))
    if output_logger:
        output_logger.debug(f"Correct predicted samples:{correct} - ({(correct / gt_count) * 100:.2f}%)")


def copy_output_to_cpu(output: torch.tensor, weights) -> torch.tensor:
    return output.to('cpu')
