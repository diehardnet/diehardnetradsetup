#!/usr/bin/python3
import argparse
import logging
import os

import torch
import torchvision
from typing import Tuple

import configs
import console_logger
import dnn_log_helper
from configs import Timer
from diehardnet.pytorch_scripts.utils import build_model


def load_dataset(batch_size: int, dataset: str, data_dir: str, test_sample: int) -> torch.tensor:
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(
                                                    mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])
    if dataset == "cifar10":
        test_set = torchvision.datasets.cifar.CIFAR10(root=data_dir, download=False, train=False,
                                                      transform=transform)
    elif dataset == "cifar100":
        test_set = torchvision.datasets.cifar.CIFAR100(root=data_dir, download=False, train=False,
                                                       transform=transform)
    else:
        raise ValueError(f"Incorrect dataset {dataset}")

    subset = torch.utils.data.SequentialSampler(range(test_sample))
    test_loader = torch.utils.data.DataLoader(dataset=test_set, sampler=subset, batch_size=batch_size, shuffle=False)
    input_dataset, input_labels = list(), list()
    # Fixme: correct this GPU load
    for inputs, labels in test_loader:
        input_dataset.append(torch.tensor([i.to(configs.DEVICE) for i in inputs]))
        input_labels.append(torch.tensor([i.to(configs.DEVICE) for i in labels]))
    input_dataset = torch.tensor(input_dataset).to(configs.DEVICE)
    input_labels = torch.tensor(input_labels).to(configs.DEVICE)

    return input_dataset, input_labels


def load_ptl_model(args):
    # Build model (Resnet only up to now)
    optim_params = {'optimizer': args.optimizer, 'epochs': args.epochs, 'lr': args.lr, 'wd': args.wd}
    n_classes = 10 if args.dataset == 'cifar10' else 100

    ptl_model = build_model(model=args.model, n_classes=n_classes, optim_params=optim_params,
                            loss=args.loss, inject_p=args.inject_p,
                            order=args.order, activation=args.activation, affine=args.affine)
    checkpoint = torch.load(args.ckpt)
    ptl_model.load_state_dict(checkpoint['state_dict'])
    return ptl_model.model


def parse_args() -> Tuple[argparse.Namespace, str]:
    """ Parse the args and return an args namespace and the tostring from the args    """
    parser = argparse.ArgumentParser(description='PyTorch DNN radiation setup')
    parser.add_argument('--model', default=configs.ALL_DNNS[0],
                        help=f'Network name. It can be ' + ', '.join(configs.ALL_DNNS))
    parser.add_argument('--iterations', default=int(1e12), help="Iterations to run forever", type=int)
    parser.add_argument('--batchsize', default=1, help="Batches to process in parallel", type=int)

    parser.add_argument('--generate', default=False, action="store_true", help="Set this flag to generate the gold")
    parser.add_argument('--disableconsolelog', default=False, action="store_true",
                        help="Set this flag disable console logging")

    parser.add_argument('--goldpath', help="Path to the gold file")
    parser.add_argument('--grtruthcsv', help="Path ground truth verification at generate process.", default=None,
                        type=str)
    parser.add_argument('--tflite', default=False, action="store_true", help="Is it necessary to use Tensorflow lite.")
    args = parser.parse_args()
    # Check if the model is correct
    if args.model not in configs.ALL_DNNS:
        parser.print_help()
        raise ValueError("Not the correct model")

    # Check if it is only to generate the gold values
    if args.generate:
        args.iterations = 1
    else:
        args.grtruthcsv = None
    args_text = " ".join([f"{k}={v}" for k, v in vars(args).items()])
    return args, args_text


def equal(rhs: torch.Tensor, lhs: torch.Tensor, threshold: float = None) -> bool:
    """ Compare based or not in a threshold, if threshold is none then it is equal comparison    """
    if threshold:
        return bool(torch.all(torch.le(torch.abs(torch.subtract(rhs, lhs)), threshold)))
    else:
        return bool(torch.equal(rhs, lhs))


def compare_classification(output_tensor: torch.tensor,
                           golden_tensor: torch.tensor,
                           golden_top_k_labels: torch.tensor,
                           setup_iteration: int,
                           current_image: str,
                           top_k: int,
                           output_logger: logging.Logger = None) -> int:
    # Make sure that they are on CPU
    dnn_output_tensor_cpu = output_tensor.to("cpu")
    # # Debug injection
    # if setup_iteration + batch_iteration == 20:
    #     for i in range(300, 900):
    #         dnn_output_tensor_cpu[3][i] = 34.2
    output_errors = 0
    # using the same approach as the detection, compare only the positions that differ
    if equal(rhs=golden_tensor, lhs=dnn_output_tensor_cpu, threshold=configs.CLASSIFICATION_ABS_THRESHOLD) is False:
        # ------------ Check the size of the tensors
        if golden_tensor.shape != dnn_output_tensor_cpu.shape:
            error_detail = f"DIFF_SIZE g:{golden_tensor.shape} o:{dnn_output_tensor_cpu.shape}"
            if output_logger:
                output_logger.error(error_detail)
            dnn_log_helper.log_error_detail(error_detail)
        # ------------ Critical error checking
        if output_logger:
            output_logger.error("Not equal output tensors")
        # Check if there is a Critical error
        top_k_labels = torch.topk(output_tensor, k=top_k).indices.squeeze(0)
        # top_k_probs = torch.tensor([torch.softmax(output_tensor, dim=1)[0, idx].item() for idx in top_k_labels])
        if torch.any(torch.not_equal(golden_top_k_labels, top_k_labels)):
            if output_logger:
                output_logger.error("Critical error identified")
            for i, tpk_found, tpk_gold in enumerate(zip(golden_top_k_labels, top_k_labels)):
                if tpk_found != tpk_gold:  # Both are integers
                    error_detail = f"critical--img:{current_image} "
                    error_detail += f"setupit:{setup_iteration} i:{i} g:{tpk_found:.6e} o:{tpk_found}"
                    if output_logger:
                        output_logger.error(error_detail)
                    dnn_log_helper.log_error_detail(error_detail)

        # ------------ Check error on the whole output
        for i, (gold, found) in enumerate(zip(golden_tensor, dnn_output_tensor_cpu)):
            if abs(gold - found) > configs.CLASSIFICATION_ABS_THRESHOLD:
                output_errors += 1
                error_detail = f"img:{current_image} setupit:{setup_iteration} i:{i} g:{gold:.6e} o:{found:.6e}"
                if output_logger:
                    output_logger.error(error_detail)
                dnn_log_helper.log_error_detail(error_detail)

    return output_errors


def main():
    # Defining a timer
    timer = Timer()
    timer.tic()
    # Disable all torch grad
    torch.set_grad_enabled(mode=False)
    args, args_text = parse_args()
    # Starting the setup
    dnn_log_helper.set_iter_interval_print(30)
    # Load the model
    model = load_ptl_model(args=args)
    model.eval()
    model = model.to(configs.DEVICE)
    main_logger_name = str(os.path.basename(__file__)).replace(".py", "")
    terminal_logger = console_logger.ColoredLogger(main_logger_name)
    args, args_conf = parse_args()
    generate = args.generate
    iterations = args.iterations
    gold_path = args.goldpath
    model_name = args.model
    disable_console_logger = args.disableconsolelog
    batch_size = args.batchsize
    data_dir = args.datadir
    dataset = args.dataset

    if disable_console_logger is False:
        for k, v in vars(args).items():
            terminal_logger.debug(f"{k}:{v}")

    # First step is to load the inputs in the memory
    timer.tic()
    input_list = load_dataset(batch_size=batch_size, dataset=dataset, data_dir=data_dir)
    timer.toc()
    if disable_console_logger is False:
        terminal_logger.debug(f"Time necessary to load the inputs: {timer}")

    # Load if it is not a gold generating op
    gold_probabilities_list = list()
    golden_top_k_labels_list = list()
    if generate is False:
        timer.tic()
        gold_probabilities_list = torch.load(gold_path)
        golden_top_k_labels_list = [
            torch.topk(output_tensor, k=configs.CLASSIFICATION_CRITICAL_TOP_K).indices.squeeze(0)
            for output_tensor in gold_probabilities_list
        ]
        timer.toc()
        if disable_console_logger is False:
            terminal_logger.debug(f"Time necessary to load the golden outputs: {timer}")

    # Start the setup
    dnn_log_helper.start_setup_log_file(framework_name="PyTorch", args_conf=args_conf, model_name=model_name,
                                        max_errors_per_iteration=configs.MAXIMUM_ERRORS_PER_ITERATION,
                                        generate=generate)

    # Main setup loop
    setup_iteration = 0
    while setup_iteration < iterations:
        total_errors = 0
        # Loop over the input list
        for img_i, batched_input in enumerate(input_list):
            timer.tic()
            dnn_log_helper.start_iteration()
            dnn_output = model(batched_input, inject=False)
            dnn_log_helper.end_iteration()
            timer.toc()
            kernel_time = timer.diff_time

            probabilities = dnn_output.to("cpu")
            # Then compare the golden with the output
            timer.tic()
            errors = 0
            if generate is False:
                errors = compare_classification(output_tensor=dnn_output,
                                                golden_tensor=gold_probabilities_list[img_i],
                                                golden_top_k_labels=golden_top_k_labels_list[img_i],
                                                setup_iteration=setup_iteration, current_image=img_i,
                                                top_k=configs.CLASSIFICATION_CRITICAL_TOP_K,
                                                output_logger=terminal_logger)
            else:
                gold_probabilities_list.append(probabilities)
            total_errors += errors
            timer.toc()

            # Printing timing information
            if disable_console_logger is False:
                comparison_time = timer.diff_time
                time_pct = (comparison_time / (comparison_time + kernel_time)) * 100.0
                iteration_out = f"It:{setup_iteration:<3} imgit:{img_i:<3} inference time:{kernel_time:.5f}, "
                iteration_out += f"gold compare time:{comparison_time:.5f} ({time_pct:.1f}%) errors:{errors}"
                terminal_logger.debug(iteration_out)

            # Reload all the memories after error
            if total_errors != 0:
                del input_list
                del model
                model = load_ptl_model(args=args)
                model.eval()
                model = model.to(configs.DEVICE)
                input_list = load_dataset(batch_size=batch_size, dataset=dataset, data_dir=data_dir)

        setup_iteration += 1

    if generate is True:
        torch.save(gold_probabilities_list, gold_path)
    print("Finish computation.")


if __name__ == '__main__':
    main()
