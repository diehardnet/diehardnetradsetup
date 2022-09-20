#!/usr/bin/python3
import argparse
import logging
import os
import sys
import time
import inspect
import traceback

import torch
import torchvision
from typing import Tuple, List
import yaml

import configs
import console_logger
import dnn_log_helper
from pytorch_scripts.utils import build_model


class Timer:
    time_measure = 0

    def tic(self): self.time_measure = time.time()

    def toc(self): self.time_measure = time.time() - self.time_measure

    @property
    def diff_time(self): return self.time_measure

    @property
    def diff_time_str(self): return str(self)

    def __str__(self): return f"{self.time_measure:.4f}s"

    def __repr__(self): return str(self)


def load_dataset(batch_size: int, dataset: str, data_dir: str, test_sample: int,
                 download_dataset: bool) -> torch.tensor:
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(
                                                    mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])
    if dataset == configs.CIFAR10:
        test_set = torchvision.datasets.cifar.CIFAR10(root=data_dir, download=download_dataset, train=False,
                                                      transform=transform)
    elif dataset == configs.CIFAR100:
        test_set = torchvision.datasets.cifar.CIFAR100(root=data_dir, download=download_dataset, train=False,
                                                       transform=transform)
    else:
        raise ValueError(f"Incorrect dataset {dataset}")

    subset = torch.utils.data.SequentialSampler(range(test_sample))
    test_loader = torch.utils.data.DataLoader(dataset=test_set, sampler=subset, batch_size=batch_size,
                                              shuffle=False, pin_memory=True)
    input_dataset, input_labels = list(), list()
    for inputs, labels in test_loader:
        input_dataset.append(inputs)
        input_labels.append(labels)
    input_dataset = torch.stack(input_dataset).to(configs.DEVICE)
    input_labels = torch.stack(input_labels).to(configs.DEVICE)

    return input_dataset, input_labels


def load_model(args: argparse.Namespace):
    checkpoint_path = f"{args.checkpointdir}/{args.ckpt}"

    # It is a mobile net or common resnet
    if args.name in [configs.MobileNetV2x14Cifar10, configs.MobileNetV2x14Cifar100]:
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", args.name, pretrained=False)
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        # Build model (Resnet only up to now)
        optim_params = {'optimizer': args.optimizer, 'epochs': args.epochs, 'lr': args.lr, 'wd': args.wd}
        n_classes = configs.CLASSES[args.dataset]

        model = build_model(model=args.model, n_classes=n_classes, optim_params=optim_params,
                            loss=args.loss, inject_p=args.inject_p,
                            order=args.order, activation=args.activation, affine=args.affine)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.model
    return model


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    """ Parse the args and return an args namespace and the tostring from the args    """
    config_parser = argparse.ArgumentParser(description='PyTorch DNN radiation setup', add_help=False)
    config_parser.add_argument('--config', default='', type=str, metavar='FILE',
                               help='YAML config file specifying default arguments.')
    args, remaining_argv = config_parser.parse_known_args()

    defaults = {"option": "default"}

    if args.config:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
        defaults.update(**cfg)

    # Parse rest of arguments
    # Don't suppress add_help here, so it will handle -h
    parser = argparse.ArgumentParser(
        # Inherit options from config_parser
        parents=[config_parser]
    )
    parser.set_defaults(**defaults)

    # parser = argparse.ArgumentParser(description='PyTorch DNN radiation setup')
    parser.add_argument('--iterations', default=int(1e12), help="Iterations to run forever", type=int)
    parser.add_argument('--testsamples', default=100, help="Test samples to be used in the test.", type=int)
    parser.add_argument('--downloaddataset', default=False, action="store_true",
                        help="Set to download the dataset, default is to not download. Needs internet.")
    parser.add_argument('--generate', default=False, action="store_true", help="Set this flag to generate the gold")
    parser.add_argument('--disableconsolelog', default=False, action="store_true",
                        help="Set this flag disable console logging")
    parser.add_argument('--goldpath', help="Path to the gold file")
    parser.add_argument('--checkpointdir', help="Path to checkpoint dir")
    parser.add_argument('--datadir', help="Path to data directory that contains the dataset")

    args = parser.parse_args()
    # Check if the model is correct
    if args.name not in configs.ALL_DNNS:
        parser.print_help()
        raise ValueError(f"Not the correct model {args.name}, valids are:" + ", ".join(configs.ALL_DNNS))

    if args.testsamples % args.batch_size != 0:
        raise ValueError("Test samples should be multiple of batch size")
    # Check if it is only to generate the gold values
    if args.generate:
        args.iterations = 1

    args_text_list = [f"{k}={v}" for k, v in vars(args).items()]
    return args, args_text_list


def log_and_crash(fatal_string: str) -> None:
    caller_frame_record = inspect.stack()[1]  # 0 represents this line
    # 1 represents line at caller
    frame = caller_frame_record[0]
    info = inspect.getframeinfo(frame)
    dnn_log_helper.log_info_detail(f"SETUP_ERROR:{fatal_string} FILE:{info.filename}:{info.lineno} F:{info.function}")
    traceback_str = traceback.format_exc()
    # It is better to always show the exception
    print(traceback_str)
    dnn_log_helper.log_info_detail(traceback_str)
    dnn_log_helper.end_log_file()
    sys.exit(1)


def equal(rhs: torch.Tensor, lhs: torch.Tensor, threshold: float = None) -> bool:
    """ Compare based or not in a threshold, if threshold is none then it is equal comparison    """
    if threshold:
        return bool(torch.all(torch.le(torch.abs(torch.subtract(rhs, lhs)), threshold)))
    else:
        return bool(torch.equal(rhs, lhs))


def compare_classification(output_tensor: torch.tensor,
                           golden_tensor: torch.tensor,
                           golden_top_k_labels: torch.tensor,
                           batch_id: int,
                           top_k: int,
                           output_logger: logging.Logger = None) -> int:
    # Make sure that they are on CPU
    out_is_cuda, golden_is_cuda = output_tensor.is_cuda, golden_tensor.is_cuda
    if out_is_cuda or golden_is_cuda:
        log_and_crash(fatal_string=f"Tensors are not on CPU. OUT IS CUDA:{out_is_cuda} GOLDEN IS CUDA:{golden_is_cuda}")

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
    if equal(rhs=output_tensor, lhs=golden_tensor, threshold=configs.CLASSIFICATION_ABS_THRESHOLD) is True:
        return output_errors

    # ------------ Check the size of the tensors
    if output_tensor.shape != golden_tensor.shape:
        info_detail = f"shape-diff g:{golden_tensor.shape} o:{output_tensor.shape}"
        if output_logger:
            output_logger.error(info_detail)
        dnn_log_helper.log_info_detail(info_detail)

    # Iterate over the batches
    for img_id, (output_batch, golden_batch, golden_batch_label) in enumerate(
            zip(output_tensor, golden_tensor, golden_top_k_labels)):
        # using the same approach as the detection, compare only the positions that differ
        if equal(rhs=golden_batch, lhs=output_batch, threshold=configs.CLASSIFICATION_ABS_THRESHOLD) is False:
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
                    error_detail_ctr = f"batch:{batch_id} critical-img:{img_id} i:{i} g:{tpk_gold:.6e} o:{tpk_found}"
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


def forward(batched_input: torch.tensor, model: torch.nn.Module, model_name: str):
    if model_name in [configs.MobileNetV2x14Cifar10, configs.MobileNetV2x14Cifar100]:
        return model(batched_input)
    else:
        return model(batched_input, inject=False)


def main():
    # Disable all torch grad
    torch.set_grad_enabled(mode=False)
    args, args_text_list = parse_args()
    # Starting the setup
    generate = args.generate
    args_text_list.append(f"GPU:{torch.cuda.get_device_name()}")
    dnn_log_helper.start_setup_log_file(framework_name="PyTorch", args_conf=args_text_list,
                                        dnn_name=args.name.strip("_"), activate_logging=not generate)
    if torch.cuda.is_available() is False:
        log_and_crash(fatal_string=f"Device {configs.DEVICE} not available.")
    dev_capability = torch.cuda.get_device_capability()
    if dev_capability[0] < configs.MINIMUM_DEVICE_CAPABILITY:
        log_and_crash(fatal_string=f"Device cap:{dev_capability} is too old.")

    # Defining a timer
    timer = Timer()
    batch_size = args.batch_size
    test_sample = args.testsamples
    data_dir = args.datadir
    dataset = args.dataset
    download_dataset = args.downloaddataset
    gold_path = args.goldpath
    iterations = args.iterations

    # Load the model
    model = load_model(args=args)
    model.eval()
    model = model.to(configs.DEVICE)
    # First step is to load the inputs in the memory
    timer.tic()
    input_list, input_labels = load_dataset(batch_size=batch_size, dataset=dataset, data_dir=data_dir,
                                            test_sample=test_sample, download_dataset=download_dataset)
    timer.toc()
    input_load_time = timer.diff_time_str
    # Terminal console
    main_logger_name = str(os.path.basename(__file__)).replace(".py", "")
    terminal_logger = console_logger.ColoredLogger(main_logger_name) if args.disableconsolelog is False else None

    # Load if it is not a gold generating op
    golden = dict(prob_list=list(), top_k_labels=list())
    timer.tic()
    if generate is False:
        golden = torch.load(gold_path)
    timer.toc()
    golden_load_diff_time = timer.diff_time_str

    if terminal_logger:
        terminal_logger.debug("\n".join(args_text_list))
        terminal_logger.debug(f"Time necessary to load the inputs: {input_load_time}")
        terminal_logger.debug(f"Time necessary to load the golden outputs: {golden_load_diff_time}")

    # Main setup loop
    setup_iteration = 0
    while setup_iteration < iterations:
        total_errors = 0
        # Loop over the input list
        for batch_id, batched_input in enumerate(input_list):
            timer.tic()
            dnn_log_helper.start_iteration()
            dnn_output = forward(batched_input=batched_input, model=model, model_name=args.name)
            dnn_log_helper.end_iteration()
            timer.toc()
            kernel_time = timer.diff_time
            # Always copy to CPU
            timer.tic()
            probabilities = dnn_output.to("cpu")
            timer.toc()
            copy_to_cpu_time = timer.diff_time
            # Then compare the golden with the output
            timer.tic()
            errors = 0
            if generate is False:
                errors = compare_classification(output_tensor=probabilities,
                                                golden_tensor=golden["prob_list"][batch_id],
                                                golden_top_k_labels=golden["top_k_labels"][batch_id],
                                                batch_id=batch_id,
                                                top_k=configs.CLASSIFICATION_CRITICAL_TOP_K,
                                                output_logger=terminal_logger)
                total_errors += errors
            else:
                golden["prob_list"].append(probabilities)
                golden["top_k_labels"].append(
                    torch.tensor(
                        [torch.topk(output_batch, k=configs.CLASSIFICATION_CRITICAL_TOP_K).indices.squeeze(0)
                         for output_batch in dnn_output])
                )
            timer.toc()
            comparison_time = timer.diff_time

            # Reload all the memories after error
            if total_errors != 0:
                if terminal_logger:
                    terminal_logger.info("RELOADING THE MODEL AND THE INPUTS AFTER ERROR")
                del input_list
                del model
                model = load_model(args=args)
                model.eval()
                model = model.to(configs.DEVICE)
                input_list, input_labels = load_dataset(batch_size=batch_size, dataset=dataset, data_dir=data_dir,
                                                        test_sample=test_sample, download_dataset=download_dataset)

            # Printing timing information
            if terminal_logger:
                wasted_time = comparison_time + copy_to_cpu_time
                time_pct = (wasted_time / (comparison_time + kernel_time + copy_to_cpu_time)) * 100.0
                iteration_out = f"It:{setup_iteration:<3} batch_id:{batch_id:<3} inference time:{kernel_time:.5f}, "
                iteration_out += f"compare time:{comparison_time:.5f} copy time:{copy_to_cpu_time:.5f} "
                iteration_out += f"({time_pct:.1f}%) errors:{errors}"
                terminal_logger.debug(iteration_out)

        setup_iteration += 1

    if generate is True:
        torch.save(golden, gold_path)
    if terminal_logger:
        terminal_logger.debug("Finish computation.")

    dnn_log_helper.end_log_file()


if __name__ == '__main__':
    try:
        main()
    except Exception as main_function_exception:
        log_and_crash(fatal_string=f"EXCEPTION:{main_function_exception}")
