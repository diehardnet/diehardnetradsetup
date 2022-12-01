#!/usr/bin/python3
import argparse
import os
from typing import Tuple, List

import torch
import torchvision
import yaml

import common
import configs
import console_logger
import dnn_log_helper
from classification_functions import load_model, compare_classification, check_dnn_accuracy
from configs import DIEHARDNET_TRANS_LEARNING_CONFIGS


def load_dataset(batch_size: int, dataset: str, data_dir: str, test_sample: int,
                 download_dataset: bool) -> Tuple[List, List]:
    transform_resize = (224, 224)
    if dataset in [configs.CIFAR10, configs.CIFAR100]:
        transform_resize = (32, 32)

    transform = torchvision.transforms.Compose([torchvision.transforms.Resize(transform_resize),
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                 std=[0.229, 0.224, 0.225])])
    if dataset == configs.CIFAR10:
        test_set = torchvision.datasets.cifar.CIFAR10(root=data_dir, download=download_dataset, train=False,
                                                      transform=transform)
    elif dataset == configs.CIFAR100:
        test_set = torchvision.datasets.cifar.CIFAR100(root=data_dir, download=download_dataset, train=False,
                                                       transform=transform)
    elif dataset == configs.IMAGENET:
        test_set = torchvision.datasets.imagenet.ImageNet(root=configs.IMAGENET_DATASET_DIR, transform=transform,
                                                          split='val')
    else:
        raise ValueError(f"Incorrect dataset {dataset}")

    subset = torch.utils.data.SequentialSampler(range(test_sample))
    test_loader = torch.utils.data.DataLoader(dataset=test_set, sampler=subset, batch_size=batch_size,
                                              shuffle=False, pin_memory=True)
    input_dataset, input_labels = list(), list()
    for inputs, labels in test_loader:
        # Only the inputs must be in the device
        input_dataset.append(inputs.to(configs.DEVICE))
        input_labels.append(labels)
    # Fixed, no need to stack if they will only be used in the host side
    # input_dataset = torch.stack(input_dataset).to(configs.DEVICE)
    # Fixed, only the input must be in the GPU
    # input_labels = torch.stack(input_labels).to(configs.DEVICE)

    return input_dataset, input_labels


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
    # if args.name not in configs.ALL_DNNS:
    #     parser.print_help()
    #     raise ValueError(f"Not the correct model {args.name}, valids are:" + ", ".join(configs.ALL_DNNS))

    if args.testsamples % args.batch_size != 0:
        raise ValueError("Test samples should be multiple of batch size")
    # Check if it is only to generate the gold values
    if args.generate:
        args.iterations = 1

    args_text_list = [f"{k}={v}" for k, v in vars(args).items()]
    return args, args_text_list


def forward(batched_input: torch.tensor, model: torch.nn.Module, model_name: str):
    if model_name in DIEHARDNET_TRANS_LEARNING_CONFIGS:
        return model(batched_input)
    else:
        return model(batched_input, inject=False)


def main():
    args, args_text_list = parse_args()
    # Starting the setup
    generate = args.generate
    args_text_list.append(f"GPU:{torch.cuda.get_device_name()}")
    # Define DNN goal
    dnn_goal = configs.DNN_GOAL[args.name]

    dnn_log_helper.start_setup_log_file(framework_name="PyTorch", args_conf=args_text_list,
                                        dnn_name=args.name.strip("_"), activate_logging=not generate, dnn_goal=dnn_goal)

    # Disable all torch grad
    torch.set_grad_enabled(mode=False)
    if torch.cuda.is_available() is False:
        dnn_log_helper.log_and_crash(fatal_string=f"Device {configs.DEVICE} not available.")
    dev_capability = torch.cuda.get_device_capability()
    if dev_capability[0] < configs.MINIMUM_DEVICE_CAPABILITY:
        dnn_log_helper.log_and_crash(fatal_string=f"Device cap:{dev_capability} is too old.")

    # Defining a timer
    timer = common.Timer()
    batch_size = args.batch_size
    test_sample = args.testsamples
    data_dir = args.datadir
    dataset = args.dataset
    download_dataset = args.downloaddataset
    gold_path = args.goldpath
    iterations = args.iterations

    # Load the model
    model = load_model(args=args)
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
            torch.cuda.synchronize(device=configs.DEVICE)
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
                                                ground_truth_labels=input_labels[batch_id],
                                                batch_id=batch_id,
                                                top_k=configs.CLASSIFICATION_CRITICAL_TOP_K,
                                                output_logger=terminal_logger)
                total_errors += errors
            else:
                golden["prob_list"].append(probabilities)
                golden["top_k_labels"].append(
                    torch.tensor(
                        [torch.topk(output_batch, k=configs.CLASSIFICATION_CRITICAL_TOP_K).indices.squeeze(0)
                         for output_batch in probabilities])
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
        check_dnn_accuracy(predicted=golden["top_k_labels"], ground_truth=input_labels, output_logger=terminal_logger)
    if terminal_logger:
        terminal_logger.debug("Finish computation.")

    dnn_log_helper.end_log_file()


if __name__ == '__main__':
    try:
        main()
    except Exception as main_function_exception:
        dnn_log_helper.log_and_crash(fatal_string=f"EXCEPTION:{main_function_exception}")
