#!/usr/bin/python3
import argparse
import os

import torch
import torchvision
from typing import Tuple

import configs
import console_logger
import dnn_log_helper
from configs import Timer
from diehardnet.pytorch_scripts.utils import build_model


def load_dataset(batch_size: int, dataset: str, data_dir: str) -> torch.utils.data.DataLoader:
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

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return test_loader


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


def compare_output_with_gold(gold_probabilities_list, probabilities) -> int:
    top1_label = int(torch.topk(probabilities, k=1).indices.squeeze(0))
    top1_prob = torch.softmax(probabilities, dim=1)[0, top1_label].item()
    gold_probabilities = gold_probabilities_list[i]
    gold_top1_label = int(torch.topk(gold_probabilities, k=1).indices.squeeze(0))
    gold_top1_prob = torch.softmax(gold_probabilities, dim=1)[0, gold_top1_label].item()
    cmp_gold_prob = torch.flatten(gold_probabilities)
    cmp_out_prob = torch.flatten(probabilities)
    if torch.any(torch.not_equal(cmp_gold_prob, cmp_out_prob)):
        for it, (g, f) in enumerate(zip(cmp_gold_prob, cmp_out_prob)):
            if g != f:
                print(f"{it} e:{g} r:{f}")
        if gold_top1_label != top1_label:
            print(f"Critical SDC detected. "
                  f"e_label:{gold_top1_label} r_label:{top1_label} "
                  f"e_prob:{gold_top1_prob} r_prob:{top1_prob}")
    return 0


def main():
    # Defining a timer
    timer = Timer()
    timer.tic()
    # Disable all torch grad
    torch.set_grad_enabled(mode=False)
    args, args_text = parse_args()
    # Starting the setup
    # Check the available device
    device = "cuda:0"
    dnn_log_helper.set_iter_interval_print(30)
    if torch.cuda.is_available() is False:
        raise RuntimeError("GPU is not available")
    # Load the model
    model = load_ptl_model(args=args)
    model.eval()
    model = model.to(device)
    main_logger_name = str(os.path.basename(__file__)).replace(".py", "")
    output_logger = console_logger.ColoredLogger(main_logger_name)
    args, args_conf = parse_args()
    for k, v in vars(args).items():
        output_logger.debug(f"{k}:{v}")
    generate = args.generate
    iterations = args.iterations
    gold_path = args.goldpath
    model_name = args.model
    disable_console_logger = args.disableconsolelog
    batch_size = args.batchsize
    data_dir = args.datadir
    dataset = args.dataset

    # First step is to load the inputs in the memory
    timer.tic()
    input_list = load_dataset(batch_size=batch_size, dataset=dataset, data_dir=data_dir)
    timer.toc()
    output_logger.debug(f"Time necessary to load the inputs: {timer}")

    # Load if it is not a gold generating op
    gold_probabilities_list = list()
    if generate is False:
        timer.tic()
        gold_probabilities_list = torch.load(gold_path)
        timer.toc()
        output_logger.debug(f"Time necessary to load the golden outputs: {timer}")

    # Start the setup
    dnn_log_helper.start_setup_log_file(framework_name="PyTorch", args_conf=args_conf, model_name=model_name,
                                        max_errors_per_iteration=configs.MAXIMUM_ERRORS_PER_ITERATION,
                                        generate=generate)

    # Main setup loop
    setup_iteration = 0
    while setup_iteration < iterations:
        total_errors = 0
        # Loop over the input list
        for batched_input in input_list:
            image_gpu = batched_input.to("cuda")
            timer.tic()
            dnn_log_helper.start_iteration()
            dnn_output = model(image_gpu, inject=False)
            dnn_log_helper.end_iteration()
            timer.toc()
            kernel_time = timer.diff_time

            probabilities = dnn_output.to("cpu")
            # Then compare the golden with the output
            timer.tic()
            errors = 0
            if generate is False:
                errors = compare_output_with_gold(gold_probabilities_list, probabilities)
            else:
                gold_probabilities_list.append(probabilities)
            total_errors += errors
            timer.toc()
            comparison_time = timer.diff_time

            iteration_out = f"It:{setup_iteration:<3} imgit:{img_i:<3}"
            iteration_out += f" inference time:{kernel_time:.5f}"
            time_pct = (comparison_time / (comparison_time + kernel_time)) * 100.0
            iteration_out += f", gold compare time:{comparison_time:.5f} ({time_pct:.1f}%) errors:{errors}"
            output_logger.debug(iteration_out)

        # Reload after error
        if total_errors != 0:
            del input_list
            del model
            model = load_ptl_model(args=args)
            model.eval()
            model = model.to(device)
            input_list = load_dataset(batch_size=batch_size, dataset=dataset, data_dir=data_dir)

        setup_iteration += 1

    if generate is True:
        torch.save(gold_probabilities_list, gold_path)
    print("Finish computation.")


if __name__ == '__main__':
    main()
