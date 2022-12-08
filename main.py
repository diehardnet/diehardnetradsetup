#!/usr/bin/python3
import argparse
import collections
import logging
import os
import random
import re
import time
from typing import Tuple, List, Dict, Union

import torch
import torchvision
import yaml
import timm

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


def load_model(args: argparse.Namespace) -> [torch.nn.Module, torchvision.transforms.Compose]:
    checkpoint_path = f"{args.checkpointdir}/{args.ckpt}"

    resize_size, model = None, None
    # First option is the baseline option
    transform = None
    if args.name in configs.RESNET50_IMAGENET_1K_V2_BASE:
        # Better for offline approach
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        resize_size = (224, 224)
        model = torchvision.models.resnet50(weights=weights)
        model.load_state_dict(torch.load(checkpoint_path))
    elif args.name in configs.DIEHARDNET_SEGMENTATION_CONFIGS:
        # Better for offline approach
        model = None
        if args.name == configs.DEEPLABV3_RESNET50:
            model = torchvision.models.segmentation.deeplabv3_resnet50(
                weights=torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
        elif args.name == configs.DEEPLABV3_RESNET101:
            model = torchvision.models.segmentation.deeplabv3_resnet101(
                weights=torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1)
        elif args.name == configs.FCN_RESNET50:
            model = torchvision.models.segmentation.fcn_resnet50(
                weights=torchvision.models.segmentation.FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
        elif args.name == configs.FCN_RESNET101:
            model = torchvision.models.segmentation.fcn_resnet101(
                weights=torchvision.models.segmentation.FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1)

        resize_size = (520, 520)
        model.load_state_dict(torch.load(checkpoint_path))
    elif args.name in configs.DIEHARDNET_VITS_CONFIGS:
        model = timm.create_model(args.model, pretrained=True)
        config = timm.data.resolve_data_config({}, model=model)
        transform = timm.data.transforms_factory.create_transform(**config)
    elif args.name in configs.DIEHARDNET_CLASSIFICATION_CONFIGS:
        resize_size = (32, 32)
        # Build model (Resnet only up to now)
        optim_params = {'optimizer': args.optimizer, 'epochs': args.epochs, 'lr': args.lr, 'wd': args.wd}
        n_classes = configs.CLASSES[args.dataset]
        # model='hard_resnet20', n_classes=10, optim_params={}, loss='bce', error_model='random',
        # inject_p=0.1, inject_epoch=0, order='relu-bn', activation='relu', nan=False, affine=True
        model = build_model(model=args.model, n_classes=n_classes, optim_params=optim_params,
                            loss=args.loss, inject_p=args.inject_p, order=args.order, activation=args.activation,
                            affine=bool(args.affine), nan=bool(args.nan))
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.model
    else:
        dnn_log_helper.log_and_crash(fatal_string=f"{args.name} model invalid")
    model.eval()
    model = model.to(configs.DEVICE)
    # Default values for CIFAR 10 and 100
    if transform is None:
        transform = torchvision.transforms.Compose([torchvision.transforms.Resize(size=resize_size),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                     std=[0.229, 0.224, 0.225])])


    return model, transform


def load_dataset(batch_size: int, dataset: str, test_sample: int,
                 transform: torchvision.transforms.Compose) -> Tuple[List, List]:
    test_set = None
    if dataset == configs.CIFAR10:
        test_set = torchvision.datasets.cifar.CIFAR10(root=configs.CIFAR_DATASET_DIR, download=False, train=False,
                                                      transform=transform)
    elif dataset == configs.CIFAR100:
        test_set = torchvision.datasets.cifar.CIFAR100(root=configs.CIFAR_DATASET_DIR, download=False, train=False,
                                                       transform=transform)
    elif dataset == configs.IMAGENET:
        test_set = torchvision.datasets.imagenet.ImageNet(root=configs.IMAGENET_DATASET_DIR, transform=transform,
                                                          split='val')
    elif dataset == configs.COCO:
        # This is only used when performing det/seg and these models already perform transforms
        test_set = torchvision.datasets.coco.CocoDetection(root=configs.COCO_DATASET_VAL,
                                                           annFile=configs.COCO_DATASET_ANNOTATIONS,
                                                           transform=transform)
    else:
        dnn_log_helper.log_and_crash(fatal_string=f"Incorrect dataset {dataset}")

    # noinspection PyUnresolvedReferences
    subset = torch.utils.data.SequentialSampler(range(test_sample))
    input_dataset, input_labels = list(), list()

    # Data loader diff if it is coco
    if dataset == configs.COCO:
        # noinspection PyUnresolvedReferences
        test_loader = torch.utils.data.DataLoader(dataset=test_set, sampler=subset, batch_size=batch_size,
                                                  shuffle=False, pin_memory=True, collate_fn=lambda x: x)
        for iterator in test_loader:
            inputs, labels = list(), list()
            for input_i, label_i in iterator:
                inputs.append(input_i)
                labels.append(label_i)
            # Only the inputs must be in the device
            input_dataset.append(torch.stack(inputs).to(configs.DEVICE))
            # Labels keys dict_keys(['segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id'])
            input_labels.append(labels)
    else:
        # noinspection PyUnresolvedReferences
        test_loader = torch.utils.data.DataLoader(dataset=test_set, sampler=subset, batch_size=batch_size,
                                                  shuffle=False, pin_memory=True)
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
    parser.add_argument('--generate', default=False, action="store_true", help="Set this flag to generate the gold")
    parser.add_argument('--disableconsolelog', default=False, action="store_true",
                        help="Set this flag disable console logging")
    parser.add_argument('--goldpath', help="Path to the gold file")
    parser.add_argument('--checkpointdir', help="Path to checkpoint dir")

    args = parser.parse_args()

    if args.testsamples % args.batch_size != 0:
        dnn_log_helper.log_and_crash(fatal_string="Test samples should be multiple of batch size")
    # double check in the names
    base_config_name = os.path.basename(args.config)
    if base_config_name.replace(".yaml", "") != args.name:
        dnn_log_helper.log_and_crash(
            fatal_string=f"DNN name must have the same name as the config file, now:{args.name} != {base_config_name}")

    # Check if it is only to generate the gold values
    if args.generate:
        args.iterations = 1

    args_text_list = [f"{k}={v}" for k, v in vars(args).items()]
    return args, args_text_list


def equal(rhs: torch.Tensor, lhs: torch.Tensor, threshold: float = 0) -> bool:
    """ Compare based or not in a threshold, if threshold is none then it is equal comparison    """
    if threshold > 0:
        return bool(torch.all(torch.le(torch.abs(torch.subtract(rhs, lhs)), threshold)))
    else:
        return bool(torch.equal(rhs, lhs))


def compare_classification(output_tensor: torch.tensor,
                           golden_tensor: torch.tensor,
                           golden_top_k_labels: torch.tensor,
                           ground_truth_labels: torch.tensor,
                           batch_id: int,
                           top_k: int,
                           output_logger: logging.Logger) -> int:
    output_errors = 0

    # Iterate over the batches
    for img_id, (output_batch, golden_batch, golden_batch_label, ground_truth_label) in enumerate(
            zip(output_tensor, golden_tensor, golden_top_k_labels, ground_truth_labels)):
        # using the same approach as the detection, compare only the positions that differ
        if equal(rhs=golden_batch, lhs=output_batch, threshold=configs.CLASSIFICATION_ABS_THRESHOLD) is False:
            # ------------ Check if there is a Critical error ----------------------------------------------------------
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

    return output_errors


def compare_segmentation(output_tensor: torch.tensor,
                         golden_tensor: torch.tensor,
                         batch_id: int,
                         output_logger: logging.Logger,
                         setup_iteration: int) -> int:
    output_errors = 0
    abs_threshold = configs.SEGMENTATION_ABS_THRESHOLD
    for img_id, (output_batch, golden_batch) in enumerate(zip(output_tensor, golden_tensor)):
        # ------------ Check error on the whole output -------------------------------------------------------------
        # Count only and save the output file
        if abs_threshold == 0:
            equal_elements = torch.sum(torch.eq(output_batch, golden_batch))
        else:
            equal_elements = torch.sum(torch.le(torch.abs(torch.subtract(output_batch, golden_batch)), abs_threshold))

        diff_elements = abs(equal_elements - golden_batch.shape[0])
        if diff_elements != 0:
            # Save the batch
            log_helper_file = re.match(r".*LOCAL:(\S+).log.*", dnn_log_helper.log_file_name).group(1)
            # Random int will guarantee that files are not overriden (I hope so)
            random_int = random.randrange(configs.RANDOM_INT_LIMIT)
            full_sdc_file_name = f"{log_helper_file}_sdc_{batch_id}_{img_id}_{setup_iteration}_{random_int}.pt"
            error_detail_out = f"batch:{batch_id} img:{img_id} has {diff_elements} different elements."
            error_detail_out += f" Saving data on {full_sdc_file_name} file."
            # Clone before saving, because using views will save the entire tensor
            torch.save(torch.clone(output_batch), full_sdc_file_name)
            if output_logger:
                output_logger.error(error_detail_out)
            dnn_log_helper.log_error_detail(error_detail_out)
            output_errors += 1

    return output_errors


def compare(output_tensor: torch.tensor,
            golden: Dict[str, List[torch.tensor]],
            ground_truth_labels: Union[List[torch.tensor], List[dict]],
            batch_id: int,
            output_logger: logging.Logger, dnn_goal: str, setup_iteration: int):
    golden_tensor = golden["output_list"][batch_id]
    # global TEST
    # TEST += 1
    # if TEST == 3:
    #     # # Simulate a non-critical error
    #     output_tensor[8, 0] *= 0.9
    #     # Simulate a critical error
    #     # output_tensor[55, 0] = 39304
    #     # # Shape SDC
    #     # output_tensor = torch.reshape(output_tensor, (4, 3200))

    # Make sure that they are on CPU
    out_is_cuda, golden_is_cuda = output_tensor.is_cuda, golden_tensor.is_cuda
    if out_is_cuda or golden_is_cuda:
        dnn_log_helper.log_and_crash(
            fatal_string=f"Tensors are not on CPU. OUT IS CUDA:{out_is_cuda} GOLDEN IS CUDA:{golden_is_cuda}")

    # First check if the tensors are equal or not
    if equal(rhs=output_tensor, lhs=golden_tensor, threshold=configs.CLASSIFICATION_ABS_THRESHOLD) is True:
        return 0

    # ------------ Check the size of the tensors
    if output_tensor.shape != golden_tensor.shape:
        info_detail = f"shape-diff g:{golden_tensor.shape} o:{output_tensor.shape}"
        if output_logger:
            output_logger.error(info_detail)
        dnn_log_helper.log_info_detail(info_detail)

    # ------------ Main check
    output_errors = 0
    if dnn_goal == configs.CLASSIFICATION:
        golden_top_k_labels = golden["top_k_labels"][batch_id]
        output_errors = compare_classification(output_tensor=output_tensor,
                                               golden_tensor=golden_tensor,
                                               golden_top_k_labels=golden_top_k_labels,
                                               ground_truth_labels=ground_truth_labels[batch_id],
                                               batch_id=batch_id,
                                               top_k=configs.CLASSIFICATION_CRITICAL_TOP_K,
                                               output_logger=output_logger)

    elif dnn_goal == configs.SEGMENTATION:
        output_errors = compare_segmentation(output_tensor=output_tensor,
                                             golden_tensor=golden_tensor,
                                             batch_id=batch_id,
                                             output_logger=output_logger, setup_iteration=setup_iteration)
    # ------------ log and return
    if output_errors != 0:
        dnn_log_helper.log_error_count(error_count=output_errors)
    return output_errors


def check_dnn_accuracy(predicted: Union[Dict[str, List[torch.tensor]], torch.tensor], ground_truth: List[torch.tensor],
                       output_logger: logging.Logger, dnn_goal: str) -> None:
    correct, gt_count = 0, 0
    if dnn_goal == configs.CLASSIFICATION:
        predicted = predicted["top_k_labels"]
        for pred, gt in zip(predicted, ground_truth):
            gt_count += gt.shape[0]
            correct += torch.sum(torch.eq(pred, gt))
    elif dnn_goal == configs.SEGMENTATION:
        for pred_i, gt in zip(predicted, ground_truth):
            # TODO: Calculate the IoU here
            gt_count += len(gt)

    if output_logger:
        output_logger.debug(f"Correct predicted samples:{correct} - ({(correct / gt_count) * 100:.2f}%)")


def update_golden(golden: torch.tensor, output: torch.tensor, dnn_goal: str) -> Dict[str, list]:
    if dnn_goal == configs.CLASSIFICATION:
        golden["output_list"].append(output)
        golden["top_k_labels"].append(
            torch.tensor([torch.topk(output_batch, k=configs.CLASSIFICATION_CRITICAL_TOP_K).indices.squeeze(0)
                          for output_batch in output])
        )
    elif dnn_goal == configs.SEGMENTATION:
        golden["output_list"].append(output)

    return golden


def copy_output_to_cpu(dnn_output: Union[torch.tensor, collections.OrderedDict],
                       dnn_goal: str) -> torch.tensor:
    if dnn_goal == configs.CLASSIFICATION:
        return dnn_output.to("cpu")
    elif dnn_goal == configs.SEGMENTATION:
        return dnn_output["out"].to('cpu')


def forward(batched_input: torch.tensor, model: torch.nn.Module, model_name: str) -> torch.tensor:
    pretrained_configs = configs.DIEHARDNET_TRANS_LEARNING_CONFIGS + configs.DIEHARDNET_SEGMENTATION_CONFIGS
    pretrained_configs +=  configs.DIEHARDNET_VITS_CONFIGS
    if model_name in pretrained_configs:
        return model(batched_input)
    else:
        return model(batched_input, inject=False)


def show(all_batches_output: torch.tensor, all_batches_input: torch.tensor, batch_id: int, model: str):
    import matplotlib.pyplot as plt
    import numpy as np
    import torchvision.transforms.functional as F
    plt.rcParams["savefig.bbox"] = 'tight'

    normalized_masks = torch.nn.functional.softmax(all_batches_output, dim=1)
    num_classes = normalized_masks.shape[1]
    for batch, classes in enumerate(normalized_masks):
        dog1_masks = normalized_masks[batch]
        class_dim = 0
        dog1_all_classes_masks = dog1_masks.argmax(class_dim) == torch.arange(num_classes)[:, None, None]
        dog1_int = all_batches_input[batch].to("cpu").type(torch.uint8)
        # noinspection PyTypeChecker
        dog_with_all_masks = torchvision.utils.draw_segmentation_masks(dog1_int, masks=dog1_all_classes_masks)
        fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False)
        dog_with_all_masks = np.asarray(F.to_pil_image(dog_with_all_masks.detach()))
        ax[0, 0].imshow(dog_with_all_masks)
        plt.savefig(f"./{model}_test{batch}_{batch_id}.jpg")
    # for batch, classes in enumerate(normalized_masks):
    # fig, axs = plt.subplots(ncols=5, nrows=5, squeeze=False)
    # iterator = iter(classes)
    # for i in range(5):
    #     for j in range(5):
    #         if i * 5 + j >= len(classes):
    #             break
    #         img = next(iterator)
    #         img = img.detach()
    #         img = F.to_pil_image(img)
    #         axs[i, j].imshow(np.asarray(img))
    #         axs[i, j].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    # img_input = all_batches_input[batch].to("cpu").detach()
    # img_input = np.asarray(F.to_pil_image(img_input))
    # axs[4, 2].imshow(img_input)
    # plt.savefig(f"./test{batch}.jpg")


def main():
    args, args_text_list = parse_args()
    # Starting the setup
    generate = args.generate
    args_text_list.append(f"GPU:{torch.cuda.get_device_name()}")
    # Define DNN goal
    dnn_goal = configs.DNN_GOAL[args.name]

    dnn_log_helper.start_setup_log_file(framework_name="PyTorch", framework_version=str(torch.__version__),
                                        args_conf=args_text_list, dnn_name=args.name, activate_logging=not generate,
                                        dnn_goal=dnn_goal)

    # Disable all torch grad
    torch.set_grad_enabled(mode=False)
    if torch.cuda.is_available() is False:
        dnn_log_helper.log_and_crash(fatal_string=f"Device {configs.DEVICE} not available.")
    dev_capability = torch.cuda.get_device_capability()
    if dev_capability[0] < configs.MINIMUM_DEVICE_CAPABILITY:
        dnn_log_helper.log_and_crash(fatal_string=f"Device cap:{dev_capability} is too old.")

    # Defining a timer
    timer = Timer()
    batch_size = args.batch_size
    test_sample = args.testsamples
    dataset = args.dataset
    gold_path = args.goldpath
    iterations = args.iterations

    # Load the model
    model, transform = load_model(args=args)
    # First step is to load the inputs in the memory
    timer.tic()
    input_list, input_labels = load_dataset(batch_size=batch_size, dataset=dataset, test_sample=test_sample,
                                            transform=transform)
    timer.toc()
    input_load_time = timer.diff_time_str
    # Terminal console
    main_logger_name = str(os.path.basename(__file__)).replace(".py", "")
    terminal_logger = console_logger.ColoredLogger(main_logger_name) if args.disableconsolelog is False else None

    # Load if it is not a gold generating op
    golden: Dict[str, List[torch.tensor]] = dict(output_list=list(), top_k_labels=list())
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
        # Loop over the input list
        batch_id = 0 # It must be like this, because I may reload the list in the middle of the process
        while batch_id < len(input_list):
            timer.tic()
            dnn_log_helper.start_iteration()
            dnn_output = forward(batched_input=input_list[batch_id], model=model, model_name=args.name)
            torch.cuda.synchronize(device=configs.DEVICE)
            dnn_log_helper.end_iteration()
            timer.toc()
            kernel_time = timer.diff_time
            # Always copy to CPU
            timer.tic()
            dnn_output_cpu = copy_output_to_cpu(dnn_output=dnn_output, dnn_goal=dnn_goal)
            timer.toc()
            copy_to_cpu_time = timer.diff_time
            # Then compare the golden with the output
            timer.tic()
            errors = 0
            if generate is False:
                errors = compare(output_tensor=dnn_output_cpu,
                                 golden=golden,
                                 ground_truth_labels=input_labels,
                                 batch_id=batch_id,
                                 output_logger=terminal_logger, dnn_goal=dnn_goal, setup_iteration=setup_iteration)
            else:
                golden = update_golden(golden=golden, output=dnn_output_cpu, dnn_goal=dnn_goal)
                # show(all_batches_output=dnn_output_cpu, all_batches_input=batched_input, batch_id=batch_id,
                #      model=args.model)

            timer.toc()
            comparison_time = timer.diff_time

            # Reload all the memories after error
            if errors != 0:
                if terminal_logger:
                    terminal_logger.info("RELOADING THE MODEL AND THE INPUTS AFTER ERROR")
                del input_list
                del model
                # Free cuda memory
                torch.cuda.empty_cache()
                model, _ = load_model(args=args)
                input_list, input_labels = load_dataset(batch_size=batch_size, dataset=dataset, test_sample=test_sample,
                                                        transform=transform)

            # Printing timing information
            if terminal_logger:
                wasted_time = comparison_time + copy_to_cpu_time
                time_pct = (wasted_time / (wasted_time + kernel_time)) * 100.0
                iteration_out = f"It:{setup_iteration:<3} batch_id:{batch_id:<3} inference time:{kernel_time:.5f}, "
                iteration_out += f"compare time:{comparison_time:.5f} copy time:{copy_to_cpu_time:.5f} "
                iteration_out += f"(wasted:{time_pct:.1f}%) errors:{errors}"
                terminal_logger.debug(iteration_out)
            batch_id += 1
        setup_iteration += 1

    if generate is True:
        torch.save(golden, gold_path)
        check_dnn_accuracy(predicted=golden, ground_truth=input_labels, output_logger=terminal_logger,
                           dnn_goal=dnn_goal)

    if terminal_logger:
        terminal_logger.debug("Finish computation.")

    dnn_log_helper.end_log_file()


if __name__ == '__main__':
    try:
        main()
    except Exception as main_function_exception:
        dnn_log_helper.log_and_crash(fatal_string=f"EXCEPTION:{main_function_exception}")
