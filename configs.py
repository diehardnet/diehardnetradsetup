# Error threshold for the test
CLASSIFICATION_ABS_THRESHOLD = 1e-3
SEGMENTATION_ABS_THRESHOLD = 1e-3
DETECTION_BOXES_ABS_THRESHOLD = 1e-5
DETECTION_SCORES_ABS_THRESHOLD = 1e-5

RANDOM_INT_LIMIT = 65535

MAXIMUM_ERRORS_PER_ITERATION = 4096
MAXIMUM_INFOS_PER_ITERATION = 256

# Device capability for pytorch
MINIMUM_DEVICE_CAPABILITY = 5  # Maxwell

CLASSIFICATION_CRITICAL_TOP_K = 1
# FORCE the gpu to be present
DEVICE = "cuda:0"

DIEHARDNET_CLASSIFICATION_CONFIGS = (
        [
            f"{dataset}{config}" for dataset in ["c10", "c100"] for config in [
            # Baseline
            "_res44_test_01_bn-relu_base",
            # Baseline + Relu6
            "_res44_test_02_bn-relu6",
            # Baseline + Relu6 + Order Inversion
            "_res44_test_02_bn-relu6_base",
            # Order inversion with relu6
            "_res44_test_02_relu6-bn",
            # Order inversion + nan filter + Relu6
            "_res44_test_02_relu6-bn_nanfilter",
            # Gelu and nan C100
            "_res44_test_02_gelu6_nans"]
        ]
        +  # Tiny imagenet
        [
            "tiny_res56_bn-relu_base",
            "tiny_res56_relu6-bn_aware",
            "tiny_res56_relu6-bn_aware_nans",
        ]
)

RESNET50_IMAGENET_1K_V2_BASE = "resnet50_imagenet1k_v2_base"
DIEHARDNET_TRANS_LEARNING_CONFIGS = [
    # Baseline
    RESNET50_IMAGENET_1K_V2_BASE,

]

# Segmentation DNNs
DEEPLABV3_RESNET50 = "deeplabv3_resnet50_base"
DEEPLABV3_RESNET101 = "deeplabv3_resnet101_base"
FCN_RESNET50 = "fcn_resnet50_base"
FCN_RESNET101 = "fcn_resnet101_base"

DIEHARDNET_SEGMENTATION_CONFIGS = [
    # Baselines
    DEEPLABV3_RESNET50,
    DEEPLABV3_RESNET101,
    # FCN_RESNET50,
    # FCN_RESNET101
]

# Classification ViTs
VITS_BASE_PATCH16_224 = "vit_base_patch16_224"
VITS_BASE_PATCH16_384 = "vit_base_patch32_384"
VITS_BASE_PATCH32_224_SAM = "vit_base_patch32_224_sam"
VITS_BASE_RESNET50_384 = "vit_base_resnet50_384"
DIEHARDNET_VITS_CONFIGS = [
    VITS_BASE_PATCH16_224,
    VITS_BASE_PATCH16_384,
    VITS_BASE_PATCH32_224_SAM,
    VITS_BASE_RESNET50_384
]

# Set the supported goals
CLASSIFICATION = "classify"
SEGMENTATION = "segmentation"

DNN_GOAL = {
    # Classification nets
    **{k: CLASSIFICATION for k in DIEHARDNET_CLASSIFICATION_CONFIGS + DIEHARDNET_TRANS_LEARNING_CONFIGS},
    **{k: CLASSIFICATION for k in DIEHARDNET_VITS_CONFIGS},
    # Segmentation nets
    **{k: SEGMENTATION for k in DIEHARDNET_SEGMENTATION_CONFIGS},
}

ITERATION_INTERVAL_LOG_HELPER_PRINT = {
    # Classification nets, cifar, very small
    **{k: 30 for k in DIEHARDNET_CLASSIFICATION_CONFIGS},
    # imagenet not so small
    **{k: 10 for k in DIEHARDNET_TRANS_LEARNING_CONFIGS},
    **{k: 1 for k in DIEHARDNET_VITS_CONFIGS},
    # Segmentation nets, huge
    **{k: 1 for k in DIEHARDNET_SEGMENTATION_CONFIGS}
}

CIFAR10 = "cifar10"
CIFAR100 = "cifar100"
IMAGENET = "imagenet"
TINY_IMAGENET = "tinyimagenet"
COCO = "coco"

CLASSES = {
    CIFAR10: 10,
    CIFAR100: 100,
    IMAGENET: 1000,
    TINY_IMAGENET: 200
}

CIFAR_DATASET_DIR = "/home/carol/cifar"
IMAGENET_DATASET_DIR = "/home/carol/ILSVRC2012"
TINY_IMAGENET_DATASET_DIR = "/home/carol/TINY_ILSVRC2012"
COCO_DATASET_DIR = "/home/carol/coco"
COCO_DATASET_VAL = f"{COCO_DATASET_DIR}/val2017"
COCO_DATASET_ANNOTATIONS = f"{COCO_DATASET_DIR}/annotations/instances_val2017.json"

# File to save last status of the benchmark when log helper not active
TMP_CRASH_FILE = "/tmp/diehardnet_crash_file.txt"
