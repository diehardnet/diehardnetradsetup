# Error threshold for the test
CLASSIFICATION_ABS_THRESHOLD = 0
DETECTION_BOXES_ABS_THRESHOLD = 1e-5
DETECTION_SCORES_ABS_THRESHOLD = 1e-5

MAXIMUM_ERRORS_PER_ITERATION = 4096
MAXIMUM_INFOS_PER_ITERATION = 256
ITERATION_INTERVAL_LOG_HELPER_PRINT = 30

# Device capability for pytorch
MINIMUM_DEVICE_CAPABILITY = 5  # Maxwell

CLASSIFICATION_CRITICAL_TOP_K = 1
# FORCE the gpu to be present
DEVICE = "cuda:0"

DIEHARDNET_CONFIGS = [
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
    "_res44_test_02_gelu6_nans",
]

ALL_DNNS = [f"{dataset}{config}.yaml" for dataset in ["c10", "c100"] for config in DIEHARDNET_CONFIGS]


CIFAR10 = "cifar10"
CIFAR100 = "cifar100"

CLASSES = {
    CIFAR10: 10,
    CIFAR100: 100
}
