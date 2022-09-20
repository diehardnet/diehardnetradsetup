"""
This wrapper is only if you don't want to use libLogHelper
"""
import configs
from libLogHelper.build import log_helper

__LOGGING_ACTIVE = False


def start_setup_log_file(framework_name: str, args_conf: list, dnn_name: str, activate_logging: bool) -> None:
    global __LOGGING_ACTIVE
    __LOGGING_ACTIVE = activate_logging
    dnn_log_header = f"framework:{framework_name} topk:{configs.CLASSIFICATION_CRITICAL_TOP_K} " + " ".join(args_conf)
    if __LOGGING_ACTIVE:
        bench_name = f"{framework_name}-{dnn_name}"
        log_helper.start_log_file(bench_name, dnn_log_header)
        log_helper.set_max_errors_iter(configs.MAXIMUM_ERRORS_PER_ITERATION)
        log_helper.set_max_infos_iter(configs.MAXIMUM_INFOS_PER_ITERATION)
        log_helper.set_iter_interval_print(configs.ITERATION_INTERVAL_LOG_HELPER_PRINT)


def start_iteration() -> None:
    if __LOGGING_ACTIVE:
        log_helper.start_iteration()


def end_iteration() -> None:
    if __LOGGING_ACTIVE:
        log_helper.end_iteration()


def end_log_file() -> None:
    if __LOGGING_ACTIVE:
        log_helper.end_log_file()


def log_info_detail(info_detail: str) -> None:
    if __LOGGING_ACTIVE:
        log_helper.log_info_detail(info_detail)


def log_error_detail(error_detail: str) -> None:
    if __LOGGING_ACTIVE:
        log_helper.log_error_detail(error_detail)


def log_error_count(error_count: int) -> None:
    if __LOGGING_ACTIVE:
        log_helper.log_error_count(error_count)
