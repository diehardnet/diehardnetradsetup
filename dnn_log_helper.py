"""
This wrapper is only if you don't want to use libLogHelper
"""
import inspect
import sys
import traceback

from libLogHelper.build import log_helper

import configs

__LOGGING_ACTIVE = False


def start_setup_log_file(framework_name: str, framework_version: str, args_conf: list, dnn_name: str,
                         activate_logging: bool, dnn_goal: str) -> None:
    global __LOGGING_ACTIVE
    __LOGGING_ACTIVE = activate_logging
    dnn_log_header = f"framework:{framework_name} framework_version:{framework_version} dnn_goal:{dnn_goal} "
    dnn_log_header += f"topk:{configs.CLASSIFICATION_CRITICAL_TOP_K} " + " ".join(args_conf)
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


def __getattr__(name: str) -> str:
    if __LOGGING_ACTIVE:
        if name == 'log_file_name':
            return log_helper.get_log_file_name()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def log_and_crash(fatal_string: str) -> None:
    caller_frame_record = inspect.stack()[1]  # 0 represents this line
    # 1 represents line at caller
    frame = caller_frame_record[0]
    info = inspect.getframeinfo(frame)
    fatal_log_string = f"SETUP_ERROR:{fatal_string} FILE:{info.filename}:{info.lineno} F:{info.function}"
    fatal_log_string += f"\nTRACEBACK:{traceback.format_exc()}"
    # It is better to always show the exception to stdout
    print(fatal_log_string)
    # Also save to tmp in the case of logging is not active
    with open(configs.TMP_CRASH_FILE, "w") as tmp_fp:
        tmp_fp.write(f"{fatal_log_string}\n")
    log_info_detail(info_detail=fatal_log_string)
    end_log_file()
    sys.exit(1)
