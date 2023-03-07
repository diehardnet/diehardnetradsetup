#!/usr/bin/python3
import os.path
import re

import pandas as pd

from common import parse_log_file


def get_log_file_name(fi_dir, config, fault_it, fi_model, group):
    config_stdout = f"{fi_dir}/{config}/{config}-group{group}-model{fi_model}-icount{fault_it}/stdout.txt"
    with open(config_stdout) as fp:
        for line in fp:
            m = re.match(r"Log file path /var/radiation-benchmarks/log/(\S+) - FILE:.*", line)
            if m:
                return m.group(1)
    raise ValueError(config_stdout)


def execute_cmd(cmd):
    print("EXECUTING:", cmd)
    if os.system(cmd) != 0:
        raise ValueError(f"Could not execute {cmd}")


def main():
    data_list = list()
    fi_models = dict(FLIP_SINGLE_BIT=0, RANDOM_VALUE=2, ZERO_VALUE=3, WARP_RANDOM_VALUE=4, WARP_ZERO_VALUE=5)
    diehardnet_configs = [
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
    faults_per_fm = 400
    fp32, fp32_str = 1, "fp32"
    data_location = "/home/fernando/Dropbox/temp/nvbitfi_ml_data/data_dsn2023"
    tar_files_cifar = {
        "logs_400_injections_c10_dhn_0511": ["c10"],
        "logs_400_injections_c100_dhn_1611": ["c100"],
        "logs_400_injections_c100_dhn_1711": ["c100"],
    }
    tar_files_imagenet = {
        "logs_400_injections_imagenet_dhn_2711": ["imagenet"],
        "logs_400_injections_imagenet_dhn_2711_titan_inria": ["imagenet"]
    }
    rad_log_dir_base = "var/radiation-benchmarks/log"
    nvbit_fi_log_dir_base = "logs"
    tmp_path = "/tmp/diehardnet"

    if os.path.isdir(tmp_path) is False:
        os.mkdir(tmp_path)

    execute_cmd(f"rm -rf {tmp_path}/*")

    for tar_file, datasets in tar_files_cifar.items():
        tar_full_path = f"{data_location}/{tar_file}.tar.gz"
        new_path = f"{tmp_path}/{tar_file}"
        if os.path.isdir(new_path) is False:
            os.mkdir(new_path)
        execute_cmd(f"rm -rf {new_path}/*")

        tar_cmd = f"tar xzf {tar_full_path} -C {new_path}"
        execute_cmd(tar_cmd)
        for dataset in datasets:
            for diehardnet_version in diehardnet_configs:
                config = f"{dataset}{diehardnet_version}"
                for fi_model_str, fi_model in fi_models.items():
                    for it in range(1, faults_per_fm + 1):
                        rad_log_dir = f"{new_path}/{rad_log_dir_base}"
                        nvbit_fi_log_dir = f"{new_path}/{nvbit_fi_log_dir_base}"
                        log_file = get_log_file_name(fi_dir=nvbit_fi_log_dir, config=config, fault_it=it,
                                                     fi_model=fi_model, group=fp32)
                        full_log_file = f"{rad_log_dir}/{log_file}"
                        new_line, _ = parse_log_file(log_path=full_log_file)
                        if len(new_line) == 1:
                            new_line = new_line[0]
                            new_line["fi_model"], new_line["group"] = fi_model, fp32_str
                            data_list.append(new_line)
                        elif len(new_line) > 1:
                            raise ValueError("Incorrect size of new line")

    for tar_file, datasets in tar_files_imagenet.items():
        tar_full_path = f"{data_location}/{tar_file}.tar.gz"
        new_path = f"{tmp_path}/{tar_file}"
        if os.path.isdir(new_path) is False:
            os.mkdir(new_path)
        execute_cmd(f"rm -rf {new_path}/*")

        tar_cmd = f"tar xzf {tar_full_path} -C {new_path}"
        execute_cmd(tar_cmd)
        config = f"imagenet1k_v2_base"
        for fi_model_str, fi_model in fi_models.items():
            for it in range(1, faults_per_fm + 1):
                rad_log_dir = f"{new_path}/{rad_log_dir_base}"
                nvbit_fi_log_dir = f"{new_path}/{nvbit_fi_log_dir_base}"
                log_file = get_log_file_name(fi_dir=nvbit_fi_log_dir, config=config, fault_it=it,
                                             fi_model=fi_model, group=fp32)
                full_log_file = f"{rad_log_dir}/{log_file}"
                new_line, _ = parse_log_file(log_path=full_log_file)
                if len(new_line) == 1:
                    new_line = new_line[0]
                    new_line["fi_model"], new_line["group"] = fi_model, fp32_str
                    data_list.append(new_line)
                elif len(new_line) > 1:
                    raise ValueError("Incorrect size of new line")

    df = pd.DataFrame(data_list)
    df = df.fillna(0)
    print(df["config"].value_counts())
    df.to_csv(f"../data/parsed_logs_fi.csv", index=False)


if __name__ == '__main__':
    main()
