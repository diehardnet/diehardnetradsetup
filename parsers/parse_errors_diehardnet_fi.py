#!/usr/bin/python3
import copy
import os.path
import re
from typing import List

import pandas as pd


def parse_log_file(log_path: str, fi_model: str, group: str) -> List[dict]:
    # ...log/2022_09_15_16_00_43_PyTorch-c100_res44_test_02_relu6-bn_200_epochs_ECC_OFF_carolinria.log
    pattern = r".*/(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_\S+_ECC_(\S+)_(\S+).log"
    m = re.match(pattern, log_path)
    if m:
        year, month, day, hour, minute, seconds, ecc, hostname = m.groups()
        # year, month, day, hour, minute, seconds = [int(i) for i in [year, month, day, hour, minute, seconds]]
        # start_dt = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=seconds)
        data_list = list()
        critical_sdc = False
        with open(log_path) as log_fp:
            header = log_fp.readline()
            h_m = re.match(r"#HEADER.*config=.*/(\S+).yaml.*", header)
            config = h_m.group(1)
            data_dict = dict(config=config, ecc=ecc, hostname=hostname, fi_model=fi_model, group=group)
            has_end = False
            has_sdc = False
            for line in log_fp:
                has_end = ("#END" in line) or has_end
                if "critical-img" in line:
                    critical_sdc = True
                sdc_m = re.match(r"#SDC Ite:(\d+) KerTime:(\S+) AccTime:(\S+) KerErr:(\d+) AccErr:(\d+)", line)
                if sdc_m:
                    has_sdc = True
                    it, ker_time, acc_time, ker_err, acc_err = sdc_m.groups()
                    curr_data = copy.deepcopy(data_dict)
                    curr_data.update(dict(it=it, ker_time=float(ker_time), acc_time=float(acc_time), ker_err=ker_err,
                                          acc_err=acc_err, sdc=1, critical_sdc=0, hostname=hostname))
                    if critical_sdc:
                        curr_data["critical_sdc"] = 1
                        critical_sdc = False
                    data_list.append(curr_data)
            if has_sdc is False:
                data_list.append(data_dict)

            for data_list_i in data_list:
                data_list_i["has_end"] = int(has_end)

        return data_list


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
    tar_files = {
        "logs_400_injections_c10_dhn_0511": ["c10"],
        "logs_400_injections_c100_dhn_1611": ["c100"],
        "logs_400_injections_c100_dhn_1711": ["c100"]
    }

    rad_log_dir_base = "var/radiation-benchmarks/log"
    nvbit_fi_log_dir_base = "logs"
    tmp_path = "/tmp/diehardnet"

    if os.path.isdir(tmp_path) is False:
        os.mkdir(tmp_path)

    execute_cmd(f"rm -rf {tmp_path}/*")

    for tar_file in tar_files:
        tar_full_path = f"{data_location}/{tar_file}.tar.gz"
        new_path = f"{tmp_path}/{tar_file}"
        if os.path.isdir(new_path) is False:
            os.mkdir(new_path)
        execute_cmd(f"rm -rf {new_path}/*")

        tar_cmd = f"tar xzf {tar_full_path} -C {new_path}"
        execute_cmd(tar_cmd)
        datasets = tar_files[tar_file]
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
                        new_line = parse_log_file(log_path=full_log_file, fi_model=fi_model_str, group=fp32_str)
                        if new_line:
                            data_list.extend(new_line)

    df = pd.DataFrame(data_list)
    df = df.fillna(0)
    print(df["config"].value_counts())
    df.to_csv(f"../data/parsed_logs_fi.csv", index=False)


if __name__ == '__main__':
    main()
