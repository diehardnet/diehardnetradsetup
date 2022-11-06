#!/usr/bin/python3
import argparse
import copy
import datetime
import re
from typing import List, Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    """ Parse the args and return an args namespace and the tostring from the args    """
    parser = argparse.ArgumentParser(description='PyTorch DNN radiation parser', add_help=False)
    # parser = argparse.ArgumentParser(description='PyTorch DNN radiation setup')
    parser.add_argument('--radlogdir', help="Path to the directory that contains the logs", required=True)
    parser.add_argument('--nvbitfilogdir', help="Path to the directory that contains the logs", required=True)

    args, remaining_argv = parser.parse_known_args()

    return args


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


def main():
    args = parse_args()
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

    for dataset in ["c10"]:  # ,  "c100"]:
        for diehardnet_version in diehardnet_configs:
            config = f"{dataset}{diehardnet_version}"
            for fi_model_str, fi_model in fi_models.items():
                for it in range(1, faults_per_fm + 1):
                    log_file = get_log_file_name(fi_dir=args.nvbitfilogdir, config=config, fault_it=it,
                                                 fi_model=fi_model, group=fp32)
                    full_log_file = f"{args.radlogdir}/{log_file}"
                    new_line = parse_log_file(log_path=full_log_file, fi_model=fi_model_str, group=fp32_str)
                    if new_line:
                        data_list.extend(new_line)

    df = pd.DataFrame(data_list)
    df = df.fillna(0)
    print(df["config"].value_counts())

    df.to_csv("parsed_logs_fi.csv", index=False)


if __name__ == '__main__':
    main()
