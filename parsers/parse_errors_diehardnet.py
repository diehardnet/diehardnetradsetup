#!/usr/bin/python3
import argparse
import copy
import datetime
import os
import re

import pandas as pd
from typing import Union, List


def parse_args() -> argparse.Namespace:
    """ Parse the args and return an args namespace and the tostring from the args    """
    parser = argparse.ArgumentParser(description='PyTorch DNN radiation parser', add_help=False)
    # parser = argparse.ArgumentParser(description='PyTorch DNN radiation setup')
    parser.add_argument('--logdir', help="Path to the directory that contains the logs", required=True)

    args, remaining_argv = parser.parse_known_args()

    return args


def parse_log_file(log_path: str) -> List[dict]:
    # ...log/2022_09_15_16_00_43_PyTorch-c100_res44_test_02_relu6-bn_200_epochs_ECC_OFF_carolinria.log
    pattern = r".*/(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\S+)_ECC_(\S+)_(\S+).log"
    m = re.match(pattern, log_path)
    if m:
        year, month, day, hour, minute, seconds, config, ecc, hostname = m.groups()
        year, month, day, hour, minute, seconds = [int(i) for i in [year, month, day, hour, minute, seconds]]
        start_dt = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=seconds)
        data_dict = dict(start_dt=start_dt, config=config, ecc=ecc, hostname=hostname)
        data_list = list()
        critical_sdc = False
        with open(log_path) as log_fp:
            for line in log_fp:
                if "critical-img" in line:
                    critical_sdc = True
                sdc_m = re.match(r"#SDC Ite:(\d+) KerTime:(\S+) AccTime:(\S+) KerErr:(\d+) AccErr:(\d+)", line)
                if sdc_m:
                    it, ker_time, acc_time, ker_err, acc_err = sdc_m.groups()
                    curr_data = copy.deepcopy(data_dict)
                    curr_data.update(dict(it=it, ker_time=float(ker_time), acc_time=float(acc_time), ker_err=ker_err,
                                          acc_err=acc_err, sdc=1, critical_sdc=0))
                    if critical_sdc:
                        curr_data["critical_sdc"] = 1
                        critical_sdc = False

        return data_list


def main():
    args = parse_args()
    data_list = list()
    for subdir, dirs, files in os.walk(args.logdir):
        for file in files:
            path = os.path.join(subdir, file)
            new_line = parse_log_file(log_path=path)
            if new_line:
                data_list.append(new_line)

    df = pd.DataFrame(data_list)
    df = df.fillna(0)
    df.to_csv("parsed_logs.csv", index=False)


if __name__ == '__main__':
    main()
