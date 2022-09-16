#!/usr/bin/python3
import argparse
import datetime
import os
import re

import pandas as pd
from typing import Union


def parse_args() -> argparse.Namespace:
    """ Parse the args and return an args namespace and the tostring from the args    """
    parser = argparse.ArgumentParser(description='PyTorch DNN radiation parser', add_help=False)
    # parser = argparse.ArgumentParser(description='PyTorch DNN radiation setup')
    parser.add_argument('--logdir', help="Path to the directory that contains the logs")

    args, remaining_argv = parser.parse_known_args()

    return args


def parse_log_file(log_path: str) -> Union[dict, None]:
    # ...log/2022_09_15_16_00_43_PyTorch-c100_res44_test_02_relu6-bn_200_epochs_ECC_OFF_carolinria.log
    pattern = r".*log/(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_PyTorch-(\S+)_ECC_(\S+)_(\S+).log"
    m = re.match(pattern, log_path)
    if m:
        year, month, day, hour, minute, seconds, config, ecc, hostname = m.groups()
        year, month, day, hour, minute, seconds = [int(i) for i in [year, month, day, hour, minute, seconds]]
        start_dt = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=seconds)
        data_dict = dict(start_dt=start_dt, config=config, ecc=ecc, hostname=hostname)
        with open(log_path) as log_fp:
            for line in log_fp:
                if "SDC" in line or "ERR" in line:
                    data_dict["sdc"] = 1
                    sdc_m = re.match(r"#SDC Ite:(\d+) KerTime:(\S+) AccTime:(\S+) KerErr:(\d+) AccErr:(\d+)", line)
                    if sdc_m:
                        it, ker_time, acc_time, ker_err, acc_err = sdc_m.groups()
                        data_dict.update(
                            dict(it=it, ker_time=float(ker_time), acc_time=float(acc_time), ker_err=ker_err,
                                 acc_err=acc_err))
                if "critical-img" in line:
                    data_dict["critical_sdc"] = 1

        return data_dict


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

    print(df[(df["sdc"] != 0) | (df["critical_sdc"] != 0)])
    print(df[(df["critical_sdc"] == 1) & (df["sdc"] == 0)])
    print(df[df["critical_sdc"] == 1])


if __name__ == '__main__':
    main()
