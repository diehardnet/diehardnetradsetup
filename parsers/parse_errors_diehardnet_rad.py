#!/usr/bin/python3
import argparse
import copy
import datetime
import os
import re
from typing import List

import pandas as pd


def parse_args() -> argparse.Namespace:
    """ Parse the args and return an args namespace and the tostring from the args    """
    parser = argparse.ArgumentParser(description='PyTorch DNN radiation parser', add_help=False)
    # parser = argparse.ArgumentParser(description='PyTorch DNN radiation setup')
    parser.add_argument('--logdir', help="Path to the directory that contains the logs", required=True)

    args, remaining_argv = parser.parse_known_args()

    return args


def parse_log_file(log_path: str) -> List[dict]:
    # ...log/2022_09_15_16_00_43_PyTorch-c100_res44_test_02_relu6-bn_200_epochs_ECC_OFF_carolinria.log
    pattern = r".*/(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_\S+_ECC_(\S+)_(\S+).log"
    m = re.match(pattern, log_path)
    if m:
        year, month, day, hour, minute, seconds, ecc, hostname = m.groups()
        year, month, day, hour, minute, seconds = [int(i) for i in [year, month, day, hour, minute, seconds]]
        start_dt = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=seconds)
        data_list = list()
        critical_sdc = False
        with open(log_path) as log_fp:
            header = log_fp.readline()
            h_m = re.match(r"#SERVER_HEADER.*--config.*/(\S+).yaml .*", header)
            data_dict = dict(start_dt=start_dt, config=h_m.group(1), ecc=ecc, hostname=hostname,
                             logfile=os.path.basename(log_path))
            last_acc_time = 0
            for line in log_fp:
                if "critical-img" in line:
                    critical_sdc = True
                sdc_m = re.match(r"#SDC Ite:(\d+) KerTime:(\S+) AccTime:(\S+) KerErr:(\d+) AccErr:(\d+)", line)
                if sdc_m:
                    it, ker_time, acc_time, ker_err, acc_err = sdc_m.groups()
                    last_acc_time = float(acc_time)
                    curr_data = copy.deepcopy(data_dict)
                    curr_data.update(dict(it=it, ker_time=float(ker_time), acc_time=0, ker_err=ker_err,
                                          acc_err=acc_err, sdc=1, critical_sdc=0, hostname=hostname))
                    if critical_sdc:
                        curr_data["critical_sdc"] = 1
                        critical_sdc = False
                    data_list.append(curr_data)
            if data_list:
                data_list[-1]["acc_time"] = last_acc_time
        return data_list


def main():
    args = parse_args()
    data_list = list()
    for subdir, dirs, files in os.walk(args.logdir):
        if any([i in subdir for i in ["carolp"]]):
            for file in files:
                path = os.path.join(subdir, file)
                new_line = parse_log_file(log_path=path)
                if new_line:
                    data_list.extend(new_line)

    df = pd.DataFrame(data_list)
    df = df.fillna(0)
    df.to_csv("../data/parsed_logs.csv", index=False)


if __name__ == '__main__':
    main()
