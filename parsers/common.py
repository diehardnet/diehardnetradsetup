import copy
import datetime
import os
import re
from typing import List

import yaml


def parse_log_file(log_path: str) -> List[dict]:
    # ...log/2022_09_15_16_00_43_PyTorch-c100_res44_test_02_relu6-bn_200_epochs_ECC_OFF_carolinria.log
    pattern = r".*/(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_\S+_ECC_(\S+)_(\S+).log"
    m = re.match(pattern, log_path)
    if m:
        year, month, day, hour, minute, seconds, ecc, hostname = m.groups()
        year, month, day, hour, minute, seconds = [int(i) for i in [year, month, day, hour, minute, seconds]]
        start_dt = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=seconds)
        data_list = list()
        with open(log_path) as log_fp:
            header = log_fp.readline()
            if "--config" in header:
                h_m = re.match(r"#SERVER_HEADER.*--config.*/(\S+).yaml .*", header)
            elif "config=" in header:
                h_m = re.match(r"#HEADER.*config=.*/(\S+).yaml .*", header)

            config = h_m.group(1)
            # Find the batch size
            with open(f"../configurations/{config}.yaml") as fp:
                batch_size = yaml.safe_load(fp)["batch_size"]
            data_dict = dict(start_dt=start_dt, config=config, ecc=ecc, hostname=hostname,
                             logfile=os.path.basename(log_path), batch_size=batch_size)
            last_acc_time = 0
            critical_sdc, evil_sdc, benign_sdc = 0, 0, 0
            for line in log_fp:
                ct_m = re.match(r"#ERR batch:\d+ critical-img:\d+ i:\d+ g:(\d+) o:(\d+) gt:(\d+)", line)
                if not ct_m:
                    ct_m = re.match("#ERR batch:\d+ critical-img:\d+ i:\d+ g:(\S+) o:(\S+)", line)
                if ct_m:
                    critical_sdc += 1
                    # golden, output, ground_truth = ct_m.group(1), ct_m.group(2), ct_m.group(3)
                    # evil_sdc += int(output != ground_truth)
                    # benign_sdc += int(output == ground_truth and golden != ground_truth)
                elif "critical-img" in line:
                    raise ValueError(f"Not a valid line {line}")

                sdc_m = re.match(r"#SDC Ite:(\d+) KerTime:(\S+) AccTime:(\S+) KerErr:(\d+) AccErr:(\d+)", line)
                if sdc_m:
                    it, ker_time, acc_time, ker_err, acc_err = sdc_m.groups()
                    last_acc_time = float(acc_time)
                    curr_data = copy.deepcopy(data_dict)
                    curr_data.update(
                        dict(it=it, ker_time=float(ker_time), acc_time=0, ker_err=ker_err, acc_err=acc_err, sdc=1,
                             critical_sdc=int(critical_sdc != 0),
                             # evil_sdc=evil_sdc, benign_sdc=benign_sdc,
                             hostname=hostname)
                    )
                    data_list.append(curr_data)
                    critical_sdc, evil_sdc, benign_sdc = 0, 0, 0

            if data_list:
                data_list[-1]["acc_time"] = last_acc_time
        return data_list
