import copy
import datetime
import os
import re
from typing import List, Tuple


def parse_log_file(log_path: str) -> Tuple[List[dict], str]:
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
            h_m = re.match(r"#SERVER_HEADER.*--config.*/(\S+).yaml .*", header)
            config = h_m.group(1)
            data_dict = dict(start_dt=start_dt, config=config, ecc=ecc, hostname=hostname,
                             logfile=os.path.basename(log_path))
            last_acc_time = 0
            critical_sdc, evil_sdc, benign_sdc = 0, 0, 0
            for line in log_fp:
                ct_m = re.match(r"#ERR batch:\d+ critical-img:\d+ i:\d+ g:\d+ o:(\d+) gt:(\d+)", line)
                if ct_m:
                    critical_sdc += 1
                    is_evil = ct_m.group(1) != ct_m.group(2)
                    evil_sdc += int(is_evil)
                    benign_sdc += int(not is_evil)

                sdc_m = re.match(r"#SDC Ite:(\d+) KerTime:(\S+) AccTime:(\S+) KerErr:(\d+) AccErr:(\d+)", line)
                if sdc_m:
                    it, ker_time, acc_time, ker_err, acc_err = sdc_m.groups()
                    last_acc_time = float(acc_time)
                    curr_data = copy.deepcopy(data_dict)
                    curr_data.update(
                        dict(it=it, ker_time=float(ker_time), acc_time=0, ker_err=ker_err, acc_err=acc_err, sdc=1,
                             critical_sdc=critical_sdc, evil_sdc=evil_sdc, benign_sdc=benign_sdc, hostname=hostname)
                    )
                    data_list.append(curr_data)
                    critical_sdc, evil_sdc, benign_sdc = 0, 0, 0

            if data_list:
                data_list[-1]["acc_time"] = last_acc_time
        return data_list, config
