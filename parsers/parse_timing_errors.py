#!/usr/bin/python3
import copy
import datetime
import os
import re

import pandas as pd


def is_error_valid(row, cross_section_dict):
    # load the Excel file
    cross_section = cross_section_dict[row["hostname"]]
    return int(cross_section[(cross_section["start_dt"] <= row["start_dt"]) &
                             (cross_section["end_dt"] >= row["start_dt"])].shape[0] != 0)


def parse_log_file(log_path):
    # ...log/2022_09_15_16_00_43_PyTorch-c100_res44_test_02_relu6-bn_200_epochs_ECC_OFF_carolinria.log
    pattern = r".*/(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_\S+_ECC_(\S+)_(\S+).log"
    m = re.match(pattern, log_path)
    if m:
        year, month, day, hour, minute, seconds, ecc, hostname = m.groups()
        year, month, day, hour, minute, seconds = [int(i) for i in [year, month, day, hour, minute, seconds]]
        start_dt = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=seconds)
        data_list = list()
        setup_crash_motiv, server_due_motiv, server_end_motiv = None, None, None

        with open(log_path) as log_fp:
            header = log_fp.readline()
            h_m = re.match(r"#SERVER_HEADER.*--config.*/(\S+).yaml .*", header)
            config = h_m.group(1)
            data_dict = dict(start_dt=start_dt, config=config, ecc=ecc, hostname=hostname,
                             logfile=os.path.basename(log_path))
            last_acc_time = 0
            critical_sdc, evil_sdc, benign_sdc = 0, 0, 0
            for line in log_fp:
                # Crash info
                setup_crash = re.match(r".*SETUP_ERROR:(.*)", line)
                if setup_crash:
                    setup_crash_motiv = setup_crash.group(1)
                server_due = re.match("#SERVER_DUE:(.*)", line)
                if server_due:
                    server_due_motiv = server_due.group(1)
                server_end = re.match("#SERVER_END(.*)", line)
                if server_end:
                    server_end_motiv = server_end.group(1)

                # SDC info
                ct_m = re.match(r"#ERR batch:\d+ critical-img:\d+ i:\d+ g:(\d+) o:(\d+) gt:(\d+)", line)
                if ct_m:
                    critical_sdc += 1
                    golden, output, ground_truth = ct_m.group(1), ct_m.group(2), ct_m.group(3)
                    evil_sdc += int(output != ground_truth)
                    benign_sdc += int(output == ground_truth and golden != ground_truth)
                elif "critical-img" in line:
                    raise ValueError(f"Not a valid line {line}")

                sdc_m = re.match(r"#SDC Ite:(\d+) KerTime:(\S+) AccTime:(\S+) KerErr:(\d+) AccErr:(\d+)", line)
                if sdc_m:
                    it, ker_time, acc_time, ker_err, acc_err = sdc_m.groups()
                    last_acc_time = float(acc_time)
                    curr_data = copy.deepcopy(data_dict)
                    curr_data.update(
                        dict(it=it, ker_time=float(ker_time), acc_time=0, ker_err=ker_err, acc_err=acc_err, sdc=1,
                             critical_sdc=int(critical_sdc != 0), evil_sdc=evil_sdc, benign_sdc=benign_sdc)
                    )
                    data_list.append(curr_data)
                    critical_sdc, evil_sdc, benign_sdc = 0, 0, 0

                it_m = re.match(r"#IT (\d+) KerTime:(\S+) AccTime:(\S+)", line)
                if it_m:
                    it, kernel_time, acc_time = it_m.groups()
                    last_acc_time = float(acc_time)
                    curr_data = copy.deepcopy(data_dict)
                    curr_data.update(
                        dict(it=it, ker_time=float(kernel_time), acc_time=0, sdc=0, critical_sdc=0, evil_sdc=0,
                             benign_sdc=0)
                    )
                    data_list.append(curr_data)
            #
            # if data_list:
            #     for nl in data_list:
            #         nl["setup_crash"], nl["server_due"] =
            #     data_list[-1]["acc_time"] = last_acc_time
        data_dict.update(dict(setup_crash_motiv=setup_crash_motiv, server_due_motiv=server_due_motiv,
                              server_end_motiv=server_end_motiv))
        return data_list, data_dict


def extract_errors_data(log_dir):
    data_list = list()
    due_list = list()
    for subdir, dirs, files in os.walk(log_dir):
        if any([i in subdir for i in ["carolp", "carolm", "carola"]]):
            print("Parsing", subdir)
            for file in files:
                path = os.path.join(subdir, file)
                new_line, due_data = parse_log_file(log_path=path)
                due_list.append(due_data)
                if new_line:
                    data_list.extend(new_line)
    return pd.DataFrame(data_list), pd.DataFrame(due_list)


def main():
    unique_hostnames = {"carola20001": None, "carolp20002": None, "carolm20004": None, "carolp20003": None}
    for hostname in unique_hostnames:
        df_hostname = pd.read_excel("/home/fernando/git_research/ChipIR_December_2022/data/cross_section.xlsx",
                                    sheet_name=hostname)
        df_hostname["start_dt"] = pd.to_datetime(df_hostname["start_dt"])
        df_hostname["end_dt"] = pd.to_datetime(df_hostname["end_dt"])

        unique_hostnames[hostname] = df_hostname

    df_iterations, df_dues = extract_errors_data(log_dir="/home/fernando/temp/ChipIR2022_December/logs")
    print("Parsing the good errors")
    df_iterations["start_dt"] = pd.to_datetime(df_iterations["start_dt"])
    df_dues["start_dt"] = pd.to_datetime(df_dues["start_dt"])

    df_iterations["is_error_valid"] = df_iterations.apply(is_error_valid, axis="columns", args=(unique_hostnames,))
    df_dues["is_error_valid"] = df_dues.apply(is_error_valid, axis="columns", args=(unique_hostnames,))

    df_iterations = df_iterations[df_iterations["is_error_valid"] == 1]
    df_dues = df_dues[df_dues["is_error_valid"] == 1]
    print(df_iterations)
    print(df_dues)
    df_iterations.to_csv("../data/parsed_logs_rad_valid_ones_timing_data.csv", index=False)
    df_dues.to_csv("../data/parsed_logs_rad_valid_dues.csv", index=False)


if __name__ == '__main__':
    main()
