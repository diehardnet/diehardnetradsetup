#!/usr/bin/python3
import argparse
import os.path
import re
import struct

import math
import pandas as pd

from common import parse_log_file

PARSER = argparse.ArgumentParser(prog='DieHardFIParser', description='Parse Diehardnet Fi Logs')
PARSER.add_argument("--uncompress", action="store_true", default=False, help="To uncompress files to /tmp")
ARGS = PARSER.parse_args()

UNCOMPRESS_FILES = bool(ARGS.uncompress)


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


def is_float32_nan(hex_str: str) -> bool:
    if len(hex_str) != 8:
        return False

    try:
        # Convert the hex string to a 32-bit unsigned integer
        hex_int = int(hex_str, 16)

        # Extract the sign (1 bit), exponent (8 bits), and fraction (23 bits)
        sign = (hex_int >> 31) & 0x1
        exponent = (hex_int >> 23) & 0xFF
        fraction = hex_int & 0x7FFFFF

        # In IEEE 754, NaN is represented by an exponent of all 1s (255) and a non-zero fraction
        return exponent == 0xFF and fraction != 0
    except ValueError:
        return False


def hex_to_float(hex_str):
    hex_no_x = hex_str.replace("0x", "").upper().ljust(8, '0')
    fp_val = struct.unpack('!f', bytes.fromhex(hex_no_x))[0]
    assert is_float32_nan(hex_str=hex_no_x) == math.isnan(fp_val), f"NaN disagreement {fp_val} and {hex_no_x}"
    return fp_val


def parse_nvbitfi_log_file(config_name, group, error_model, base_dir, injection_count):
    base_path = f"{base_dir}/logs/{config_name}"
    full_path = f"{base_path}/{config_name}-group{group}-model{error_model}-icount{injection_count}"
    inj_nvbit_log = f"{full_path}/nvbitfi-injection-log-temp.txt"
    return_data = list()
    any_nan, any_inf = False, False
    with open(inj_nvbit_log) as fp:
        before_val, after_val, opcode = None, None, None
        file_lines = fp.readlines()
        assert len(file_lines) > 0, (f"Incorrect number of lines {config_name}-"
                                     f"group{group}-model{error_model}-icount{injection_count}")

        for line in file_lines:
            val_m = re.match(r"beforeVal: (0[xX][0-9a-fA-F]+);afterVal: (0[xX][0-9a-fA-F]+)", line)
            if val_m:
                assert (before_val is None) and (after_val is None), "PAU values are supposed to be none"
                before_val = hex_to_float(hex_str=val_m.group(1))
                assert math.isnan(before_val) is False, f"PAU-NAN {line}"
                assert math.isinf(before_val) is False, f"PAU-INF {line}"

                after_val = hex_to_float(hex_str=val_m.group(2))

                # opcode_m = re.match(r"opcode: (\S+)", line)
                # if opcode_m:
                # assert opcode is None, "PAU opcode is supposed to be none"
                # opcode = opcode_m.group(1)
                # if np.isinf(before_val) or np.isnan(before_val): # and opcode in FP_INSTRUCTIONS:
                #     raise ValueError(f"Pau no numero {before_val} {opcode} {inj_nvbit_log}")
                if math.isnan(after_val):
                    any_nan = True
                if math.isinf(after_val):
                    any_inf = True
                return_data.append(dict(before_val=before_val, after_val=after_val,
                                        is_nan=math.isnan(after_val), is_inf=math.isinf(after_val),
                                        abs_diff=abs(after_val - before_val), opcode=opcode))
                before_val, after_val, opcode = None, None, None

    return_infos = {"nan": any_nan, "inf": any_inf}
    return return_infos


def main():
    data_list = list()
    nan_data_list = list()
    fi_models = dict(FLIP_SINGLE_BIT=0, RANDOM_VALUE=2, ZERO_VALUE=3, WARP_RANDOM_VALUE=4, WARP_ZERO_VALUE=5)
    faults_per_fm = 400
    fp32, fp32_str = 1, "fp32"
    data_location = "/home/fernando/Dropbox/DieHardNet/tetc_nvbitfi_logs"
    cifar_defaults_configs = [
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
    c10_configs = [f"c10{diehardnet_version}" for diehardnet_version in cifar_defaults_configs]
    c100_configs = [f"c100{diehardnet_version}" for diehardnet_version in cifar_defaults_configs]

    tar_files_cifar = {
        "logs_400_injections_c10_dhn_0511": c10_configs,
        "logs_400_injections_c100_dhn_1611": c100_configs,
        "logs_400_injections_c100_dhn_1711": c100_configs,
    }
    tar_files_tinyimagenet = {
        # "logs_400_injections_tinyimagenet_base_dhn_1907": ["diehardnet_tiny_res56_bn-relu_base"],
        "logs_400_injections_tinyimagenet_base_dhn_2907": ["diehardnet_tiny_res56_bn-relu_base"],
        "logs_400_injections_tinyimagenet_aware_dhn_2707": ["diehardnet_tiny_res56_relu6-bn_aware"],
        "logs_400_injections_tinyimagenet_aware_nans_dhn_2707": ["diehardnet_tiny_res56_relu6-bn_aware_nans"]
    }
    rad_log_dir_base = "var/radiation-benchmarks/log"
    nvbit_fi_log_dir_base = "logs"
    tmp_path = "/tmp/diehardnet"

    if os.path.isdir(tmp_path) is False:
        os.mkdir(tmp_path)

    untar_files = {**tar_files_cifar, **tar_files_tinyimagenet}

    if UNCOMPRESS_FILES:
        execute_cmd(f"rm -rf {tmp_path}/*")
        for tar_file, datasets in untar_files.items():
            new_path = f"{tmp_path}/{tar_file}"

            # uncompress only if necessary
            if os.path.isdir(new_path) is False:
                os.mkdir(new_path)
            execute_cmd(f"rm -rf {new_path}/*")
            tar_full_path = f"{data_location}/{tar_file}.tar.gz"
            tar_cmd = f"tar xzf {tar_full_path} -C {new_path}"
            execute_cmd(tar_cmd)

    for tar_file, configs in untar_files.items():
        new_path = f"{tmp_path}/{tar_file}"
        for config in configs:
            print(new_path)
            for fi_model_str, fi_model in fi_models.items():
                for it in range(1, faults_per_fm + 1):
                    rad_log_dir = f"{new_path}/{rad_log_dir_base}"
                    nvbit_fi_log_dir = f"{new_path}/{nvbit_fi_log_dir_base}"
                    log_file = get_log_file_name(fi_dir=nvbit_fi_log_dir, config=config, fault_it=it,
                                                 fi_model=fi_model, group=fp32)
                    full_log_file = f"{rad_log_dir}/{log_file}"
                    new_line = parse_log_file(log_path=full_log_file)
                    return_info_nans = parse_nvbitfi_log_file(config_name=config, group=fp32,
                                                              error_model=fi_model, base_dir=new_path,
                                                              injection_count=it)
                    nan_data_list.append(dict(
                        config=config, fault_it=it, fi_model=fi_model, group=fp32, **return_info_nans,
                    ))
                    if len(new_line) == 1:
                        new_line = new_line[0]
                        new_line["fi_model"], new_line["group"] = fi_model, fp32_str
                        data_list.append(new_line)
                        nan_data_list[-1]["sdc"] = new_line["sdc"]
                        nan_data_list[-1]["critical_sdc"] = new_line["critical_sdc"]
                        nan_data_list[-1]["there_is_nan_on_output"] = new_line["there_is_nan_on_output"]
                        nan_data_list[-1]["there_is_inf_on_output"] = new_line["there_is_inf_on_output"]
                    elif len(new_line) > 1:
                        print(fi_model, fp32_str)
                        print(new_line)
                        raise ValueError("Incorrect size of new line")

    df = pd.DataFrame(data_list)
    df = df.fillna(0)
    print(df["config"].value_counts())
    df.to_csv(f"../data/parsed_logs_fi.csv", index=False)

    additional_data_df = pd.DataFrame(nan_data_list)
    additional_data_df = additional_data_df.fillna(0)
    additional_data_df.to_csv("../data/additional_data_fi.csv", index=False)


if __name__ == '__main__':
    main()
