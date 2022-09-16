#!/usr/bin/python3

import configparser
import glob
import json
import os.path
import time
from socket import gethostname
from pathlib import Path

CONFIG_FILE = "/etc/radiation-benchmarks.conf"
ITERATIONS = int(1e12)
TEST_SAMPLES = 128 * 50
DOWNLOAD_MODELS = False

DNN_MODELS = {
    # Diehardnet
    # Cifar 100
    "c100_res44_test_01_bn-relu_base.yaml",
    "c100_res44_test_02_bn-relu6.yaml",
    "c100_res44_test_02_relu6-bn.yaml",
    "c100_res44_test_02_bn-relu6_base.yaml",
    "c100_res44_test_02_gelu6_nans.yaml",
    # Cifar 10
    "c10_res44_test_02_bn-relu6_base.yaml",
    "c10_res44_test_01_bn-relu_base.yaml",
    "c10_res44_test_02_bn-relu6.yaml",
    "c10_res44_test_02_relu6-bn.yaml",
    "c10_res44_test_02_gelu6_nans.yaml",
    # Mobile net v2
    "cifar100_mobilenetv2_x1_4.yaml",
    "cifar10_mobilenetv2_x1_4.yaml"
}
TEST_JSONS = False


def main():
    try:
        config = configparser.RawConfigParser()
        config.read(CONFIG_FILE)
        server_ip = config.get('DEFAULT', 'serverip')
    except IOError as e:
        raise IOError("Configuration setup error: " + str(e))

    hostname = gethostname()
    home = str(Path.home())
    jsons_path = f"data/{hostname}_jsons"
    if os.path.isdir(jsons_path) is False:
        os.mkdir(jsons_path)

    if DOWNLOAD_MODELS:
        print("Download all the models")
        os.system("./download_models.py")

    current_directory = os.getcwd()
    script_name = "main.py"
    for dnn_model in DNN_MODELS:
        # Default filename will build the other names
        default_file_name = dnn_model.replace(".yaml", "")
        json_file_name = f"{jsons_path}/{default_file_name}.json"
        data_dir = f"{current_directory}/data"
        gold_path = f"{data_dir}/{default_file_name}.pt"
        checkpoint_dir = f"{data_dir}/checkpoints"
        config_path = f"{current_directory}/configurations/{dnn_model}"
        parameters = [
            f"{current_directory}/{script_name}",
            f"--iterations {ITERATIONS}",
            f"--testsamples {TEST_SAMPLES}",
            f"--config {config_path}",
            f"--checkpointdir {checkpoint_dir}",
            f"--datadir {data_dir}",
            f"--goldpath {gold_path}",
        ]
        execute_parameters = parameters + ["--disableconsolelog"]
        command_list = [{
            "killcmd": f"pkill -9 -f {script_name}",
            "exec": " ".join(execute_parameters),
            "codename": script_name,
            "header": " ".join(execute_parameters)
        }]

        generate_cmd = " ".join(parameters + ["--generate"])
        # dump json
        with open(json_file_name, "w") as json_fp:
            json.dump(obj=command_list, fp=json_fp, indent=4)

        print(f"Executing generate for {generate_cmd}")
        if os.system(generate_cmd) != 0:
            raise OSError(f"Could not execute command {generate_cmd}")

    print("Json creation and golden generation finished")
    print(f"You may run: scp -r {jsons_path} carol@{server_ip}:{home}/radiation-setup/machines_cfgs/")


def test_all_jsons(timeout=30):
    hostname = gethostname()
    jsons_path = f"data/{hostname}_jsons"

    for file in glob.glob(rf"{jsons_path}/*.json", recursive=True):
        with open(file, "r") as fp:
            json_data = json.load(fp)

        for v in json_data:
            print("EXECUTING", v["exec"])
            os.system(v['exec'] + "&")
            time.sleep(timeout)
            os.system(v["killcmd"])


if __name__ == "__main__":
    if TEST_JSONS:
        test_all_jsons()
    else:
        main()
