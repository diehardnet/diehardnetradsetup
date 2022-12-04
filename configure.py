#!/usr/bin/python3
import argparse
import configparser
import glob
import json
import os.path
import time
from socket import gethostname
from pathlib import Path
import configs

ALL_DNNS = configs.DIEHARDNET_CLASSIFICATION_CONFIGS
ALL_DNNS += configs.DIEHARDNET_TRANS_LEARNING_CONFIGS
ALL_DNNS += configs.DIEHARDNET_SEGMENTATION_CONFIGS

CONFIG_FILE = "/etc/radiation-benchmarks.conf"
ITERATIONS = int(1e12)
TEST_SAMPLES = {
    **{k: 128 * 64 for k in configs.DIEHARDNET_CLASSIFICATION_CONFIGS},
    **{k: 64 * 16 for k in configs.DIEHARDNET_TRANS_LEARNING_CONFIGS},
    **{k: 32 * 4 for k in configs.DIEHARDNET_SEGMENTATION_CONFIGS},
}


def configure(download_datasets: bool, download_models: bool):
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

    if download_models:
        print("Download all the models")
        download_models_process()

    current_directory = os.getcwd()
    script_name = "main.py"
    for dnn_model in ALL_DNNS:
        # Default filename will build the other names
        default_file_name = dnn_model.replace(".yaml", "")
        json_file_name = f"{jsons_path}/{default_file_name}.json"
        data_dir = f"{current_directory}/data"
        gold_path = f"{data_dir}/{default_file_name}.pt"
        checkpoint_dir = f"{data_dir}/checkpoints"
        config_path = f"{current_directory}/configurations/{dnn_model}.yaml"
        parameters = [
            f"{current_directory}/{script_name}",
            f"--iterations {ITERATIONS}",
            f"--testsamples {TEST_SAMPLES[dnn_model]}",
            f"--config {config_path}",
            f"--checkpointdir {checkpoint_dir}",
            f"--goldpath {gold_path}",
        ]
        execute_parameters = parameters + ["--disableconsolelog"]
        command_list = [{
            "killcmd": f"pkill -9 -f {script_name}",
            "exec": " ".join(execute_parameters),
            "codename": default_file_name,
            "header": " ".join(execute_parameters)
        }]

        if download_datasets:
            parameters += ["--downloaddataset"]
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
    print("JSONS PATH:", jsons_path)
    for file in glob.glob(rf"{jsons_path}/*.json", recursive=True):
        with open(file, "r") as fp:
            json_data = json.load(fp)

        for v in json_data:
            print("EXECUTING", v["exec"])
            os.system(v['exec'] + "&")
            time.sleep(timeout)
            os.system(v["killcmd"])


def download_models_process():
    links = [
        # Mobile net
        # CIFAR 10
        # "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/mobilenetv2/cifar10_mobilenetv2_x1_4-3bbbd6e2.pt",
        # # CIFAR 100
        # "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/mobilenetv2/cifar100_mobilenetv2_x1_4-8a269f5e.pt",

        # Diehardnet all
        "https://www.dropbox.com/s/4497lt4a72l9yw3/chipir_2022.tar.gz",

        # Diehardnet transfer learning
        "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",

        # Deeplav3
        "https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth",

    ]
    check_points = "data/checkpoints"
    if os.path.isdir(check_points) is False:
        os.mkdir(check_points)

    for link in links:
        file_name = os.path.basename(link)
        print(f"Downloading {file_name}")
        final_path = f"{check_points}/{file_name}"

        if os.path.isfile(final_path) is False:
            assert os.system(f"wget {link} -P {check_points}") == 0, "Download mobile net weights not successful"

        if ".tar.gz" in file_name:
            assert os.system(
                f"tar xzf {final_path} -C {check_points}") == 0, "Extracting the checkpoints not successful"


def main():
    parser = argparse.ArgumentParser(description='Configure a setup', add_help=True)
    parser.add_argument('--testjsons', default=0,
                        help="How many seconds to test the jsons, if 0 (default) it does the configure", type=int)
    parser.add_argument('--downloaddataset', default=False, action="store_true", help="Download the datasets")
    parser.add_argument('--downloadmodels', default=False, action="store_true", help="Download the models")

    args = parser.parse_args()

    if args.testjsons != 0:
        test_all_jsons(timeout=args.testjsons)
    else:
        configure(download_datasets=args.downloaddataset, download_models=args.downloadmodels)


if __name__ == "__main__":
    main()
