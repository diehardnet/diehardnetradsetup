#!/usr/bin/python3
import argparse
import configparser
import json
import os.path
import time
from pathlib import Path
from socket import gethostname

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
    if download_datasets:
        print("Download CIFAR and COCO datasets")
        download_datasets_process()

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
    current_directory = os.getcwd()
    for config in ALL_DNNS:
        file = f"{current_directory}/data/{hostname}_jsons/{config}.json"
        with open(file, "r") as fp:
            json_data = json.load(fp)

        for v in json_data:
            print("EXECUTING", v["exec"])
            os.system(v['exec'] + "&")
            time.sleep(timeout)
            os.system(v["killcmd"])


def download_models_process():
    links = [
        # Diehardnet all
        "https://www.dropbox.com/s/4497lt4a72l9yw3/chipir_2022.tar.gz",

        # Diehardnet transfer learning
        "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",

        # Deeplav3
        "https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth",
        "https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth",
        # FCN
        "https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth",
        "https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth"

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


def download_datasets_process():
    links = {
        # Cifar 10
        configs.CIFAR10: [["https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"], configs.CIFAR_DATASET_DIR],
        # Cifar 100
        configs.CIFAR100: [["https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"], configs.CIFAR_DATASET_DIR],
        # Coco
        configs.COCO: [["http://images.cocodataset.org/zips/val2017.zip",
                        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"],
                       configs.COCO_DATASET_DIR],

    }
    for dataset in links:
        data_links, dataset_dir = links[dataset]
        if os.path.isdir(dataset_dir) is False:
            os.mkdir(dataset_dir)
        current_dir = os.getcwd()
        os.chdir(dataset_dir)

        for dtl in data_links:
            uncompress_cmd = "tar xzf" if ".tar.gz" in dtl else "unzip"
            dtl_file = dtl.rsplit('/', 1)[-1]
            # Never been done before
            if os.path.isfile(dtl_file) is False:
                if os.system(f"wget {dtl}") != 0:
                    raise ConnectionError("Failed to download the dataset")
            if os.system(f"{uncompress_cmd} {dtl_file}") != 0:
                raise IOError(f"Could not uncompress the file {dtl_file}")
        os.chdir(current_dir)


def main():
    parser = argparse.ArgumentParser(description='Configure a setup', add_help=True)
    parser.add_argument('--testjsons', default=0,
                        help="How many seconds to test the jsons, if 0 (default) it does the configure", type=int)
    parser.add_argument('--downloadmodels', default=False, action="store_true", help="Download the models")
    parser.add_argument('--downloaddataset', default=False, action="store_true",
                        help="Set to download the dataset, default is to not download. Needs internet.")
    args = parser.parse_args()

    if os.path.islink("pytorch_scripts") is False:
        raise NotADirectoryError("You have to create a symlink to pytorch_scripts to be able to run the setup")

    if args.testjsons != 0:
        test_all_jsons(timeout=args.testjsons)
    else:
        configure(download_datasets=args.downloaddataset, download_models=args.downloadmodels)


if __name__ == "__main__":
    main()
