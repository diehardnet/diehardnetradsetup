#!/usr/bin/python3
import argparse
import os

import pandas as pd
import yaml

from common import parse_log_file


def parse_args() -> argparse.Namespace:
    """ Parse the args and return an args namespace and the tostring from the args    """
    parser = argparse.ArgumentParser(description='PyTorch DNN radiation parser', add_help=False)
    # parser = argparse.ArgumentParser(description='PyTorch DNN radiation setup')
    parser.add_argument('--logdir', help="Path to the directory that contains the logs", required=True)

    args, remaining_argv = parser.parse_known_args()

    return args


def main():
    args = parse_args()
    data_list = list()
    for subdir, dirs, files in os.walk(args.logdir):
        print("Parsing", subdir)
        if any([i in subdir for i in ["carolp", "carolm", "carola"]]):
            for file in files:
                path = os.path.join(subdir, file)
                new_line, config = parse_log_file(log_path=path)
                if new_line:
                    with open(f"../configurations/{config}.yaml") as fp:
                        batch_size = yaml.safe_load(fp)["batch_size"]
                    for dt in new_line:
                        dt["batch_size"] = batch_size
                    data_list.extend(new_line)

    df = pd.DataFrame(data_list)
    df = df.fillna(0)
    df.to_csv("../data/parsed_logs_rad.csv", index=False)


if __name__ == '__main__':
    main()
