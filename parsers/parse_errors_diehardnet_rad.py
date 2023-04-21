#!/usr/bin/python3
import argparse
import os

import pandas as pd

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
        if any([i in subdir for i in ["carolp", "carolm", "carola"]]):
            print("Parsing", subdir)
            for file in files:
                path = os.path.join(subdir, file)
                new_line = parse_log_file(log_path=path)
                if new_line:
                    data_list.extend(new_line)

    df = pd.DataFrame(data_list)
    df = df.fillna(0)
    df.to_csv("../data/parsed_logs_rad.csv", index=False)


if __name__ == '__main__':
    main()
