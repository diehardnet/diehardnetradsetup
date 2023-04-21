#!/usr/bin/python3
import pandas as pd


def is_error_valid(row, cross_section_dict):
    # load the Excel file
    cross_section = cross_section_dict[row["hostname"]]
    return int(cross_section[(cross_section["start_dt"] <= row["start_dt"]) &
                             (cross_section["end_dt"] >= row["start_dt"])].shape[0] != 0)


def main():
    df = pd.read_csv("../data/parsed_logs_rad.csv")
    unique_hostnames = dict()
    for hostname in df["hostname"].unique():
        df_hostname = pd.read_excel("/home/fernando/git_research/ChipIR_December_2022/data/cross_section.xlsx",
                                    sheet_name=hostname)
        df_hostname["start_dt"] = pd.to_datetime(df_hostname["start_dt"])
        df_hostname["end_dt"] = pd.to_datetime(df_hostname["end_dt"])

        unique_hostnames[hostname] = df_hostname

    df["start_dt"] = pd.to_datetime(df["start_dt"])

    df["is_error_valid"] = df.apply(is_error_valid, axis="columns", args=(unique_hostnames,))
    df = df[df["is_error_valid"] == 1]
    print(df)
    df.to_csv("../data/parsed_logs_rad_valid_ones.csv", index=False)


if __name__ == '__main__':
    main()
