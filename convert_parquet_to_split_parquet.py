import pandas as pd
import numpy as np

import csv
from pathlib import Path
from constants import DEBUG_POSTGRESQL_PARQUET_FOLDER, DEBUG_POSTGRESQL_PARQUET_TRAIN, DEBUG_POSTGRESQL_PARQUET_FUTURE, \
    PG_LOG_DTYPES, DEBUG_POSTGRESQL_PARQUET_DATA, DEBUG_POSTGRESQL_CSV_TRAIN, DEBUG_POSTGRESQL_CSV_FUTURE

from collections import namedtuple

LogEntry = namedtuple("LogEntry", PG_LOG_DTYPES.keys())


def main():
    pq_files = sorted(list(Path(DEBUG_POSTGRESQL_PARQUET_FOLDER).glob("postgres*.parquet")))

    out_dir = Path(DEBUG_POSTGRESQL_PARQUET_DATA)
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(pq_files) == 0:
        print("No Parquet files found?")
        return

    df = pd.concat(pd.read_parquet(pq_file) for pq_file in pq_files)
    # Hold out half of the dataframe.
    num_splits = 2
    splits = np.array_split(df, num_splits)
    print("Splitting Parquet into train/test data.")
    splits[0].to_parquet(DEBUG_POSTGRESQL_PARQUET_TRAIN)
    print("Train: ", splits[0]["log_time"].min(), splits[0]["log_time"].max())
    splits[1].to_parquet(DEBUG_POSTGRESQL_PARQUET_FUTURE)
    print("Test: ", splits[1]["log_time"].min(), splits[1]["log_time"].max())

    split_point = splits[1].iloc[0]
    source_csvlog = "./artifacts/train/query_log.csv"

    with open(source_csvlog, "r", newline="") as source_csv, open(DEBUG_POSTGRESQL_CSV_TRAIN, "w") as train_csv, open(
            DEBUG_POSTGRESQL_CSV_FUTURE, "w") as future_csv:
        reader = csv.reader(source_csv, quoting=csv.QUOTE_ALL)
        train_writer = csv.writer(train_csv, lineterminator="\n", quoting=csv.QUOTE_ALL)
        future_writer = csv.writer(future_csv, lineterminator="\n", quoting=csv.QUOTE_ALL)
        switch = False

        for i, row in enumerate(reader):
            log_entry = LogEntry(*row)
            if str(log_entry.session_id) == str(split_point["session_id"]) and str(log_entry.session_line_num) == str(
                    split_point["session_line_num"]):
                switch = True
                print(f"Split the input CSV at line {i}.")

            writer = future_writer if switch else train_writer
            writer.writerow(row)


if __name__ == "__main__":
    main()
