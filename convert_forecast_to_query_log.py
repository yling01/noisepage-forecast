import multiprocessing
import os
import pickle
import queue
import time
from pathlib import Path

import pandas as pd
import psycopg

from constants import DEBUG_FORECAST_FOLDER, DB_CONN_STRING, DEBUG_POSTGRESQL_PARQUET_TRAIN
from generated_forecast_md import GeneratedForecastMD
from sql_util import substitute


def work_fn(ts_start, session_jobs):
    worker_pid = os.getpid()
    print(f"Worker [{worker_pid}] started.")
    with psycopg.connect(DB_CONN_STRING, autocommit=True) as conn:
        with conn.cursor() as cursor:
            while True:
                try:
                    work = session_jobs.get(block=False)
                    forecast_path = work["forecast_path"]
                    session_id = work["session_id"]
                    session_df = work["session_df"]
                    session_start = ts_start + work["arrive_after"]
                    ts_current = pd.Timestamp.now().tz_localize("UTC")
                    # TODO(WAN): little bobby tables
                    session_df["query"] = session_df.apply(
                        lambda row: substitute(row["query_template"], row["query_params"]), axis=1)

                    wait_time_s = (session_start - ts_current).total_seconds()
                    if wait_time_s > 0:
                        time.sleep(wait_time_s)
                    # print(f"Worker [{worker_pid}] executing session [{session_id}] from: {forecast_path}")

                    queries = session_df["query"].values
                    assert queries[0] == "BEGIN", f"Unsupported: {queries[0]}"
                    assert queries[-1] in ["COMMIT", "ROLLBACK"], f"Unsupported: {queries[-1]}"

                    with conn.transaction():
                        for query in queries[1:-1]:
                            try:
                                cursor.execute(query)
                            except psycopg.Error as err:
                                raise psycopg.Rollback
                                # print(f"Worker [{worker_pid}] session [{session_id}] error {err} at this query, aborting: {query}")
                        # if queries[-1] == "ROLLBACK":
                        #     raise psycopg.Rollback
                except queue.Empty:
                    print(f"Worker finished: {worker_pid}")
                    break


def main():
    forecasts = []
    forecasts.extend(sorted(Path(DEBUG_FORECAST_FOLDER).glob("*.parquet")))
    print(f"Forecasts: {forecasts}")

    forecasts_path = Path(DEBUG_FORECAST_FOLDER)
    with open(forecasts_path / "generated_forecast_md.pkl", "rb") as f:
        generated_forecast_md: GeneratedForecastMD = pickle.load(f)
        granularity = generated_forecast_md.granularity

    with multiprocessing.Manager() as manager:
        session_jobs = manager.Queue()
        for forecast in forecasts:
            df = pd.read_parquet(forecast)
            num_sessions = df["session_id"].nunique()
            if num_sessions == 0:
                continue

            # Every session must fit into the given granularity.
            time_between = granularity / num_sessions
            # Uniform arrival times.
            starts = pd.Series([time_between * i for i in range(num_sessions)])

            # sort=False to preserve the order within the session df, i.e., "1,2,11" instead of "1,11,2".
            groups = df.groupby("session_id", sort=False)
            print(f"Loading {len(groups)} sessions from {forecast}.")
            for i, (session_id, session_df) in enumerate(groups):
                job = {
                    "arrive_after": starts[i],
                    "session_id": session_id,
                    "session_df": session_df,
                    "forecast_path": forecast,
                }
                session_jobs.put(job)

        ts_start = pd.Timestamp.now().tz_localize("UTC")

        workers = []
        for i in range(os.cpu_count()):
            worker = multiprocessing.Process(target=work_fn, args=(ts_start, session_jobs), name=f"Worker {i}")
            workers.append(worker)

        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()


if __name__ == "__main__":
    main()
