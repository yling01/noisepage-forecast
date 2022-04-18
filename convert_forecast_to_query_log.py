import os

from constants import DEBUG_FORECAST_FOLDER
import pandas as pd
from pathlib import Path
import pickle
import time

from generated_forecast_md import GeneratedForecastMD

import multiprocessing
import queue

from sql_util import substitute


def worker(ts_start, session_jobs):
    worker_pid = os.getpid()
    print(f"Worker [{worker_pid}] started.")
    while True:
        try:
            work = session_jobs.get(block=False)
            session_id = work["session_id"]
            session_df = work["session_df"]
            session_start = ts_start + work["arrive_after"]
            ts_current = pd.Timestamp.now().tz_localize(None)
            session_df["query"] = session_df.apply(lambda row: substitute(row["query_template"], row["query_params"]), axis=1)

            wait_time_s = (session_start - ts_current).total_seconds()
            if wait_time_s > 0:
                time.sleep(wait_time_s)
            print(f"Worker [{worker_pid}] executing session [{session_id}].")
            for query in session_df["query"]:
                pass #print(query)
        except queue.Empty:
            print(f"Worker finished: {worker_pid}")
            break



def main():
    forecasts_path = Path(DEBUG_FORECAST_FOLDER)
    with open(forecasts_path / "generated_forecast_md.pkl", "rb") as f:
        generated_forecast_md: GeneratedForecastMD = pickle.load(f)
        granularity = generated_forecast_md.granularity

    with multiprocessing.Manager() as manager:
        session_jobs = manager.Queue()
        for forecast in forecasts_path.glob("*.parquet"):
            df = pd.read_parquet(forecast)
            num_sessions = df["session_id"].nunique()

            # Every session must fit into the given granularity.
            time_between = granularity / num_sessions
            # Uniform arrival times.
            starts = pd.Series([time_between * i for i in range(num_sessions)])

            # sort=False to preserve the order within the session df, i.e., "1,2,11" instead of "1,11,2".
            for i, (session_id, session_df) in enumerate(df.groupby("session_id", sort=False)):
                job = {
                    "arrive_after": starts[i],
                    "session_id": session_id,
                    "session_df": session_df,
                }
                session_jobs.put(job)

        ts_start = pd.Timestamp.now().tz_localize(None)
        with multiprocessing.Pool() as pool:
            pool.apply(worker, (ts_start, session_jobs))


if __name__ == "__main__":
    main()
