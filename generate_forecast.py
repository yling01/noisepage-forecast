from distfit import distfit
from tqdm import tqdm, trange
from neuralprophet import NeuralProphet
from pathlib import Path
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import math

from sql_parsing import substitute
from constants import DEBUG_POSTGRESQL_PARQUET_FOLDER
from forecast_metadata import ForecastMD

"""
TODO(WAN):

- fmd class. Stores all raw BEGIN times in a df.
- Hook up distfit again.

"""


def generate_forecast_arrivals(fmd, target_timestamp, granularity, plot):
    """

    Parameters
    ----------
    fmd : ForecastMD
        The forecast metadata that has the arrival times stored.

    target_timestamp : str
        The timestamp into the future that should be forecasted until.

    granularity : pd.Timedelta
        The granularity at which forecasting should be performed.
        For example, pd.Timedelta(hours=1) will forecast data at an hour granularity.

    plot : bool
        If true, will draw plots.

    Returns
    -------
    forecasted_arrivals : pd.DataFrame
    """

    # Assumption: every transaction starts with a BEGIN.
    # Therefore, only the BEGIN entries need to be considered.
    # TODO(WAN): Other ways of starting transactions.
    arrivals = pd.concat(fmd.arrivals).to_frame("log_time")
    begin_times = arrivals.set_index("log_time").resample(granularity).size()

    # Convert the dataframe to NeuralProphet format.
    ndf = begin_times.iloc[:-1].to_frame()
    ndf = ndf.tz_localize(None).reset_index().rename(columns={"log_time": "ds", 0: "y"})

    # Get the forecast time range.
    ts_last = ndf["ds"].max()
    horizon_target = pd.to_datetime(target_timestamp)
    assert horizon_target > ts_last, "Horizon is not in the future?"

    # ndf_forecast is the forecast dataframe that needs to have its "y" columns filled in.
    # NeuralProphet's make_future_dataframe is not used because it doesn't seem to expose the freq.
    dr = pd.date_range(start=ts_last, end=horizon_target, freq=granularity, inclusive="right")
    ndf_forecast = pd.DataFrame({"ds": dr, "y": [None] * len(dr)})
    ndf_forecast = pd.concat([ndf, ndf_forecast])

    model = NeuralProphet()
    # Train NeuralProphet on the original data.
    metrics = model.fit(ndf, freq=granularity)
    # Forecast the future.
    forecasted_arrivals = model.predict(ndf_forecast)

    if plot:
        model.plot(forecasted_arrivals)
        model.plot_components(forecasted_arrivals)
        plt.show()

    return forecasted_arrivals


def generate_forecast(fmd, target_timestamp, granularity=pd.Timedelta(hours=1), plot=False):
    """

    Parameters
    ----------
    fmd : ForecastMD

    Returns
    -------

    """
    forecast_arrivals = generate_forecast_arrivals(fmd, target_timestamp, granularity, plot)

    fit = fmd.fit_historical_params()

    with open("output.log", "w") as f:
        for i, row in forecast_arrivals.iterrows():
            current_ts = row.ds
            num_forecasted_sessions = math.ceil(row.yhat1)

            # Generate a sample path for each session.
            for session_num in trange(num_forecasted_sessions, desc=f"Generating sessions for {current_ts}."):
                session_line_num = 1

                qt_cur = fmd.qt_enc.transform("BEGIN")
                qt_ends = fmd.qt_enc.transform(["COMMIT", "ROLLBACK"])
                query = fmd.qt_enc.inverse_transform(qt_cur)

                # Generate a sample path for the current session.
                sample_path = []
                while True:
                    # Emit.
                    sample_path.append((session_num, session_line_num, query))
                    # Stop.
                    if qt_cur in qt_ends:
                        break

                    # Advance the session line number.
                    session_line_num += 1

                    # Pick the next query template by sampling the Markov chain.
                    transitions = fmd.transition_txns[qt_cur].items()
                    candidate_templates = [k for k, _ in transitions]
                    probs = np.array([v['weight'] for _, v in transitions])
                    probs = probs / np.sum(probs)
                    qt_cur = np.random.choice(candidate_templates, p=probs)
                    # Generate the parameters.
                    params = {}
                    for param_idx in fit.get(qt_cur, {}):
                        fit_obj = fit[qt_cur][param_idx]
                        fit_type = fit_obj["type"]
                        param_val = None
                        if fit_type == "distfit":
                            dist = fit_obj["distfit"]
                            param_val = str(dist.generate(n=1, verbose=0)[0])
                        else:
                            assert fit_type == "sample"
                            param_val = np.random.choice(fit_obj["sample"])
                        assert param_val is not None
                        params[f"${param_idx + 1}"] = param_val


                    # Combine the query template and the parameters.
                    query = fmd.qt_enc.inverse_transform(qt_cur)
                    try:
                        query = substitute(query, params)
                    except ValueError as exc:
                        query = f"ERROR: params [{params}] query [{query}]"
                        continue
                # Write the sample path.
                for session_num, session_line_num, query in sample_path:
                    print(session_num, session_line_num, query, file=f)
            # Stop after the first timestep.
            break


def main():
    # fmd = ForecastMD()
    # pq_files = [sorted(list(Path(DEBUG_POSTGRESQL_PARQUET_FOLDER).glob("*.parquet")))[0]]
    # for pq_file in tqdm(pq_files, desc="Reading Parquet files.", disable=True):
    #     df = pd.read_parquet(pq_file)
    #     df["query_template"] = df["query_template"].replace("", np.nan)
    #     dropna_before = df.shape[0]
    #     df = df.dropna(subset=["query_template"])
    #     dropna_after = df.shape[0]
    #     print(f"Dropped {dropna_before - dropna_after} empty query templates in {pq_file}.")
    #     fmd.augment(df)
    # fmd.fit_historical_params()
    # fmd.save("fmd.pkl")
    fmd = ForecastMD.load("fmd.pkl")
    generate_forecast(fmd, '2022-04-07 11:40:00', pd.Timedelta(seconds=1), plot=True)


if __name__ == "__main__":
    main()
