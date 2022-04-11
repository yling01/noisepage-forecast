from tqdm import trange
from neuralprophet import NeuralProphet
import pandas as pd

import numpy as np
import math

"""
TODO(WAN):

- DFM class. Stores all raw BEGIN times in a df.
- Hook up distfit again.

"""

def generate_forecast_arrivals(df, target_timestamp, granularity=pd.Timedelta(hours=1), plot=False):
    """

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the source data to be forecast.
        Must contain a "query_template" column.

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

    assert "query_template" in df.columns, "Must contain a query_template column!"

    # Assumption: every transaction starts with a BEGIN.
    # Therefore, only the BEGIN entries need to be considered.
    # TODO(WAN): Other ways of starting transactions.
    begin_times = df[df["query_template"] == "BEGIN"].set_index("log_time").resample(granularity).size()

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

    return forecasted_arrivals


def generate_forecast(dfm):
    with open("output.log", "w") as f:
        for i, row in forecast_arrivals.iterrows():
            current_ts = row.ds
            num_forecasted_sessions = math.ceil(row.yhat1)

            # Generate a sample path for each session.
            for session_num in trange(num_forecasted_sessions, desc=f"Generating sessions for {current_ts}."):
                session_line_num = 1

                qt_cur = dfm.qt_enc.transform(["BEGIN"])[0]
                qt_ends = dfm.qt_enc.transform(["COMMIT", "ROLLBACK"])
                query = dfm.qt_enc.inverse_transform([qt_cur])[0]

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
                    transitions = dfm.transition_txns[qt_cur].items()
                    candidate_templates = [k for k, _ in transitions]
                    probs = np.array([v['weight'] for _, v in transitions])
                    probs = probs / np.sum(probs)
                    qt_cur = np.random.choice(candidate_templates, p=probs)
                    # Fill in the parameters.
                    # TODO(WAN): Do that.
                    query = dfm.qt_enc.inverse_transform([qt_cur])[0]
                # Write the sample path.
                for session_num, session_line_num, query in sample_path:
                    print(session_num, session_line_num, query, file=f)
            # Stop after the first timestep.
            break
