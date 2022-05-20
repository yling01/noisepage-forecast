import pandas as pd
import re
import numpy as np
from sql_util import substitute


def substitute_params(query_template, params):
    assert type(query_template) == str
    query = query_template
    keys = [f"${i}" for i in range(1, len(params) + 1)]
    for k, v in reversed(list(zip(keys, params))):
        # note: The reversing is crucial! $1 is a prefix of $10.
        query = query.replace(k, v)
    return query


def substitute_given_row(row):
    query, params = row["query_template"], row["query_params"]
    if query is pd.NA or query is np.nan:
        return pd.NA
    keys = list(map(lambda x: f'${x}', range(1, len(params) + 1)))
    params_as_dict = dict(zip(keys, params))
    return substitute(str(query), params_as_dict, onerror="ignore")


def get_grouped_dataframe_interval(df: pd.DataFrame, interval: pd.Timedelta = None) -> pd.DataFrame:
    """
    Get the pre-grouped version of query log data.
    @param df: preprocessed query log, "query_template" column has to be present
    @param interval: time interval to group and count the query templates,
                     pd is only aggregated by template if not specified
    @return: Dataframe containing the pre-grouped query log data.
             Optionally grouped on query template and log time.
    """

    assert ("query_template" in df)

    if interval is None:
        gb = df.groupby("query_template").size()
        gb.drop("", axis=0, inplace=True)
    else:
        gb = df.groupby("query_template").resample(interval).size()
        gb.drop("", axis=0, level=0, inplace=True)
    grouped_df = pd.DataFrame(gb, columns=["count"])
    return grouped_df


def get_params(grouped_df, query):
    """
    Find the parameters associated with a particular query.
    @param grouped_df:
    @param query:
    @return:
    """
    params = grouped_df.query("query_template == @query")
    return params.droplevel(0).squeeze(axis=1)


def sample_params(grouped_by_params: pd.DataFrame, query: str, n: int, replace: bool = True,
                  weights: bool = True) -> np.ndarray:
    """
    Find a sampling of parameters associated with a particular query.
    @param grouped_by_params: preprocessed query log grouped by "query_template" and "query_params"
    @param query: The query template to look up parameters for.
    @param n: The number of parameter vectors to sample.
    @param replace: True if the sampling should be done with replacement.
    @param weights: True if the sampling should use the counts as weights.
                    False if the sampling should use equal probability weighting.
    @return: Sample of the parameters associated with a particular query.
    """
    params = get_params(grouped_by_params, query)
    weight_vec = params if weights else None
    sample = params.sample(n, replace=replace, weights=weight_vec)
    return sample.index.to_numpy()


def parse_time_delta(time_delta_as_string: str) -> pd.Timedelta:
    """
    Parse a string representation of time interval and construct a pd.Timedelta object
    @param time_delta_as_string: string representation of a time interval following (\d+)( *)(\w+)
    @return: a pd.Timedelta object constructed from the string representation
    """
    m = re.match(r"(?P<time>\d+)( *)(?P<unit>\w+)", time_delta_as_string)
    time = int(m.group("time"))
    unit = m.group("unit")
    return pd.Timedelta(time, unit)
