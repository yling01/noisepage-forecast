import pandas as pd
import re


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


def get_params(df, query):
    raise NotImplementedError
    pass


def sample_params(df: pd.DataFrame, query: str, n: int, replace: bool = True, weights: bool = True):
    """
    Find a sampling of parameters associated with a particular query.
    @param df: preprocessed query log, "query_template" column has to be present
    @param query: The query template to look up parameters for.
    @param n: The number of parameter vectors to sample.
    @param replace: True if the sampling should be done with replacement.
    @param weights: True if the sampling should use the counts as weights.
                    False if the sampling should use equal probability weighting.
    @return: Sample of the parameters associated with a particular query.
    """
    params = get_params(df, query)
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
