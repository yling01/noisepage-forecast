import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import constants as K
from pathlib import Path
import query_log_util


class Clusterer:
    """
    Cluster query templates based on the algorithms from QueryBot5000.

    [QueryBot5000]
    Lin Ma, Dana Van Aken, Ahmed Hefny, Gustavo Mezerhane, Andrew Pavlo,
    and Geoffrey J. Gordon. 2018. Query-based Workload Forecasting for
    Self-Driving Database Management Systems. SIGMOD 2018.

    Attributes
    ----------
    _df : pd.Dataframe
        Dataframe of counts grouped by (template, log_time_s)
        where log_time_s is aggregated to the clustering_interval
    n_samples : int
        Number of samples to use for calculating similarity between arrival rates.
    rho : float
        Similarity threshold used to determine template cluster membership.
    min_time : pd.Timestamp
        Earliest timestamp seen in _df.
    max_time : pd.Timestamp
        Latest timestamp seen in _df.
    cluster_interval : pd.Timedelta
        Time interval the df is aggregated by.
    n : int
        Number of datapoints in _df.
    cluster_gap : int
        Only use every x "time steps" to iterate for online clustering.
    n_gaps : int
        Number of time steps to to run online clustering.
    _dbgname : dict (string:int)
        Reverse lookup from query template string to an id.
    """

    def __init__(
            self,
            dataframe,
            n_samples=10000,
            rho=0.8,
            cluster_interval=pd.Timedelta(seconds=1),
    ):
        """
        Cluster the provided dataframe according to QueryBot5000.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataframe containing the query templates to be clustered.
        n_samples : int
            The number of timestamps to sample.
        rho : float
            Cosine similarity threshold for query template clustering.
        cluster_interval : pd.TimeDelta
            Time interval to group and count the query templates.
        """
        assert dataframe.index.names == ["query_template", "log_time_s"]
        assert dataframe.columns.values == ["count"]
        self._df = dataframe
        self.n_samples = n_samples
        self.rho = rho

        # Cluster interval of every second.
        self.min_time = self._get_timestamps().min()
        self.max_time = self._get_timestamps().max()

        self.interval_delta = cluster_interval
        self.n = int((self.max_time - self.min_time) / self.interval_delta + 1)

        self.cluster_gap = 1
        self.n_gaps = self.n // self.cluster_gap + 1

        # Represent query templates with integers for concise readability.
        self._dbgname = {
            template_str: template_id for template_id, template_str in dict(enumerate(self._get_queries())).items()
        }

        # Cluster the queries.
        self.assignment_df = self._cluster_offline()

    def _get_queries(self):
        """
        Get the query templates being clustered.

        Returns
        -------
        queries : List[str]
            A list of the query templates being clustered.
        """
        return sorted(set(self._df.index.get_level_values(0)))

    def _get_timestamps(self):
        """
        Get all the timestamps across all the query templates.

        Returns
        -------
        timestamps : pd.DatetimeIndex
            All the timestamps.
        """

        # TODO(Mike): Are we ever relying on the date time index here to
        # reconstruct the time series with the clustering interval?
        # Could anything go wrong if this only has
        # 00:00, 00:01, 00:03, 00:04, but missing 00:02?
        return self._df.index.get_level_values(1)

    @staticmethod
    def _query_df(df, template, timestamps):
        """
        Get template counts, sampled by timestamps

        Parameters
        ----------
        df
        template
        timestamps

        Returns
        -------
        results : pd.DataFrame
        """
        # The first level can be dropped since query_template == template.
        df = df.query("`query_template` == @template and `log_time_s` in @timestamps").droplevel(0)
        return df.reindex(timestamps, fill_value=0)

    @staticmethod
    def _sample_timestamps(n, start_time, end_time, n_samples, interval):
        """

        Parameters
        ----------
        n : int
        start_time : pd.Timestamp
        end_time : pd.Timestamp
        n_samples : int
        interval : pd.TimeDelta

        Returns
        -------
        samples : pd.DatetimeArray
            Array of timestamps that were sampled.
        """
        if n > n_samples:
            offsets = np.random.choice(a=n, size=n_samples, replace=False)
        else:
            offsets = np.arange(n)
        timestamps = []
        for offset in offsets:
            next_time = start_time + interval * offset
            if next_time >= end_time:
                break
            timestamps.append(next_time)
        return pd.array(timestamps)

    def _cluster_offline(self):
        next_time = self.max_time + self.cluster_gap * self.interval_delta
        # TODO(Mike): only consider the last 10 seconds? or sample everything?
        start_time = self.min_time
        # Sample timestamps to consider.
        timestamps = self._sample_timestamps(self.n, start_time, next_time, self.n_samples, self.interval_delta)
        counts = np.array(
            [
                # Create (k,n) matrix where there are
                # k templates, n_sample features for DBSCAN.
                self._query_df(self._df, template, timestamps).values.reshape((-1))
                for template in self._get_queries()
            ]
        )

        clustering = DBSCAN(eps=1 - self.rho, metric="cosine", min_samples=1).fit(counts)
        labels = clustering.labels_
        reverse_lookup = {template_id: template_str for template_str, template_id in self._dbgname.items()}
        final_assignments = {reverse_lookup[template_id]: cluster_id for template_id, cluster_id in enumerate(labels)}
        return pd.DataFrame(final_assignments.items(), columns=["query_template", "cluster"]).set_index(
            "query_template"
        )


def main():
    print(f"Loading preprocessor data from {K.DEBUG_POSTGRESQL_PARQUET_FOLDER}.")

    # obtain the preprocessed dataframe
    pq_files = sorted(list(Path(K.DEBUG_POSTGRESQL_PARQUET_FOLDER).glob("*.parquet")))
    df = pd.concat(pd.read_parquet(pq_file) for pq_file in pq_files)

    # note: the dataframe generated from the parquet does not have an index column, need to add this explicitly
    df.set_index("log_time", inplace=True)

    # todo (Mike): This should not be hardcoded, since many components
    # todo: of the forecaster depend on this. Should be a shared constant somewhere.
    cluster_interval = pd.Timedelta(milliseconds=250)
    df = query_log_util.get_grouped_dataframe_interval(df, cluster_interval)
    df.index.rename(["query_template", "log_time_s"], inplace=1)

    print("Clustering query templates.")
    clusterer = Clusterer(df, cluster_interval=cluster_interval)

    print("Generating cluster assignments.")
    clusterer.assignment_df.to_parquet(K.DEBUG_QB5000_CLUSTERER_OUTPUT)
    print("Done!")


if __name__ == "__main__":
    main()
