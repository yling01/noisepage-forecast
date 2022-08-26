import csv
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import constants as K
from QB5000_0_model import LSTM, ForecastDataset
import query_log_util


class ClusterForecaster:
    """
    Predict cluster in workload using trained LSTMs.

    Attributes
    ----------
    prediction_interval : pd.Timedelta
        Time interval to aggregate cluster counts by.
    prediction_horizon : pd.Timedelta
        The prediction horizon of the models to train.
    prediction_seqlen : int
        Number of intervals to feed the LSTM for a prediction.
    models : Dict[int, LSTM]
        Dictionary of trained models to perform inference by

    """

    MODEL_PREFIX = "model_"

    @staticmethod
    def cluster_to_file(path, cluster):
        """Generate model file path from cluster name"""
        return f"{path}/{ClusterForecaster.MODEL_PREFIX}{cluster}.pkl"

    @staticmethod
    def get_cluster_from_file(filename):
        """Infer cluster id from file name"""
        m = re.search(f"(?<={ClusterForecaster.MODEL_PREFIX})[^/]*(?=\\.pkl)", filename)
        if m is None:
            raise RuntimeError("Could not get cluster name")
        return m[0]

    def __init__(
            self,
            train_df,
            prediction_seqlen,
            prediction_interval,
            prediction_horizon,
            save_path,
            top_k=5,
            override=False,
    ):
        """Construct the ClusterForecaster object.
        Parameters
        ----------
        train_df : pd.DataFrame
            Training data grouped by cluster and timestamp
        save_path : str
            Directory for loading/saving trained models
        top_k : int
            Only train models for the top k most common clusters.
        override : bool
            Determines whether we should (re)train models anyway, even if they are
            in the directory.
        """
        assert train_df.index.names[0] == "cluster"
        assert train_df.index.names[1] == "log_time_s"

        self.prediction_seqlen = prediction_seqlen
        self.prediction_interval = prediction_interval
        self.prediction_horizon = prediction_horizon
        self.models = {}

        if not override:
            model_files = glob.glob(str(Path(save_path) / f"{self.MODEL_PREFIX}*.pkl"))
            for filename in model_files:
                cluster_name = self.get_cluster_from_file(filename)
                self.models[int(cluster_name)] = LSTM.load(filename)
                print(f"loaded model for cluster {cluster_name}")
            print(f"Loaded {len(model_files)} models")

        if train_df is None:
            return

        # Only consider top k clusters.
        cluster_totals = train_df.groupby(level=0).sum().sort_values(by="count", ascending=False)
        labels = cluster_totals.index[:top_k]

        print("Training on cluster time series..")

        mintime = train_df.index.get_level_values(1).min()
        maxtime = train_df.index.get_level_values(1).max()

        dtindex = pd.DatetimeIndex([mintime, maxtime])

        for cluster in labels:
            if cluster in self.models and not override:
                print(f"Already have model for cluster {cluster}, skipping")
                continue

            print(f"training model for cluster {cluster}")
            cluster_counts = train_df[train_df.index.get_level_values(0) == cluster].droplevel(0)

            # This zero-fills the start and ends of the cluster time series.
            cluster_counts = cluster_counts.reindex(cluster_counts.index.append(dtindex), fill_value=0)
            cluster_counts = cluster_counts.resample(prediction_interval).sum()
            self._train_cluster(cluster_counts, cluster, save_path)

    def _train_cluster(self, cluster_counts, cluster, save_path):
        dataset = ForecastDataset(
            cluster_counts,
            sequence_length=self.prediction_seqlen,
            horizon=self.prediction_horizon,
            interval=self.prediction_interval,
        )

        self.models[cluster] = LSTM(
            horizon=self.prediction_horizon,
            interval=self.prediction_interval,
            sequence_length=self.prediction_seqlen,
        )

        self.models[cluster].fit(dataset)
        self.models[cluster].save(self.cluster_to_file(save_path, cluster))

    def predict(self, cluster_df, cluster, start_time, end_time):
        """
        Given a cluster dataset, attempt to return prediction of query count
        from a cluster within the given time-range.
        """
        assert cluster_df.index.names[0] == "cluster"
        assert cluster_df.index.names[1] == "log_time_s"

        # Cluster not in the data.
        if cluster not in cluster_df.index.get_level_values(0):
            return None

        # No model for given cluster.
        if cluster not in self.models.keys():
            return None

        cluster_counts = cluster_df[cluster_df.index.get_level_values(0) == cluster].droplevel(0)

        # Truncate cluster_df to the time range necessary to generate prediction range.

        # TODO(Mike): Right now, if the sequence required to predict a certain interval
        # is not present in the data, we simply do not make any predictions (i.e. return 0)
        # Should we produce a warning/error so the user is aware there is insufficient
        # data?
        trunc_start = start_time - self.prediction_horizon - (self.prediction_seqlen) * self.prediction_interval
        trunc_end = end_time - self.prediction_horizon

        truncated = cluster_counts[(cluster_counts.index >= trunc_start) & (cluster_counts.index < trunc_end)]

        dataset = ForecastDataset(
            truncated,
            sequence_length=self.prediction_seqlen,
            horizon=self.prediction_horizon,
            interval=self.prediction_interval,
        )

        # generate predictions
        predictions = [self.models[cluster].predict(seq) for seq, _ in dataset]

        # tag with timestamps
        pred_arr = [[dataset.get_y_timestamp(i), pred] for i, pred in enumerate(predictions)]

        pred_df = pd.DataFrame(pred_arr, columns=["log_time_s", "count"])
        pred_df.set_index("log_time_s", inplace=True)
        return pred_df[start_time:]


class WorkloadGenerator:
    """
    Use preprocessed query template/params and cluster to generate
    representative workload.
    """

    def __init__(self, df, assignment_df):
        self.df = df

        grouped_df = query_log_util.get_grouped_dataframe_interval(df)

        # Join to cluster and group by.
        joined = grouped_df.join(assignment_df)

        # Calculate weight of template within each cluster.
        joined["cluster"].fillna(-1, inplace=True)
        summed = joined.groupby(["cluster", "query_template"]).sum()
        self._percentages = summed / summed.groupby(level=0).sum()

    def get_workload(self, cluster, cluster_count):
        """Given a cluster id and a sample size, produce a "sample" workload,
        sampling from the preprocessed queries.

        Parameters
        ----------
        cluster : int
            The cluster to generate for
        cluster_count : scalar
            The number of queries to sample

        Returns
        -------
        predicted_queries : pd.Dataframe
            A sampled workload in the form of (query, count) pairs
        """

        templates = self._percentages[self._percentages.index.get_level_values(0) == cluster].droplevel(0)
        # note: this is making the assumption that the make up of the workload within a cluster
        #  does not change over time
        templates = templates * cluster_count

        # only care about insert and delete query templates
        insert_delete_templates = templates[templates.index.str.contains("^(?:DELETE|INSERT)")]

        insert_delete_templates = insert_delete_templates.sort_values(ascending=False, by="count").astype(int)

        return insert_delete_templates


def main():
    # note: the prediction horizon and prediction interval are temporarily set to be the same as the interval
    # note: at which the query logs are aggregated to be clustered
    pred_horizon = pred_interval = query_log_util.parse_time_delta(K.TIME_INTERVAL)

    # create prediction model directory if necessary
    Path(K.DEBUG_QB5000_MODEL_DIR_NEW).mkdir(parents=True, exist_ok=True)

    # note: directly read the preprocessed parquet
    pq_files = sorted(list(Path(K.DEBUG_POSTGRESQL_PARQUET_FOLDER).glob("*.parquet")))
    df = pd.concat(pd.read_parquet(pq_file) for pq_file in pq_files)

    # note: the dataframe generated from the parquet does not have an index column, need to add this explicitly
    df.set_index("log_time", inplace=True)
    grouped_df = query_log_util.get_grouped_dataframe_interval(df, pred_interval)
    grouped_df.index.rename(["query_template", "log_time_s"], inplace=1)

    print("reading cluster assignments.")
    assignment_df = pd.read_parquet(K.DEBUG_QB5000_CLUSTERER_OUTPUT)

    # Join to cluster and group by (cluster,time).
    joined = grouped_df.join(assignment_df)
    joined["cluster"].fillna(-1, inplace=True)
    clustered_df = joined.groupby(["cluster", "log_time_s"]).sum()

    # TODO (MIKE): check how many templates are not part of known clusters (i.e. cluster = -1).

    forecaster = ClusterForecaster(
        clustered_df,
        prediction_seqlen=K.PREDICTION_LENGTH,
        prediction_interval=pred_interval,
        prediction_horizon=pred_horizon,
        save_path=K.DEBUG_QB5000_MODEL_DIR_NEW,
        override=K.OVERWRITE_MODEL,
    )

    # Use preprocessor to sample template and parameter distributions.
    wg = WorkloadGenerator(df, assignment_df)
    clusters = set(assignment_df["cluster"].values)

    # note: instead of hard code the start and end timestamp, read the start and the end of the log dynamically
    with open(K.DEBUG_QB5000_PREPROCESSOR_TIMESTAMP) as f:
        log_start_time = pd.Timestamp(f.readline().strip())
        log_end_time = pd.Timestamp(f.readline().strip())

    # note: use half of the log as training data and the other half as validation
    start_ts = log_start_time + (log_end_time - log_start_time) / 2
    end_ts = log_end_time

    cluster_predictions = []
    for cluster in clusters:
        start_time = pd.Timestamp(start_ts)
        end_time = pd.Timestamp(end_ts)
        pred_df = forecaster.predict(clustered_df, cluster, start_time, end_time)
        if pred_df is None:
            # No data or model for cluster.
            continue
        prediction_count = pred_df["count"].sum()
        print(f"Prediction for {cluster}: {prediction_count}")
        cluster_predictions.append(wg.get_workload(cluster, prediction_count))

    predicted_queries = pd.concat(cluster_predictions)
    predicted_queries.to_csv(K.DEBUG_QB5000_FORECASTER_PREDICTION_CSV_NEW, header=None, quoting=csv.QUOTE_ALL)


if __name__ == "__main__":
    main()
