import pickle
from multiprocessing import cpu_count

from typing import Dict

import networkx as nx
import numpy as np
import pandas as pd
from ddsketch import DDSketch
from pandas.api.types import is_datetime64_any_dtype
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


class QueryTemplateEncoder:
    """
    Why not sklearn.preprocessing.LabelEncoder()?

    - Not all labels (query templates) are known ahead of time.
    - Not that many query templates, so hopefully this isn't a bottleneck.
    """

    def __init__(self):
        self._encodings = {}
        self._inverse = {}
        self._next_label = 1

    def fit(self, labels):
        for label in labels:
            if label not in self._encodings:
                self._encodings[label] = self._next_label
                self._inverse[self._next_label] = label
                self._next_label += 1
        return self

    def transform(self, labels):
        if hasattr(labels, "__len__") and not isinstance(labels, str):
            return [self._encodings[label] for label in labels]
        return self._encodings[labels]

    def fit_transform(self, labels):
        return self.fit(labels).transform(labels)

    def inverse_transform(self, encodings):
        if hasattr(encodings, "__len__") and not isinstance(encodings, str):
            return [self._inverse[encoding] for encoding in encodings]
        return self._inverse[encodings]


class QueryTemplateMD:
    def __init__(self, query_template, query_template_encoding):
        self._query_template = query_template
        self._query_template_encoding = query_template_encoding
        self._think_time_sketch = DDSketch()
        self._params = []
        self._params_times = []

    def record(self, row):
        """
        Given a row in the dataframe, update the metadata object for this specific query template.
        """
        assert row["query_template"] == self._query_template, "Mismatched query template?"
        think_time = row["think_time"]
        # Unquote the parameters.
        params = [x[1:-1] for x in row["query_params"]]
        self._think_time_sketch.add(think_time)
        if len(params) > 0:
            self._params.append(params)
            self._params_times.append(row["log_time"])

    def get_historical_params(self):
        """
        Returns
        -------
        historical_params : pd.DataFrame
        """
        if len(self._params) > 0:
            params = np.stack(self._params)
            params = pd.DataFrame(params, index=pd.DatetimeIndex(self._params_times))
            for col in params.columns:
                if is_datetime64_any_dtype(params[col]):
                    params[col] = params[col].astype("datetime64")
                elif all(params[col].str.isnumeric()):
                    params[col] = params[col].astype(int)
                elif all(params[col].str.replace(".", "", regex=False).str.isnumeric()):
                    params[col] = params[col].astype(float)
            return params.convert_dtypes()
        return pd.DataFrame([], index=pd.DatetimeIndex([]))

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"QueryTemplateMD[{self._query_template_encoding=}, {len(self._params)=}, {self._query_template=}]"


class ForecastMD:
    SESSION_BEGIN = "SESSION_BEGIN"
    SESSION_END = "SESSION_END"

    def __init__(self):
        self.qtmds: Dict[QueryTemplateMD] = {}
        self.qt_enc = QueryTemplateEncoder()
        # Dummy tokens for session begin and session end.
        self.qt_enc.fit([self.SESSION_BEGIN, self.SESSION_END, pd.NA])

        # networkx dict_of_dicts format.
        self.transition_sessions = {}
        self.transition_txns = {}

        self.arrivals = []
        self._pq_files = []

        self.cache = {}

    def augment(self, pq_file):
        if pq_file in self._pq_files:
            print(f"ForecastMD object already contains this Parquet file, skipping: {pq_file}")
            return

        self._pq_files.append(pq_file)
        df = pd.read_parquet(pq_file)
        df["log_time"] = df["log_time"].dt.tz_convert("UTC")
        print(f"{pq_file} has timestamps from {df['log_time'].min()} to {df['log_time'].max()}.")
        df["query_template"] = df["query_template"].replace("", np.nan)
        dropna_before = df.shape[0]
        df = df.dropna(subset=["query_template"])
        dropna_after = df.shape[0]
        print(f"Dropped {dropna_before - dropna_after} empty query template rows in {pq_file}. "
              f"{dropna_after} rows remain.")

        # Invalidate the cache.
        self.cache = {}

        # Sort the dataframe. All the below code assumes that the dataframe is sorted chronologically.
        df = df.sort_values(["log_time", "session_line_num"])

        # Drop any torn txns from the df.
        # Specifically, it is assumed that each txn has a clear begin and end marker.
        valid_starts = ["BEGIN"]
        valid_ends = ["COMMIT", "ROLLBACK"]
        vtxid_group = df.groupby("virtual_transaction_id", sort=False)
        good_starts = vtxid_group.nth(0)["query_template"].isin(valid_starts)
        good_starts = good_starts[good_starts].index
        good_ends = vtxid_group.nth(-1)["query_template"].isin(valid_ends)
        good_ends = good_ends[good_ends].index
        good_vtxids = (good_starts.intersection(good_ends)).values

        pre_drop_rows = df.shape[0]
        df = df.drop(df[~df["virtual_transaction_id"].isin(good_vtxids)].index)
        post_drop_rows = df.shape[0]
        dropped_rows = pre_drop_rows - post_drop_rows
        if dropped_rows > 0:
            print(
                f"Dropped {dropped_rows} rows belonging to torn/unconforming transactions. {post_drop_rows} rows remain.")

        # Augment the dataframe while updating internal state.

        # Encode the query templates.
        print("Encoding query templates.")
        df["query_template_enc"] = self.qt_enc.fit_transform(df["query_template"])

        # Lagged time.
        df["think_time"] = (df["log_time"] - df["log_time"].shift(1)).shift(-1).dt.total_seconds()

        def record(row):
            qt_enc = row["query_template_enc"]
            if qt_enc not in self.qtmds:
                qt = row["query_template"]
                self.qtmds[qt_enc] = QueryTemplateMD(query_template=qt,
                                                     query_template_encoding=self.qt_enc.transform(qt))
            self.qtmds[qt_enc].record(row)

        print("Recording query template info.")
        df.apply(record, axis=1)

        print("Updating transitions for sessions.")
        self._update_transition_dict(self.transition_sessions, self._compute_transition_dict(df, "session_id"))
        print("Updating transitions for transactions.")
        self._update_transition_dict(self.transition_txns, self._compute_transition_dict(df, "virtual_transaction_id"))

        # We need to keep the arrivals around.
        # Assumption: every transaction starts with a BEGIN.
        # Therefore, only the BEGIN entries need to be considered.
        # TODO(WAN): Other ways of starting transactions.
        print("Keeping historical arrival times.")
        begin_times = df.loc[df["query_template"] == "BEGIN", "log_time"]
        self.arrivals.append(begin_times)

    def visualize(self, target):
        assert target in ["sessions", "txns"], f"Bad target: {target}"

        if target == "sessions":
            transitions = self.transition_sessions
        else:
            assert target == "txns"
            transitions = self.transition_txns

        def rewrite(s):
            l = 24
            return "\n".join(s[i:i + l] for i in range(0, len(s), l))

        G = nx.DiGraph(transitions)
        nx.relabel_nodes(G, {k: rewrite(self.qt_enc.inverse_transform(k)) for k in G.nodes}, copy=False)
        AG = nx.drawing.nx_agraph.to_agraph(G)
        AG.layout("dot")
        AG.draw(f"{target}.pdf")

    @staticmethod
    def _update_transition_dict(current, other):
        for src in other:
            current[src] = current.get(src, {})
            for dst in other[src]:
                current[src][dst] = current[src].get(dst, {"weight": 0})
                current[src][dst]["weight"] += other[src][dst]["weight"]
                # Set the label for printing.
                current[src][dst]["label"] = current[src][dst]["weight"]

    def _compute_transition_dict(self, df, group_key):
        assert group_key in ["session_id", "virtual_transaction_id"], f"Unknown group key: {group_key}"

        group_fn = None
        if group_key == "session_id":
            group_fn = self._group_session
        elif group_key == "virtual_transaction_id":
            group_fn = self._group_txn
        assert group_fn is not None, "Forgot to add a case?"

        transitions = {}
        groups = df.groupby(group_key)
        chunksize = max(1, len(groups) // cpu_count())
        grouped = process_map(group_fn, groups, chunksize=chunksize, desc=f"Grouping on {group_key}.", disable=True)
        # TODO(WAN): Parallelize.
        for group_id, group_qt_encs in tqdm(grouped, desc=f"Computing transition matrix for {group_key}.",
                                            disable=True):
            for transition in zip(group_qt_encs, group_qt_encs[1:]):
                src, dst = transition
                transitions[src] = transitions.get(src, {})
                transitions[src][dst] = transitions[src].get(dst, {"weight": 0})
                transitions[src][dst]["weight"] += 1
                transitions[src][dst]["label"] = transitions[src][dst]["weight"]
        return transitions

    def _group_txn(self, item):
        group_id, df = item
        qt_encs = df["query_template_enc"].values
        return group_id, qt_encs

    def _group_session(self, item):
        group_id, df = item
        qt_encs = df["query_template_enc"].values
        qt_encs = np.concatenate([
            self.qt_enc.transform([self.SESSION_BEGIN]),
            qt_encs,
            self.qt_enc.transform([self.SESSION_END]),
        ])
        return group_id, qt_encs

    def get_qtmd(self, query_template):
        """
        Parameters
        ----------
        query_template : str

        Returns
        -------
        qtmd : QueryTemplateMD
        """
        encoding = self.qt_enc.transform(query_template)
        return self.qtmds[encoding]

    def get_cache(self):
        return self.cache

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
