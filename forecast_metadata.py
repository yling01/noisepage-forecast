import pickle
from multiprocessing import cpu_count

import networkx as nx
import numpy as np
import pandas as pd
from ddsketch import DDSketch
from distfit import distfit
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
    def __init__(self):
        self._think_time_sketch = DDSketch()
        self._params = []

    def record(self, row):
        """
        Given a row in the dataframe, update the metadata object for this specific query template.

        Parameters
        ----------
        row

        Returns
        -------

        """
        think_time = row["think_time"]
        # Unquote the parameters.
        params = [x[1:-1] for x in row["query_params"]]
        self._think_time_sketch.add(think_time)
        if len(params) > 0:
            self._params.append(params)

    def get_historical_params(self):
        if len(self._params) > 0:
            params = np.stack(self._params)
            params = pd.DataFrame(params)
            for col in params.columns:
                if is_datetime64_any_dtype(params[col]):
                    params[col] = params[col].astype("datetime64")
                elif all(params[col].str.isnumeric()):
                    params[col] = params[col].astype(int)
                elif all(params[col].str.replace(".", "").str.isnumeric()):
                    params[col] = params[col].astype(float)
            return params.convert_dtypes()
        return []


class ForecastMD:
    SESSION_BEGIN = "SESSION_BEGIN"
    SESSION_END = "SESSION_END"

    def __init__(self):
        self.qtmds = {}
        self.qt_enc = QueryTemplateEncoder()
        # Dummy tokens for session begin and session end.
        self.qt_enc.fit([self.SESSION_BEGIN, self.SESSION_END, pd.NA])

        # networkx dict_of_dicts format.
        self.transition_sessions = {}
        self.transition_txns = {}

        self.arrivals = []

        self.cache = {}

    def augment(self, df):
        # Invalidate the cache.
        self.cache = {}
        # Augment the dataframe while updating internal state.

        # Encode the query templates.
        print("Encoding query templates.")
        df["query_template_enc"] = self.qt_enc.fit_transform(df["query_template"])

        # Lagged time.
        df["think_time"] = (df["log_time"] - df["log_time"].shift(1)).shift(-1).dt.total_seconds()

        def record(row):
            qt_enc = row["query_template_enc"]
            self.qtmds[qt_enc] = self.qtmds.get(qt_enc, QueryTemplateMD())
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
        df = df.sort_values(["log_time", "session_line_num"])
        qt_encs = df["query_template_enc"].values
        return group_id, qt_encs

    def _group_session(self, item):
        group_id, df = item
        df = df.sort_values(["log_time", "session_line_num"])
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
        query_template

        Returns
        -------
        qtmd : QueryTemplateMD
        """
        encoding = self.qt_enc.transform(query_template)
        return self.qtmds[encoding]

    def fit_historical_params(self):
        if "fit" in self.cache:
            return self.cache["fit"]

        fit = {}
        for qt_enc, qtmd in tqdm(self.qtmds.items(), total=len(self.qtmds), desc="Fitting query templates."):
            print(f"Fitting query template {qt_enc}: {self.qt_enc.inverse_transform(qt_enc)}")
            fit[qt_enc] = {}
            params = qtmd.get_historical_params()

            if len(params) == 0:
                # No parameters.
                continue

            for idx, col in enumerate(params.columns):
                fit[qt_enc][idx] = {}
                if str(params[col].dtype) == "string":
                    fit[qt_enc][idx]["type"] = "sample"
                    fit[qt_enc][idx]["sample"] = params[col]
                    print(
                        f"Query template {qt_enc} parameter {idx} is a string. Storing values to be sampled.")
                else:
                    assert not str(params[col].dtype) == "object", "Bad dtype?"
                    fit[qt_enc][idx]["type"] = "distfit"
                    dist = distfit()
                    dist.fit_transform(params[col], verbose=0)
                    print(
                        f"Query template {qt_enc} parameter {idx} fitted to distribution: {dist.model['distr'].name} {dist.model['params']}")
                    fit[qt_enc][idx]["distfit"] = dist
        self.cache["fit"] = fit
        return fit

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
