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
        return [self._encodings[label] for label in labels]

    def fit_transform(self, labels):
        return self.fit(labels).transform(labels)

    def inverse_transform(self, encodings):
        return [self._inverse[encoding] for encoding in encodings]


class QtMeta:
    def __init__(self):
        self._think_time_sketch = DDSketch()

    def record(self, think_time):
        self._think_time_sketch.add(think_time)

class DfMeta:
    SESSION_BEGIN = "SESSION_BEGIN"
    SESSION_END = "SESSION_END"

    def __init__(self):
        self.qtms = {}
        self.qt_enc = QueryTemplateEncoder()
        # Dummy tokens for session begin and session end.
        self.qt_enc.fit([self.SESSION_BEGIN, self.SESSION_END, pd.NA])

        # networkx dict_of_dicts format.
        self.transition_sessions = {}
        self.transition_txns = {}

    def augment(self, df):
        # Augment the dataframe while updating internal state.

        # Encode the query templates.
        print("Encoding query templates.")
        df["query_template_enc"] = self.qt_enc.fit_transform(df["query_template"])

        # Lagged time.
        df["think_time"] = (df["log_time"] - df["log_time"].shift(1)).shift(-1).dt.total_seconds()

        def record_thinks(row):
            qt_enc = row["query_template_enc"]
            think_time = row["think_time"]
            self.qtms[qt_enc] = self.qtms.get(qt_enc, QtMeta())
            self.qtms[qt_enc].record(think_time)

        print("Computing think times.")
        df.apply(record_thinks, axis=1)

        print("Updating transitions for sessions.")
        self._update_transition_dict(self.transition_sessions, self._compute_transition_dict("session_id"))
        print("Updating transitions for transactions.")
        self._update_transition_dict(self.transition_txns, self._compute_transition_dict("virtual_transaction_id"))

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
        nx.relabel_nodes(G, {k: rewrite(dfm.qt_enc.inverse_transform([k])[0]) for k in G.nodes}, copy=False)
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

    def _compute_transition_dict(self, group_key):
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


dfm = DfMeta()
for pq_file in tqdm(sorted(list(Path(DEBUG_POSTGRESQL_PARQUET_FOLDER).glob("*.parquet"))),
                    desc="Reading Parquet files.",
                    disable=True):
    df = pd.read_parquet(pq_file)
    df["query_template"] = df["query_template"].replace("", np.nan)
    dropna_before = df.shape[0]
    df = df.dropna(subset=["query_template"])
    dropna_after = df.shape[0]
    print(f"Dropped {dropna_before - dropna_after} empty query templates in {pq_file}.")
    dfm.augment(df)
    break
# dfm.visualize("sessions")
# dfm.visualize("txns")