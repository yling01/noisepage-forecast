import csv
import glob
import re
import time
from pathlib import Path
from typing import List

import pandas as pd
import pglast
from pandarallel import pandarallel
from plumbum import cli
from tqdm.contrib.concurrent import process_map
import constants as K

# Enable parallel pandas operations.
# pandarallel is a little buggy. For example, progress_bar=True does not work,
# and if you are using PyCharm you will want to enable "Emulate terminal in
# output console" instead of using the PyCharm Python Console.
# The reason we're using this library anyway is that:
# - The parallelization is dead simple: change .blah() to .parallel_blah().
# - swifter has poor string perf; we're mainly performing string ops.
# - That said, Wan welcomes any switch that works.
pandarallel.initialize(verbose=1)


class Preprocessor:
    """
    Convert PostgreSQL query logs into pandas DataFrame objects.
    """

    def get_dataframe(self):
        """
        Get a raw dataframe of query log data.

        Returns
        -------
        df : pd.DataFrame
            Dataframe containing the query log data.
            Note that irrelevant query log entries are still included.
        """
        return self._df

    def get_grouped_dataframe_interval(self, interval=None):
        """
        Get the pre-grouped version of query log data.

        Parameters
        ----------
        interval : pd.TimeDelta or None
            time interval to group and count the query templates
            if None, pd is only aggregated by template

        Returns
        -------
        grouped_df : pd.DataFrame
            Dataframe containing the pre-grouped query log data.
            Grouped on query template and optionally log time.
        """
        gb = None
        if interval is None:
            gb = self._df.groupby("query_template").size()
            gb.drop("", axis=0, inplace=True)
        else:
            gb = self._df.groupby("query_template").resample(interval).size()
            gb.drop("", axis=0, level=0, inplace=True)
        grouped_df = pd.DataFrame(gb, columns=["count"])
        return grouped_df

    def get_grouped_dataframe_params(self):
        """
        Get the pre-grouped version of query log data.

        Returns
        -------
        grouped_df : pd.DataFrame
            Dataframe containing the pre-grouped query log data.
            Grouped on query template and query parameters.
        """
        return self._grouped_df_params

    def get_params(self, query):
        """
        Find the parameters associated with a particular query.

        Parameters
        ----------
        query : str
            The query template to look up parameters for.

        Returns
        -------
        params : pd.Series
            The counts of parameters associated with a particular query.
            Unfortunately, due to quirks of the PostgreSQL CSVLOG format,
            the types of parameters are unreliable and may be stringly typed.
        """
        params = self._grouped_df_params.query("query_template == @query")
        return params.droplevel(0).squeeze(axis=1)

    def sample_params(self, query, n, replace=True, weights=True):
        """
        Find a sampling of parameters associated with a particular query.

        Parameters
        ----------
        query : str
            The query template to look up parameters for.
        n : int
            The number of parameter vectors to sample.
        replace : bool
            True if the sampling should be done with replacement.
        weights : bool
            True if the sampling should use the counts as weights.
            False if the sampling should be equal probability weighting.

        Returns
        -------
        params : np.ndarray
            Sample of the parameters associated with a particular query.
        """
        params = self.get_params(query)
        weight_vec = params if weights else None
        sample = params.sample(n, replace=replace, weights=weight_vec)
        return sample.index.to_numpy()

    @staticmethod
    def substitute_params(query_template, params):
        assert type(query_template) == str
        query = query_template
        keys = [f"${i}" for i in range(1, len(params) + 1)]
        for k, v in reversed(list(zip(keys, params))):
            # The reversing is crucial! Note that $1 is a prefix of $10.
            query = query.replace(k, v)
        return query

    @staticmethod
    def _read_csv(csvlog, log_columns):
        """
        Read a PostgreSQL CSVLOG file into a pandas DataFrame.

        Parameters
        ----------
        csvlog : str
            Path to a CSVLOG file generated by PostgreSQL.
        log_columns : List[str]
            List of columns in the csv log.

        Returns
        -------
        df : pd.DataFrame
            DataFrame containing the relevant columns for query forecasting.
        """
        # This function must have a separate non-local binding from _read_df
        # so that it can be pickled for multiprocessing purposes.
        return pd.read_csv(
            csvlog,
            names=log_columns,
            parse_dates=["log_time", "session_start_time"],
            usecols=[
                "log_time",
                "session_start_time",
                "command_tag",
                "message",
                "detail",
            ],
            header=None,
            index_col=False,
        )

    @staticmethod
    def _read_df(csvlogs, log_columns):
        """
        Read the provided PostgreSQL CSVLOG files into a single DataFrame.

        Parameters
        ----------
        csvlogs : List[str]
            List of paths to CSVLOG files generated by PostgreSQL.
        log_columns : List[str]
            List of columns in the csv log.

        Returns
        -------
        df : pd.DataFrame
            DataFrame containing the relevant columns for query forecasting.
        """
        return pd.concat(process_map(Preprocessor._read_csv, csvlogs, [log_columns for _ in csvlogs]))

    @staticmethod
    def _extract_query(message_series):
        """
        Extract SQL queries from the CSVLOG's message column.

        Parameters
        ----------
        message_series : pd.Series
            A series corresponding to the message column of a CSVLOG file.

        Returns
        -------
        query : pd.Series
            A str-typed series containing the queries from the log.
        """
        simple = r"statement: ((?:DELETE|INSERT|SELECT|UPDATE).*)"
        extended = r"execute .+: ((?:DELETE|INSERT|SELECT|UPDATE).*)"
        regex = f"(?:{simple})|(?:{extended})"
        query = message_series.str.extract(regex, flags=re.IGNORECASE)
        # Combine the capture groups for simple and extended query protocol.
        query = query[0].fillna(query[1])
        print("TODO(WAN): Disabled SQL format for being too slow.")
        # Prettify each SQL query for standardized formatting.
        # query = query.parallel_map(pglast.prettify, na_action='ignore')
        # Replace NA values (irrelevant log messages) with empty strings.
        query.fillna("", inplace=True)
        return query.astype(str)

    @staticmethod
    def _extract_params(detail_series):
        """
        Extract SQL parameters from the CSVLOG's detail column.
        If there are no such parameters, an empty {} is returned.

        Parameters
        ----------
        detail_series : pd.Series
            A series corresponding to the detail column of a CSVLOG file.

        Returns
        -------
        params : pd.Series
            A dict-typed series containing the parameters from the log.
        """

        def extract(detail):
            detail = str(detail)
            prefix = "parameters: "
            idx = detail.find(prefix)
            if idx == -1:
                return {}
            parameter_list = detail[idx + len(prefix):]
            params = {}
            for pstr in parameter_list.split(", "):
                pnum, pval = pstr.split(" = ")
                assert pnum.startswith("$")
                assert pnum[1:].isdigit()
                params[pnum] = pval
            return params

        return detail_series.parallel_apply(extract)

    @staticmethod
    def _substitute_params(df, query_col, params_col):
        """
        Substitute parameters into the query, wherever possible.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe of query log data.
        query_col : str
            Name of the query column produced by _extract_query.
        params_col : str
            Name of the parameter column produced by _extract_params.
        Returns
        -------
        query_subst : pd.Series
            A str-typed series containing the query with parameters inlined.
        """

        def substitute(query, params):
            # Consider '$2' -> "abc'def'ghi".
            # This necessitates the use of a SQL-aware substitution,
            # even if this is much slower than naive string substitution.
            new_sql, last_end = [], 0
            for token in pglast.parser.scan(query):
                token_str = str(query[token.start: token.end + 1])
                if token.start > last_end:
                    new_sql.append(" ")
                if token.name == "PARAM":
                    assert token_str.startswith("$")
                    assert token_str[1:].isdigit()
                    new_sql.append(params[token_str])
                else:
                    new_sql.append(token_str)
                last_end = token.end + 1
            new_sql = "".join(new_sql)
            return new_sql

        def subst(row):
            return substitute(row[query_col], row[params_col])

        return df.parallel_apply(subst, axis=1)

    @staticmethod
    def _parse(query_series):
        """
        Parse the SQL query to extract (prepared queries, parameters).

        Parameters
        ----------
        query_series : pd.Series
            SQL queries with the parameters inlined.

        Returns
        -------
        queries_and_params : pd.Series
            A series containing tuples of (prepared SQL query, parameters).
        """

        def parse(sql):
            new_sql, params, last_end = [], [], 0
            for token in pglast.parser.scan(sql):
                token_str = str(sql[token.start: token.end + 1])
                if token.start > last_end:
                    new_sql.append(" ")
                if token.name in ["ICONST", "FCONST", "SCONST"]:
                    # Integer, float, or string constant.
                    new_sql.append("$" + str(len(params) + 1))
                    params.append(token_str)
                else:
                    new_sql.append(token_str)
                last_end = token.end + 1
            new_sql = "".join(new_sql)
            return new_sql, tuple(params)

        return query_series.parallel_apply(parse)

    def _from_csvlogs(self, csvlogs, log_columns, store_query_subst=False):
        """
        Glue code for initializing the Preprocessor from CSVLOGs.

        Parameters
        ----------
        csvlogs : List[str]
            List of PostgreSQL CSVLOG files.
        log_columns : List[str]
            List of columns in the csv log.
        store_query_subst: bool
            True if the "query_subst" column should be stored.

        Returns
        -------
        df : pd.DataFrame
            A dataframe representing the query log.
        """
        time_end, time_start = None, time.perf_counter()

        def clock(label):
            nonlocal time_end, time_start
            time_end = time.perf_counter()
            print("\r{}: {:.2f} s".format(label, time_end - time_start))
            time_start = time_end

        df = self._read_df(csvlogs, log_columns)
        clock("Read dataframe")

        print("Extract queries: ", end="", flush=True)
        df["query_raw"] = self._extract_query(df["message"])
        df.drop(columns=["message"], inplace=True)
        clock("Extract queries")

        print("Extract parameters: ", end="", flush=True)
        df["params"] = self._extract_params(df["detail"])
        df.drop(columns=["detail"], inplace=True)
        clock("Extract parameters")

        print("Substitute parameters into query: ", end="", flush=True)
        df["query_subst"] = self._substitute_params(df, "query_raw", "params")
        df.drop(columns=["query_raw", "params"], inplace=True)
        clock("Substitute parameters into query")

        print("Parse query: ", end="", flush=True)
        parsed = self._parse(df["query_subst"])
        df[["query_template", "query_params"]] = pd.DataFrame(parsed.tolist(), index=df.index)
        clock("Parse query")

        # Only keep the relevant columns to optimize for storage, unless otherwise specified.
        stored_columns = ["log_time", "query_template", "query_params"]
        if store_query_subst:
            stored_columns.append("query_subst")
        return df[stored_columns]

    def __init__(self, parquet_path=None, csvlogs=None, log_columns=None, store_query_subst=False):
        """
        Initialize the preprocessor with either CSVLOGs or a Parquet dataframe.

        Parameters
        ----------
        parquet_path : str | None
            Path to a Parquet file containing a Preprocessor's get_dataframe().
            If specified, none of the other keyword arguments have any effect.

        csvlogs : List[str] | None
            List of PostgreSQL CSVLOG files.

        log_columns : List[str] | None
            List of columns for the PostgreSQL CSVLOG format.

        store_query_subst : bool
            True if the "query_subst" column should be stored.
            This stores an approximation of the "raw SQL query" used to generate the query template and parameters.
            This is not necessarily the raw SQL query itself since that may exist in various forms depending on the
            query protocol format.
        """
        if csvlogs is not None:
            df = self._from_csvlogs(csvlogs, log_columns, store_query_subst=store_query_subst)
            df.set_index("log_time", inplace=True)
        else:
            assert parquet_path is not None
            df = pd.read_parquet(parquet_path)
            # convert params from array back to tuple so it is hashable
            df["query_params"] = df["query_params"].map(lambda x: tuple(x))

        # Grouping queries by template-parameters count.
        gbp = df.groupby(["query_template", "query_params"]).size()
        grouped_by_params = pd.DataFrame(gbp, columns=["count"])
        # grouped_by_params.drop('', axis=0, level=0, inplace=True)
        # TODO(WAN): I am not sure if I'm wrong or pandas is wrong.
        #  Above raises ValueError: Must pass non-zero number of levels/codes.
        #  So we'll do this instead...
        grouped_by_params = grouped_by_params[~grouped_by_params.index.isin([("", ())])]
        self._df = df
        self._grouped_df_params = grouped_by_params


class PreprocessorCLI(cli.Application):
    # The columns that constitute a CSVLOG file, as defined by PostgreSQL.
    # See: https://www.postgresql.org/docs/14/runtime-config-logging.html
    _PG_LOG_COLUMNS: List[str] = [
        "log_time",
        "user_name",
        "database_name",
        "process_id",
        "connection_from",
        "session_id",
        "session_line_num",
        "command_tag",
        "session_start_time",
        "virtual_transaction_id",
        "transaction_id",
        "error_severity",
        "sql_state_code",
        "message",
        "detail",
        "hint",
        "internal_query",
        "internal_query_pos",
        "context",
        "query",
        "query_pos",
        "location",
        "application_name",
        "backend_type",
    ]

    query_log_folder = cli.SwitchAttr(
        "--query-log-folder", str, default=K.DEBUG_QB5000_CSV_FOLDER,
        help="The location containing postgresql*.csv query logs."
    )
    output_parquet = cli.SwitchAttr(
        "--output-parquet", str, default=K.DEBUG_QB5000_PREPROCESSOR,
        help="The location to write the output Parquet to."
    )
    output_timestamp = cli.SwitchAttr(
        "--output-timestamp",
        str,
        default=K.DEBUG_QB5000_PREPROCESSOR_TIMESTAMP,
        help="If specified, the min and max timestamps will be output to this file.",
    )
    output_query_templates = cli.SwitchAttr(
        "--output-query-templates", str, default=None, help="If specified, output location for SQL query templates."
    )
    output_queries = cli.SwitchAttr(
        "--output-queries", str, default=None, help="If specified, output location for SQL queries."
    )

    log_type = cli.SwitchAttr(
        "--log-type",
        argtype=str,
        default="pg14",
        help="Defines what the columns of the csvlog are. Can be the following options: {pg14, pg12, tiramisu}.",
    )

    def main(self):
        pgfiles = glob.glob(str(Path(self.query_log_folder) / "*.csv"))
        assert len(pgfiles) > 0, f"No PostgreSQL query log files found in: {self.query_log_folder}"

        # Resolve CSV log columns.
        assert self.log_type in [
            "pg14",
            "pg12",
            "tiramisu",
        ], f"--log-type {self.log_type} is invalid."

        log_columns = self._PG_LOG_COLUMNS
        if self.log_type == "pg12":
            log_columns = log_columns[:-1]
        elif self.log_type == "tiramisu":
            log_columns = [
                "log_time",
                "command_tag",
                "session_start_time",
                "message",
                "detail",
            ]

        print(f"Preprocessing CSV logs in: {self.query_log_folder}")
        preprocessor = Preprocessor(
            csvlogs=pgfiles, log_columns=log_columns, store_query_subst=self.output_queries is not None
        )
        print(f"Storing Parquet: {self.output_parquet}.")
        preprocessor.get_dataframe().to_parquet(self.output_parquet, compression="gzip")

        # Optionally write min and max timestamps of queries to infer forecast window.
        if self.output_timestamp is not None:
            with open(self.output_timestamp, "w") as ts_file:
                ts_file.write(preprocessor.get_dataframe().index.min().isoformat() + "\n")
                ts_file.write(preprocessor.get_dataframe().index.max().isoformat() + "\n")

        # Optionally write out the query templates and queries out to a file.
        if self.output_query_templates is not None:
            templates = preprocessor.get_dataframe()["query_template"]
            templates = pd.Series(templates[templates != ""].unique())
            templates.to_csv(self.output_query_templates, header=False, index=False, quoting=csv.QUOTE_ALL)
        if self.output_queries is not None:
            queries = preprocessor.get_dataframe()["query_subst"]
            queries = queries[queries != ""]
            queries.to_csv(self.output_queries, header=False, index=False, quoting=csv.QUOTE_ALL)


if __name__ == "__main__":
    PreprocessorCLI.run()
