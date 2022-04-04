import re
from pathlib import Path

import numpy as np
import pandas as pd
import pglast
from tqdm.contrib.concurrent import process_map

from constants import (DEBUG_POSTGRESQL_PARQUET_FOLDER,
                       DEBUG_SPLIT_POSTGRESQL_FOLDER, PG_LOG_DTYPES)


def convert_postgresql_csv_to_parquet(pg_csvlog, pq_path):
    df = pd.read_csv(
        pg_csvlog,
        names=PG_LOG_DTYPES.keys(),
        parse_dates=["log_time", "session_start_time"],
        usecols=[
            "log_time",
            "session_id",
            "session_line_num",
            "session_start_time",
            "command_tag",
            "message",
            "detail",
            "virtual_transaction_id",
            "transaction_id",
        ],
        dtype=PG_LOG_DTYPES,
        header=None,
    )

    tokens = "DELETE|INSERT|SELECT|UPDATE|BEGIN|COMMIT|ROLLBACK|SHOW"
    simple = f"statement: ((?:{tokens}).*)"
    extended = f"execute .+: ((?:{tokens}).*)"
    regex = f"(?:{simple})|(?:{extended})"

    query = df["message"].str.extract(regex, flags=re.IGNORECASE)
    # Combine the capture groups for simple and extended query protocol.
    query = query[0].fillna(query[1])
    # print("TODO(WAN): Disabled SQL format for being too slow.")
    # Prettify each SQL query for standardized formatting.
    # query = query.parallel_map(pglast.prettify, na_action='ignore')
    df["query_raw"] = query
    df["params"] = df["detail"].apply(_extract_params)

    df["query_subst"] = df[["query_raw", "params"]].apply(_substitute, axis=1)
    df = df.drop(columns=["query_raw", "params"])

    template_param = df["query_subst"].apply(_parse)
    df = df.assign(
        query_template=template_param.map(lambda x: x[0]),
        query_params=template_param.map(lambda x: x[1]),
    )
    df["query_template"] = df["query_template"].astype("category")

    stored_columns = {
        "log_time",
        "query_template",
        "query_params",
        "session_id",
        "session_line_num",
        "virtual_transaction_id",
        "transaction_id",
    }

    df = df.drop(columns=set(df.columns) - stored_columns)
    df = df.set_index("log_time")
    df.to_parquet(pq_path)


def _extract_params(detail):
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


def _substitute(row):
    query, params = row["query_raw"], row["params"]
    if query is pd.NA or query is np.nan:
        return pd.NA
    query = str(query)
    # Consider '$2' -> "abc'def'ghi".
    # This necessitates the use of a SQL-aware substitution,
    # even if this is much slower than naive string substitution.
    new_sql, last_end = [], 0
    try:
        tokens = pglast.parser.scan(query)
    except pglast.parser.ParseError:
        print(f"Bad query: {query}")
        return ""
    for token in tokens:
        token_str = str(query[token.start: token.end + 1])
        if token.start > last_end:
            new_sql.append(" ")
        if token.name == "PARAM":
            assert token_str.startswith("$")
            assert token_str[1:].isdigit()
            if token_str not in params:
                print(f"Bad query param: {token_str} {query} {params}")
                return ""
            new_sql.append(params[token_str])
        else:
            new_sql.append(token_str)
        last_end = token.end + 1
    new_sql = "".join(new_sql)
    return new_sql


def _parse(sql):
    if sql is pd.NA or sql is np.nan:
        return "", ()
    sql = str(sql)
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


def _convert(csvlog):
    pq_path = Path(DEBUG_POSTGRESQL_PARQUET_FOLDER)
    pq_path.mkdir(exist_ok=True)
    pq_name = csvlog.name + ".parquet"
    convert_postgresql_csv_to_parquet(csvlog, pq_path / pq_name)


def main():
    csvlogs = list(Path(DEBUG_SPLIT_POSTGRESQL_FOLDER).glob("postgres_*.csv"))
    process_map(_convert, csvlogs)


if __name__ == "__main__":
    main()
