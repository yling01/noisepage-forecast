import pandas as pd
import psycopg
import constants as K
from preprocessor import Preprocessor
import collections
from sql_metadata import Parser
from tqdm import tqdm
import os

if __name__ == "__main__":
    # always restore the forecast_db from dump
    os.system(K.DEBUG_RESTORE_DB_COMMAND)

    # obtain the preprocessor output containing the query templates and the actual queries
    preprocessor = Preprocessor(parquet_path=K.DEBUG_QB5000_PREPROCESSOR_OUTPUT)
    df = preprocessor.get_dataframe().drop(columns=["query_params"])
    df_insert_delete = df[df["query_template"].str.contains("^(?:UPDATE|INSERT|DELETE)+")]

    # get all tables
    conn = psycopg.connect(K.DB_CONN_STRING)
    cur = conn.cursor()

    cur.execute(K.GET_TABLE_NAMES)
    tables = cur.fetchall()

    tables = list(map(lambda x: x[0], tables))

    columns = ['count'] + tables

    # delta dataframe keeps a record of how many tuples are inserted/deleted in each table for all templates (avg)
    delta_df = pd.DataFrame(columns=columns)

    insert_statement_ignored = delete_statement_ignored = 0
    batch_bar = tqdm(total=len(df_insert_delete), dynamic_ncols=True, leave=False, position=0, desc='Query replay')

    for index, row in df_insert_delete.iterrows():
        template, query = row["query_template"], row["query_subst"]
        parsed_query = Parser(query)
        tables_referenced = parsed_query.tables
        query_type = str(parsed_query.query_type).split(".")[-1]

        before = collections.defaultdict(int)
        after = collections.defaultdict(int)

        # if insert or delete query, get the change in numer of tuples for all tables
        if query_type == "DELETE" or query_type == "INSERT":
            if template not in delta_df.index:
                delta_df.loc[template] = dict(zip(columns, [0 for _ in columns]))

            delta_df.loc[template]['count'] += 1

            for table in tables_referenced:
                cur.execute(f"SELECT count(*) FROM {table}")
                before[table] = cur.fetchall()[0][0]

        try:
            cur.execute(query)
            conn.commit()
        except Exception as e:
            if query_type == "DELETE":
                delete_statement_ignored += 1
            elif query_type == "INSERT":
                insert_statement_ignored += 1
            # note: this might be problematic, if an earlier insertion corresponds to a later deletion, it will fail
            # todo: should check why some queries fail (all due to primary key constraint violation)
            conn.rollback()
            delta_df.loc[template]['count'] -= 1
            continue

        if query_type == "DELETE" or query_type == "INSERT":
            for table in tables_referenced:
                cur.execute(f"SELECT count(*) FROM {table}")
                after[table] = cur.fetchall()[0][0]

                delta_df.loc[template][table] += after[table] - before[table]

        batch_bar.update()
    batch_bar.close()
    delta_df.to_csv(K.DEBUG_QB5000_INSERT_DELETE_CSV)

    print("Finished replaying insert, delete, and update queries.")
    print(f"{delete_statement_ignored} delete statements and {insert_statement_ignored} insert statements are ignored.")

    conn.close()

    # restore forecast_db at the end
    os.system(K.DEBUG_RESTORE_DB_COMMAND)
