FORECAST_FIFO = "forecast.fifo"

# QB5000 directories
DEBUG_QB5000_ROOT = "../artifacts/QB5000_forecast"
DEBUG_QB5000_QUERY_LOG_CSV_FOLDER = "../artifacts/train/"
DEBUG_QB5000_PREPROCESSOR_OUTPUT = "/".join((DEBUG_QB5000_ROOT, "out_preprocessor.parquet"))
DEBUG_QB5000_PREPROCESSOR_TIMESTAMP = "/".join((DEBUG_QB5000_ROOT, "out_clusterer_timestamp.txt"))
DEBUG_QB5000_QUERY_TEMPLATES_CSV = "/".join((DEBUG_QB5000_ROOT, "out_query_templates.csv"))
DEBUG_QB5000_SQL_QUERY_CSV = "/".join((DEBUG_QB5000_ROOT, "out_sql_query.csv"))
DEBUG_QB5000_CLUSTERER_OUTPUT = "/".join((DEBUG_QB5000_ROOT, "out_clusterer.parquet"))
DEBUG_QB5000_MODEL_DIR = "/".join((DEBUG_QB5000_ROOT, "model"))
DEBUG_QB5000_FORECASTER_PREDICTION_CSV = "/".join((DEBUG_QB5000_ROOT, "out_forecaster.csv"))
DEBUG_QB5000_INSERT_DELETE_CSV = "/".join((DEBUG_QB5000_ROOT, "out_insert_delete.csv"))

# SQL statement
GET_TABLE_NAMES = "SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname != 'pg_catalog' AND schemaname != 'information_schema'"

# mysql_to_postgresql.
DEBUG_MYSQL_LOG = "/home/kapi/admissions/magneto.log.2016-09-04"
DEBUG_MYSQL_CSV = "/tmp/meowmeow.csv"
DEBUG_POSTGRESQL_CSV = "/tmp/meowmeow2.csv"

# postgresql_to_split_postgresql.
DEBUG_POSTGRESQL_CSV_FOLDER = "./artifacts/train/"
DEBUG_SPLIT_POSTGRESQL_FOLDER = "./artifacts/tmp/meowsplit/"
DEBUG_POSTGRESQL_PARQUET_FOLDER = "./artifacts/tmp/meowquet/"
DEBUG_POSTGRESQL_PARQUET_DATA = "./artifacts/tmp/data/"
DEBUG_POSTGRESQL_PARQUET_TRAIN = DEBUG_POSTGRESQL_PARQUET_DATA + "train.parquet"
DEBUG_POSTGRESQL_PARQUET_FUTURE = DEBUG_POSTGRESQL_PARQUET_DATA + "future.parquet"
DEBUG_POSTGRESQL_CSV_TRAIN = DEBUG_POSTGRESQL_PARQUET_DATA + "train.csv"
DEBUG_POSTGRESQL_CSV_FUTURE = DEBUG_POSTGRESQL_PARQUET_DATA + "future.csv"

DEBUG_FORECAST_FOLDER = "./artifacts/tmp/forecasted/"

DEFAULT_DB = "forecast_db"
DEFAULT_USER = "forecast_user"
DEFAULT_PASS = "forecast_pass"
DB_CONN_STRING = f"host=127.0.0.1 port=5432 dbname={DEFAULT_DB} user={DEFAULT_USER} password={DEFAULT_PASS} sslmode=disable application_name=psql"

# Database dump directory
DEBUG_DB_DUMP_DIR = "../artifacts/dump"
DEBUG_RESTORE_DB_COMMAND = f"PGPASSWORD={DEFAULT_PASS} pg_restore --host=localhost --username={DEFAULT_USER} --clean --if-exists --dbname={DEFAULT_DB} {DEBUG_DB_DUMP_DIR}"

PG_LOG_DTYPES = {
    "log_time": str,
    "user_name": str,
    "database_name": str,
    "process_id": "Int64",
    "connection_from": str,
    "session_id": str,
    "session_line_num": "Int64",
    "command_tag": str,
    "session_start_time": str,
    "virtual_transaction_id": str,
    "transaction_id": "Int64",
    "error_severity": str,
    "sql_state_code": str,
    "message": str,
    "detail": str,
    "hint": str,
    "internal_query": str,
    "internal_query_pos": "Int64",
    "context": str,
    "query": str,
    "query_pos": "Int64",
    "location": str,
    "application_name": str,
    # PostgreSQL 13+.
    "backend_type": str,
    # PostgreSQL 14+.
    "leader_pid": "Int64",
    "query_id": "Int64",
}
