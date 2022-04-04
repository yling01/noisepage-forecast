import csv
import sys
from collections import namedtuple
from pathlib import Path

from constants import (DEBUG_POSTGRESQL_CSV, DEBUG_SPLIT_POSTGRESQL_FOLDER,
                       PG_LOG_DTYPES)

LogEntry = namedtuple("LogEntry", PG_LOG_DTYPES.keys())


def _write_lines(buffer, filename):
    out_folder = Path(DEBUG_SPLIT_POSTGRESQL_FOLDER)
    out_folder.mkdir(exist_ok=True)
    out_file = out_folder / filename
    with open(out_file, "w", newline="") as out_csv:
        writer = csv.writer(out_csv, lineterminator="\n", quoting=csv.QUOTE_ALL)
        writer.writerows(buffer)


def main():
    # Expand field size because of massive queries.
    csv.field_size_limit(sys.maxsize)

    source_csvlog = Path(DEBUG_POSTGRESQL_CSV)
    min_batch_size = 500000
    active_sessions = set()
    buffer = []
    split_num = 0

    def _write_batch():
        nonlocal active_sessions, buffer, source_csvlog, split_num
        filename = f"postgres_{split_num}.csv" if len(active_sessions) == 0 else f"incomplete_{source_csvlog.stem}.csv"
        _write_lines(buffer, filename)
        buffer = []
        split_num += 1

    with open(source_csvlog, "r", newline="") as f:
        reader = csv.reader(f, quoting=csv.QUOTE_ALL)
        for row in reader:
            log_entry = LogEntry(*row)
            session_id = log_entry.session_id
            message = log_entry.message
            buffer.append(row)

            is_begin = message.startswith("connection received: ") and not message.startswith(
                "connection received: Access denied")
            is_end = any(
                message.startswith(prefix) for prefix in ["disconnection: ", "connection received: Access denied"])

            if is_begin:
                assert session_id not in active_sessions, f"ERROR: multiple connection received events for session {session_id}?"
                active_sessions.add(session_id)
            elif is_end:
                if session_id not in active_sessions:
                    # This may happen because we are given dirty data.
                    # We can't do much about that, so we can't assert.
                    print(f"WARNING: deactivating non-existent session {session_id}.")
                else:
                    active_sessions.remove(session_id)

            if len(active_sessions) == 0 and len(buffer) >= min_batch_size:
                _write_batch()
    _write_batch()


if __name__ == "__main__":
    # TODO(WAN): Take the incomplete of the previous, prepend it to the next log file ingested.
    main()
