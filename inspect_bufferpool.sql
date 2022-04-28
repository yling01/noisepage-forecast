SELECT
    pg_namespace.nspname AS namespace_name,
    pg_class.relname AS relation_name,
    pg_table_size(pg_class.oid) AS relation_size_bytes,
    round(count(*) * 8192) AS num_bytes_in_buffer_pool,
    round(100.0 * count(*) / (SELECT setting FROM pg_settings WHERE name='shared_buffers')::integer, 2) AS percentage_of_buffer_pool,
    round(100.0 * count(*) * 8192 / pg_table_size(pg_class.oid), 2) AS percentage_of_relation
FROM pg_database, pg_buffercache, pg_class, pg_namespace
    WHERE pg_database.datname = current_database()
    AND pg_database.oid = pg_buffercache.reldatabase
    AND pg_buffercache.relfilenode = pg_class.relfilenode
    AND pg_namespace.oid = pg_class.relnamespace
GROUP BY pg_namespace.nspname, pg_class.relname, pg_class.oid
ORDER BY num_bytes_in_buffer_pool DESC
LIMIT 50;