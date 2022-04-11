import pglast

def substitute(query, params, onerror="raise"):
    # Consider '$2' -> "abc'def'ghi".
    # This necessitates the use of a SQL-aware substitution,
    # even if this is much slower than naive string substitution.
    new_sql, last_end = [], 0
    try:
        tokens = pglast.parser.scan(query)
    except pglast.parser.ParseError as exc:
        message = f"Bad query: {query}"
        if onerror != "ignore":
            raise ValueError(message)
        print(message)
        return ""
    for token in tokens:
        token_str = str(query[token.start: token.end + 1])
        if token.start > last_end:
            new_sql.append(" ")
        if token.name == "PARAM":
            assert token_str.startswith("$")
            assert token_str[1:].isdigit()
            if token_str not in params:
                message = f"Bad query param: {token_str} {query} {params}"
                if onerror != "ignore":
                    raise ValueError(message)
                print(message)
                return ""
            new_sql.append(params[token_str])
        else:
            new_sql.append(token_str)
        last_end = token.end + 1
    new_sql = "".join(new_sql)
    return new_sql