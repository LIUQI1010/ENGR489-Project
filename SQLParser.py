import sqlparse
from sqlparse.sql import IdentifierList, Identifier
from sqlparse.tokens import Keyword, DML

def is_subselect(parsed):
    if not parsed.is_group:
        return False
    for item in parsed.tokens:
        if item.ttype is DML and item.value.upper() == 'SELECT':
            return True
    return False

def extract_from_part(parsed):
    from_seen = False
    for item in parsed.tokens:
        if from_seen:
            if is_subselect(item):
                for x in extract_from_part(item):
                    yield x
            elif item.ttype is Keyword:
                return
            else:
                yield item
        elif item.ttype is Keyword and item.value.upper() == 'FROM':
            from_seen = True

def extract_table_identifiers(token_stream):
    for item in token_stream:
        if isinstance(item, IdentifierList):
            for identifier in item.get_identifiers():
                alias = identifier.get_alias()
                yield alias if alias else identifier.get_real_name()
        elif isinstance(item, Identifier):
            alias = item.get_alias()
            yield alias if alias else item.get_real_name()
        elif item.ttype is Keyword:
            yield item.value

def parse_sql(file_path):
    with open(file_path, 'r') as file:
        sql_query = file.read()

    parsed = sqlparse.parse(sql_query)[0]
    from_clause = list(extract_from_part(parsed))
    tables = list(extract_table_identifiers(from_clause))

    query_dict = {
        "SELECT": [],
        "FROM": tables,
        "WHERE": [],
        "JOIN": []
    }

    # Extract SELECT part
    select_seen = False
    for token in parsed.tokens:
        if token.ttype is DML and token.value.upper() == 'SELECT':
            select_seen = True
        elif select_seen:
            if token.ttype is Keyword and token.value.upper() == 'FROM':
                break
            if token.ttype is not None and token.ttype is not Keyword:
                query_dict["SELECT"].append(token.value.strip())

    # Extract WHERE part
    where_seen = False
    for token in parsed.tokens:
        if token.ttype is Keyword and token.value.upper() == 'WHERE':
            where_seen = True
        elif where_seen:
            if token.ttype is Keyword and token.value.upper() in ('GROUP', 'ORDER', 'LIMIT'):
                break
            if token.ttype is not None:
                query_dict["WHERE"].append(token.value.strip())

    # Extract JOIN part
    join_seen = False
    for token in parsed.tokens:
        if token.ttype is Keyword and 'JOIN' in token.value.upper():
            join_seen = True
            query_dict["JOIN"].append(token.value.strip())
        elif join_seen:
            if token.ttype is Keyword and token.value.upper() in ('ON', 'USING'):
                join_seen = False
                query_dict["JOIN"][-1] += ' ' + token.value.strip()
            elif token.ttype is not None:
                query_dict["JOIN"][-1] += ' ' + token.value.strip()

    return query_dict
