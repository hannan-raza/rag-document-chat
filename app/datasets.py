"""
CSV dataset ingestion for Module 2 (chat-with-your-data / text-to-SQL).
Each uploaded CSV becomes its own real Postgres table; the `datasets`
table keeps one metadata row per file (owner, filename, table name, schema).
"""
import io
import re
import uuid
import json
import pandas as pd

from app.db import connect


def make_table_name(user_id, source):
    """A safe, unique table name for one user's uploaded file.
    We don't trust the filename (spaces / punctuation / SQL-unsafe chars),
    so the real identity is a random suffix; the readable filename lives in `source`.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", source.lower()).strip("_")[:20] or "data"
    return f"ds_{slug}_{uuid.uuid4().hex[:8]}"


def infer_pg_type(dtype):
    """Map a pandas column dtype to a Postgres column type.
    Unknown / mixed columns fall back to TEXT, which always works.
    """
    name = str(dtype)
    if name.startswith("int"):
        return "BIGINT"
    if name.startswith("float"):
        return "NUMERIC"
    if name.startswith("bool"):
        return "BOOLEAN"
    if name.startswith("datetime"):
        return "TIMESTAMP"
    return "TEXT"


def safe_col(name):
    """Sanitize a CSV header into a safe SQL column name."""
    clean = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    return clean or "col"


def create_dataset(csv_bytes, source, user_id):
    """Parse a CSV, create its own table, load the rows, and register it.
    Returns a small summary dict.
    """
    # 1. parse
    df = pd.read_csv(io.BytesIO(csv_bytes))
    if df.empty:
        raise ValueError("CSV has no rows.")

    # 2. sanitize column names (dedupe collisions after sanitizing)
    seen, cols = {}, []
    for original in df.columns:
        base = safe_col(original)
        if base in seen:
            seen[base] += 1
            base = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
        cols.append(base)
    df.columns = cols

    # 3. work out the schema (column name + Postgres type)
    schema = [{"name": c, "type": infer_pg_type(df[c].dtype)} for c in df.columns]

    table_name = make_table_name(user_id, source)

    conn = connect()
    conn.autocommit = False  # all-or-nothing: table + rows + registry commit together
    try:
        # 4. create the dataset's own table
        col_defs = ", ".join(f'"{c["name"]}" {c["type"]}' for c in schema)
        conn.execute(f'CREATE TABLE "{table_name}" ({col_defs});')

        # 5. insert the rows
        placeholders = ", ".join(["%s"] * len(df.columns))
        col_list = ", ".join(f'"{c}"' for c in df.columns)
        insert_sql = f'INSERT INTO "{table_name}" ({col_list}) VALUES ({placeholders})'
        # NaN -> None so empty cells become SQL NULL
        rows = [tuple(None if pd.isna(v) else v for v in row)
                for row in df.itertuples(index=False, name=None)]
        conn.cursor().executemany(insert_sql, rows)

        # 6. register the dataset (one metadata row)
        conn.execute(
            "INSERT INTO datasets (user_id, source, table_name, columns, row_count) "
            "VALUES (%s, %s, %s, %s, %s)",
            (user_id, source, table_name, json.dumps(schema), len(df)),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        # best-effort cleanup of a half-created table
        try:
            conn.execute(f'DROP TABLE IF EXISTS "{table_name}";')
            conn.commit()
        except Exception:
            pass
        raise
    finally:
        conn.close()

    return {"source": source, "table": table_name, "rows": len(df), "columns": schema}# ---------------------------------------------------------------------------
# Text-to-SQL query path
# ---------------------------------------------------------------------------
from app.providers import generate


def get_dataset(user_id, dataset_id):
    """Look up one dataset the user owns. Returns (table_name, schema) or None.
    Scoped by user_id so a user can only ever reach their own tables.
    """
    conn = connect()
    conn.autocommit = True
    row = conn.execute(
        "SELECT table_name, columns FROM datasets WHERE id = %s AND user_id = %s",
        (dataset_id, user_id),
    ).fetchone()
    conn.close()
    if not row:
        return None
    return row[0], row[1]  # table_name, schema (JSONB -> list already)


def list_datasets_meta(user_id):
    """Lightweight catalog for the dataset-picker: id, filename, column names only.
    Scoped by user_id. Reads the `datasets` registry ONLY — never touches the
    ds_ data tables, so this stays cheap regardless of how big the CSVs are.
    """
    conn = connect()
    conn.autocommit = True
    try:
        rows = conn.execute(
            "SELECT id, source, columns FROM datasets WHERE user_id = %s ORDER BY id",
            (user_id,),
        ).fetchall()
    finally:
        conn.close()
    # columns is JSONB -> already a list of {name, type}; expose just the names
    return [{"id": r[0], "source": r[1], "columns": [c["name"] for c in r[2]]}
            for r in rows]


def pick_datasets(user_id, question):
    """Choose which of the user's datasets (if any) are relevant to a question.

    This is the "schema linking" / table-retrieval step for the unified query:
    a single cheap LLM call sees only metadata (filename + column names, NEVER
    the row data) and returns the dataset ids worth generating SQL against.

    Returns a list of dataset ids (possibly empty). Isolation + safety:
    - only datasets this user owns are ever considered (registry is user-scoped);
    - the LLM reply is parsed defensively — non-numbers and any id the user
      doesn't own are dropped, order preserved, duplicates removed.
    """
    meta = list_datasets_meta(user_id)
    if not meta:
        return []

    catalog = "\n".join(
        f'[{m["id"]}] {m["source"]} — columns: {", ".join(m["columns"])}'
        for m in meta
    )
    valid_ids = {m["id"] for m in meta}

    prompt = (
        "You are routing a user's question to the right data tables.\n"
        "Below is a catalog of the user's datasets (id, filename, and column "
        "names only). The question may be compound and mix data and non-data "
        "parts. Select a dataset if it could help answer the question OR ANY "
        "PART OF IT — even if other parts need a different source. Ignore parts "
        "that no dataset can answer.\n"
        "Reply with ONLY the matching ids, comma-separated (e.g. 1,3). "
        "If none of the datasets are relevant to any part, reply with the "
        "single word: none.\n"
        "Do not explain.\n\n"
        f"Datasets:\n{catalog}\n\n"
        f"Question: {question}\n\n"
        "Relevant ids:"
    )
    reply = generate(prompt, max_tokens=50, temperature=0).strip()

    picked = []
    for tok in reply.replace(" ", "").split(","):
        if tok.isdigit():
            ds_id = int(tok)
            if ds_id in valid_ids and ds_id not in picked:
                picked.append(ds_id)
    return picked


def _sample_rows(table_name, limit=3):
    """A few example rows so the LLM sees real values (helps it write correct SQL)."""
    conn = connect()
    conn.autocommit = True
    try:
        cur = conn.execute(f'SELECT * FROM "{table_name}" LIMIT %s', (limit,))
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    finally:
        conn.close()
    return cols, rows


def _schema_text(table_name, schema):
    """Build the schema description we hand to the LLM: columns+types plus samples."""
    lines = [f"Table name: {table_name}", "Columns:"]
    for c in schema:
        lines.append(f'  - "{c["name"]}" ({c["type"]})')
    cols, rows = _sample_rows(table_name)
    if rows:
        lines.append("\nSample rows:")
        lines.append("  " + " | ".join(cols))
        for r in rows:
            lines.append("  " + " | ".join("" if v is None else str(v) for v in r))
    return "\n".join(lines)


# only allow a single read-only SELECT statement through to the database
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|"
    r"copy|attach|vacuum)\b",
    re.IGNORECASE,
)


def _is_safe_select(sql):
    """True only if sql is a single SELECT with no mutating keywords or stacked statements."""
    s = sql.strip().rstrip(";").strip()
    if not s.lower().startswith("select"):
        return False
    if ";" in s:                 # block stacked statements (e.g. "select ..; drop ..")
        return False
    if _FORBIDDEN.search(s):
        return False
    return True


def generate_sql(question, table_name, schema):
    """Ask the LLM to turn the question into a single SELECT against this table."""
    schema_desc = _schema_text(table_name, schema)
    prompt = (
        "You are a Postgres expert. Given the table schema below, write ONE SQL "
        "query that answers the user's question.\n"
        "Rules:\n"
        f'- Query ONLY the table "{table_name}".\n'
        "- Use a single SELECT statement. No INSERT/UPDATE/DELETE/DDL.\n"
        "- Return ONLY the SQL, no explanation, no markdown fences.\n\n"
        f"{schema_desc}\n\n"
        f"Question: {question}\n\n"
        "SQL:"
    )
    sql = generate(prompt, max_tokens=300).strip()
    # strip accidental ```sql fences if the model adds them
    sql = re.sub(r"^```(?:sql)?|```$", "", sql, flags=re.IGNORECASE).strip()
    return sql


def run_select(sql, limit=100):
    """Execute a validated SELECT and return (columns, rows), capped."""
    conn = connect()
    conn.autocommit = True
    try:
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchmany(limit)
    finally:
        conn.close()
    return cols, rows


def phrase_answer(question, cols, rows):
    """Turn raw query results into a natural-language answer grounded in them."""
    preview = [dict(zip(cols, r)) for r in rows[:20]]
    prompt = (
        "Answer the user's question in one or two sentences, using ONLY the query "
        "results provided. Be direct and specific.\n\n"
        f"Question: {question}\n\n"
        f"Query results: {json.dumps(preview, default=str)}\n\n"
        "Answer:"
    )
    return generate(prompt, max_tokens=250).strip()


def run_dataset_query(user_id, dataset_id, question):
    """Resolve the user's dataset, generate SQL, enforce ALL Module 2 safety
    guards, then execute. Returns raw results (no natural-language phrasing):
    {dataset_id, table_name, sql, columns, rows}. Raises ValueError if the
    dataset isn't the user's or the generated SQL fails a guard.

    This is the SINGLE home for the generated-SQL guards (CLAUDE.md §6) — both
    the single-dataset endpoint (query_dataset) and the unified orchestrator go
    through here, so the guards can't drift out of sync.
    """
    found = get_dataset(user_id, dataset_id)
    if not found:
        raise ValueError("Dataset not found.")
    table_name, schema = found

    sql = generate_sql(question, table_name, schema)

    if not _is_safe_select(sql):
        raise ValueError("Generated query was not a safe read-only SELECT.")

    # defense in depth: the LLM must target only this user's table
    if table_name not in sql:
        raise ValueError("Generated query did not target the expected table.")

    cols, rows = run_select(sql)
    return {
        "dataset_id": dataset_id,
        "table_name": table_name,
        "sql": sql,
        "columns": cols,
        "rows": rows,
    }


def query_dataset(user_id, dataset_id, question):
    """Full text-to-SQL flow for one question against one dataset (with phrasing)."""
    result = run_dataset_query(user_id, dataset_id, question)
    answer_text = phrase_answer(question, result["columns"], result["rows"])
    return {
        "question": question,
        "answer": answer_text,
        "sql": result["sql"],
        "columns": result["columns"],
        "rows": [list(r) for r in result["rows"]],
    }