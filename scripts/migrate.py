"""
Schema migration: add a 'source' column to the documents table so each chunk
knows which PDF it came from (enables per-document delete/refresh).
Safe to re-run. Run from project root:  python -m scripts.migrate
"""
from app.db import connect

conn = connect()
conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS source TEXT;")
conn.commit()

cols = conn.execute(
    "SELECT column_name, data_type FROM information_schema.columns "
    "WHERE table_name = 'documents';"
).fetchall()
print("documents columns:", cols)
conn.close()
