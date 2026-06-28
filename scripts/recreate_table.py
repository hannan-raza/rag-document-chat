"""
(Re)create the documents table at the embedding dimension of the ACTIVE provider.
OpenAI text-embedding-3-small = 1536 dims; Bedrock Titan V2 = 1024 dims.
Switching providers requires recreating the table at the matching size, then
re-ingesting. WARNING: this drops all existing rows.
Run from project root:  python -m scripts.recreate_table
"""
from app.db import connect
from app.providers import DIMENSIONS, PROVIDER

conn = connect()
print(f"Provider: {PROVIDER} -> embedding dimension: {DIMENSIONS}")
conn.execute("DROP TABLE IF EXISTS documents;")
conn.execute(f"""
    CREATE TABLE documents (
        id SERIAL PRIMARY KEY,
        content TEXT,
        embedding VECTOR({DIMENSIONS}),
        source TEXT
    );
""")
conn.commit()

cols = conn.execute(
    "SELECT column_name, data_type FROM information_schema.columns "
    "WHERE table_name = 'documents';"
).fetchall()
print("Recreated. Columns:", cols)
conn.close()
