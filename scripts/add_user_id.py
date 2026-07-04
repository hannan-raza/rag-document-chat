"""Add user_id column to documents for per-user ownership.
Wipes existing test rows (pre-production, no real owners)."""
from app.db import connect

conn = connect()
conn.execute("TRUNCATE TABLE documents;")  # test data, no real owners
conn.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS user_id TEXT;")
conn.commit()
conn.close()
print("Migration done: documents truncated + user_id column added.")
