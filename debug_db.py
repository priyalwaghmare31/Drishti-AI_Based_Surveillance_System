import sqlite3, json, os
DB = os.path.join(os.path.dirname('.'), 'surveillance.db')   # or data.db / whatever file you use
print("DB file:", DB)
conn = sqlite3.connect(DB)
cur = conn.cursor()

# list tables
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("Tables:", cur.fetchall())

# show columns of people table (since it's 'people', not 'persons')
try:
    cur.execute("PRAGMA table_info(people)")
    print("people columns:", cur.fetchall())
except Exception as e:
    print("PRAGMA error:", e)

# dump first 5 rows
try:
    cur.execute("SELECT face_id, name, email, embeddings FROM people LIMIT 5")
    rows = cur.fetchall()
    print("First rows count:", len(rows))
    for r in rows:
        print("ROW face_id,name,email:", r[0], r[1], r[2])
        print("embeddings type/len/preview:", type(r[3]), (len(r[3]) if r[3] else 0), (r[3][:80] if r[3] else r[3]))
except Exception as e:
    print("SELECT error:", e)

conn.close()
