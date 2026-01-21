import sqlite3
import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

conn = sqlite3.connect("songs.db")
df = pd.read_sql("SELECT * FROM songs", conn)
conn.close()

print(df)


