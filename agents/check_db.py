import sqlite3
import sys

db_path = "checkpoints.db"
try:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cur.fetchall()
    print("✅ 数据库正常，包含表：", [t[0] for t in tables])
except sqlite3.DatabaseError as e:
    print("❌ 数据库损坏：", e)
    print("建议：删除 checkpoints.db 后重新运行主程序")