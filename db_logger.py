import sqlite3
import csv
import os
from datetime import datetime


class SystemLogger:
    def __init__(self, csv_path, db_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        self.csv_path = csv_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                timestamp TEXT,
                event TEXT
            )
        """)
        self.conn.commit()

    def log(self, event):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, event])

        self.cursor.execute("INSERT INTO logs VALUES (?, ?)", (timestamp, event))
        self.conn.commit()