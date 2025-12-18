# reset_orders_table.py

import os
import sqlite3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "DATA", "orders.db")

def reset_orders_table():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # ✅ orders 테이블의 모든 행 삭제 (테이블 구조는 유지)
    cur.execute("DELETE FROM orders")

    # AUTOINCREMENT id 값을 1로 초기화
    cur.execute("DELETE FROM sqlite_sequence WHERE name='orders'")

    conn.commit()
    conn.close()
    print("orders 테이블의 모든 데이터를 삭제하고 초기화했습니다.")

if __name__ == "__main__":
    reset_orders_table()
