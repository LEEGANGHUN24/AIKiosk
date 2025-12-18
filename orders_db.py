# orders_db.py

import os
import sqlite3
import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "DATA", "orders.db")


def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    # 관리자 테이블
    cur.execute("""
    CREATE TABLE IF NOT EXISTS admins (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        login_id  TEXT NOT NULL UNIQUE,
        password  TEXT NOT NULL,
        name      TEXT NOT NULL
    )
    """)

    # 주문 테이블
    cur.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        order_no     TEXT NOT NULL,
        order_time   TEXT NOT NULL,
        mode         TEXT NOT NULL,
        menu_name    TEXT NOT NULL,
        menu_options TEXT,
        price        INTEGER NOT NULL,
        pay_method   TEXT NOT NULL
    )
    """)

    # ➕ ⭐ 기본 관리자 계정 자동 생성 (이미 있으면 추가하지 않음)
    cur.execute("SELECT COUNT(*) FROM admins")
    count = cur.fetchone()[0]

    if count == 0:
        cur.execute(
            "INSERT INTO admins (login_id, password, name) VALUES (?, ?, ?)",
            ("admin1", "1234", "관리자")
        )
        print("기본 관리자 계정 생성됨: admin1 / 1234 / 관리자")

    conn.commit()
    conn.close()


def create_admin(login_id: str, password: str, name: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO admins (login_id, password, name) VALUES (?, ?, ?)",
        (login_id, password, name)
    )
    conn.commit()
    conn.close()


def save_order(order_mode: str, pay_method: str, cart_items):
    from datetime import datetime

    if not cart_items:
        return

    conn = get_conn()
    cur = conn.cursor()

    order_no = datetime.now().strftime("%Y%m%d-%H%M%S")
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for item in cart_items:
        menu_name = item.get("menu_name", "")
        total_price = int(item.get("total_price", 0))

        parts = []
        for opt in item.get("options", []):
            n = opt.get("kor_name", "")
            cnt = int(opt.get("count", 0))
            parts.append(n if cnt == 1 else f"{n} x{cnt}")

        options_text = ", ".join(parts) if parts else ""

        cur.execute(
            """
            INSERT INTO orders(order_no, order_time, mode, menu_name, menu_options, price, pay_method)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (order_no, now_str, order_mode, menu_name, options_text, total_price, pay_method)
        )


    conn.commit()
    conn.close()
    
def check_admin(login_id: str, password: str) -> bool:
    """관리자 로그인 검증"""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM admins WHERE login_id=? AND password=?",
        (login_id, password)
    )
    row = cur.fetchone()
    conn.close()
    return row is not None

if __name__ == "__main__":
    print("DB 초기화 중... (DATA/orders.db 생성)")
    init_db()
    print("DB 생성 완료!")
