# check_orders.py

from orders_db import get_conn

conn = get_conn()
cur = conn.cursor()

cur.execute("SELECT id, order_no, order_time, mode, menu_name, menu_options, price, pay_method FROM orders ORDER BY id DESC")
rows = cur.fetchall()

for r in rows:
    print(r)

conn.close()
