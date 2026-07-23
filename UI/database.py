import sqlite3
from datetime import datetime

DB_NAME = "food.db"

def create_database():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # 주문 정보
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS orders(
        order_id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_time TEXT,
        total_price INTEGER
    )
    """)

    # 주문 상세
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS order_items(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_id INTEGER,
        menu_name TEXT,
        quantity INTEGER,
        price INTEGER
    )
    """)

    # 재고 테이블
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS inventory(
        menu_name TEXT PRIMARY KEY,
        stock INTEGER,
        max_stock INTEGER
    )
    """)
    default_items = [
        ("볶음밥", 30, 30),
        ("파전", 30, 30),
        ("막걸리", 50, 50)
    ]

    cursor.executemany("""
    INSERT OR IGNORE INTO inventory(menu_name, stock, max_stock)
    VALUES (?, ?, ?)
    """, default_items)

    conn.commit()
    conn.close()

def save_order(data):

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 주문 저장
    cursor.execute("""
        INSERT INTO orders(order_time, total_price)
        VALUES (?, ?)
    """, (now, data["totalPrice"]))

    # 방금 생성된 주문번호
    order_id = cursor.lastrowid

    # 주문한 메뉴 저장
    for item in data["items"]:
        cursor.execute("""
            INSERT INTO order_items(order_id, menu_name, quantity, price)
            VALUES (?, ?, ?, ?)
        """, (
            order_id,
            item["name"],
            item["qty"],
            item["price"]
        ))
        # 재고 차감
        cursor.execute("""
            UPDATE inventory
            SET stock = stock - ?
            WHERE menu_name = ?
        """, (
            item["qty"],
            item["name"]
        ))

        # 디버깅 출력
        print("메뉴:", repr(item["name"]))
        print("수정된 행 수:", cursor.rowcount)

    print(f"주문번호 {order_id} 저장 완료")

    for item in data["items"]:
        print(item)

    conn.commit()
    conn.close()

    return order_id

def show_orders():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM orders")

    rows = cursor.fetchall()

    conn.close()

    print("\n===== 주문 내역 =====")

    for row in rows:
        print(row)

def get_orders():

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT order_id, order_time, total_price
        FROM orders
        ORDER BY order_id DESC
    """)

    orders = cursor.fetchall()

    result = []

    for order in orders:

        order_id = order[0]

        cursor.execute("""
            SELECT menu_name, quantity, price
            FROM order_items
            WHERE order_id=?
        """, (order_id,))

        items = cursor.fetchall()

        result.append({
            "order_id": order_id,
            "order_time": order[1],
            "total_price": order[2],
            "order_items": items
        })

    conn.close()

    return result

def get_today_sales():

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    today = datetime.now().strftime("%Y-%m-%d")

    cursor.execute("""
        SELECT SUM(total_price)
        FROM orders
        WHERE DATE(order_time) = ?
    """, (today,))

    total = cursor.fetchone()[0]

    conn.close()

    if total is None:
        return 0

    return total

def get_inventory():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT menu_name, stock, max_stock
        FROM inventory
    """)

    inventory = cursor.fetchall()

    conn.close()

    return inventory

def update_inventory(menu_name, stock, max_stock):

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE inventory
        SET stock=?,
            max_stock=?
        WHERE menu_name=?
    """, (
        stock,
        max_stock,
        menu_name
    ))

    conn.commit()
    conn.close()

def check_stock(data):

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    for item in data["items"]:

        cursor.execute("""
            SELECT stock
            FROM inventory
            WHERE menu_name=?
        """, (item["name"],))

        stock = cursor.fetchone()[0]

        if stock < item["qty"]:
            conn.close()

            return False, f"{item['name']}의 재고가 부족합니다."

    conn.close()

    return True, ""