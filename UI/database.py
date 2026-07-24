import sqlite3
from datetime import datetime

DB_NAME = "food.db"

# 메뉴별 소모 식자재 매핑 (이름, 단위당 소모량)
RECIPES = {
    "볶음밥": [("팥", 100), ("콩", 100), ("쌀", 100)],
    "부침개": [("부침개 반죽", 100)],
    "막걸리": [("막걸리", 1)]
}

def create_database():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # 주문 정보
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS orders(
        order_id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_time TEXT,
        total_price INTEGER,
        status TEXT DEFAULT 'waiting'
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

    # 식자재 재고 테이블
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ingredients(
        ingredient_name TEXT PRIMARY KEY,
        stock INTEGER,
        unit TEXT
    )
    """)
    
    # 기본 식자재 세팅 (팥, 콩, 쌀, 반죽은 10kg=10000g, 막걸리는 100병)
    default_ingredients = [
        ("팥", 10000, "g"),
        ("콩", 10000, "g"),
        ("쌀", 10000, "g"),
        ("부침개 반죽", 10000, "g"),
        ("막걸리", 100, "병")
    ]

    cursor.executemany("""
    INSERT OR IGNORE INTO ingredients(ingredient_name, stock, unit)
    VALUES (?, ?, ?)
    """, default_ingredients)

    conn.commit()
    conn.close()

def save_order(data):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 주문 저장
    cursor.execute("""
        INSERT INTO orders(order_time, total_price, status)
        VALUES (?, ?, ?)
    """, (now, data["totalPrice"], "WAITING"))

    # 방금 생성된 주문번호
    order_id = cursor.lastrowid

    # 주문한 메뉴 저장 (재고 차감은 조리 시작 시 수행)
    for item in data["items"]:
        menu_name = item["name"]
        qty = item["qty"]
        
        cursor.execute("""
            INSERT INTO order_items(order_id, menu_name, quantity, price)
            VALUES (?, ?, ?, ?)
        """, (order_id, menu_name, qty, item["price"]))

    print(f"주문번호 {order_id} 저장 완료 (조리 대기)")
    conn.commit()
    conn.close()
    return order_id

def get_orders():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT order_id, order_time, total_price, status
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
            "status": order[3],
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
        SELECT ingredient_name, stock, unit
        FROM ingredients
    """)
    inventory = cursor.fetchall()
    conn.close()
    return inventory

def update_inventory(ingredient_name, stock):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE ingredients
        SET stock=?
        WHERE ingredient_name=?
    """, (stock, ingredient_name))
    conn.commit()
    conn.close()

def check_stock(data):
    # 각 식자재별 총 필요량 계산
    required_ingredients = {}
    for item in data["items"]:
        menu_name = item["name"]
        qty = item["qty"]
        if menu_name in RECIPES:
            for ing_name, amount in RECIPES[menu_name]:
                required_ingredients[ing_name] = required_ingredients.get(ing_name, 0) + (amount * qty)

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # 재고 비교
    for ing_name, req_amount in required_ingredients.items():
        cursor.execute("""
            SELECT stock, unit
            FROM ingredients
            WHERE ingredient_name=?
        """, (ing_name,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return False, f"[{ing_name}] 식자재 정보가 없습니다."
        
        stock, unit = row
        if stock < req_amount:
            conn.close()
            return False, f"[{ing_name}] 재고가 부족합니다. (현재: {stock}{unit}, 필요: {req_amount}{unit})"

    conn.close()
    return True, ""

def update_order_status(order_id, status):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # 현재 상태 확인 (중복 차감 방지)
    cursor.execute("SELECT status FROM orders WHERE order_id=?", (order_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return
        
    current_status = row[0].upper()
    
    # 대기(WAITING)에서 조리중(COOKING)으로 넘어갈 때 재고 차감
    if current_status == "WAITING" and status.upper() == "COOKING":
        cursor.execute("SELECT menu_name, quantity FROM order_items WHERE order_id=?", (order_id,))
        items = cursor.fetchall()
        
        for menu_name, qty in items:
            if menu_name in RECIPES:
                for ing_name, amount_per_item in RECIPES[menu_name]:
                    total_deduction = amount_per_item * qty
                    cursor.execute("""
                        UPDATE ingredients
                        SET stock = stock - ?
                        WHERE ingredient_name = ?
                    """, (total_deduction, ing_name))
    
    cursor.execute("""
        UPDATE orders
        SET status=?
        WHERE order_id=?
    """, (status, order_id))
    
    conn.commit()
    conn.close()