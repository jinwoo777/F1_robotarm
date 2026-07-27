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
        status TEXT DEFAULT 'waiting',
        table_number TEXT DEFAULT '1'
    )
    """)

    # 기존 DB(food.db)에는 table_number 컬럼이 없을 수 있어 마이그레이션 —
    # QR 테이블 오더 도입 전에 만들어진 DB 파일과의 호환을 위해 필요.
    cursor.execute("PRAGMA table_info(orders)")
    existing_columns = {row[1] for row in cursor.fetchall()}
    if "table_number" not in existing_columns:
        cursor.execute("ALTER TABLE orders ADD COLUMN table_number TEXT DEFAULT '1'")

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

    # 직원호출 기록 — 손님이 테이블 오더 화면에서 호출하면 대기(PENDING) 상태로 쌓이고,
    # 관리자가 admin.html에서 확인하면 DONE으로 바뀐다.
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS staff_calls(
        call_id INTEGER PRIMARY KEY AUTOINCREMENT,
        table_number TEXT,
        call_time TEXT,
        status TEXT DEFAULT 'PENDING'
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
    table_number = str(data.get("tableNumber", "1"))

    # 주문 저장
    cursor.execute("""
        INSERT INTO orders(order_time, total_price, status, table_number)
        VALUES (?, ?, ?, ?)
    """, (now, data["totalPrice"], "WAITING", table_number))

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
        SELECT order_id, order_time, total_price, status, table_number
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
            "table_number": order[4],
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

def get_hourly_orders_today():
    """오늘 시간대별(0~23시) 주문 건수. {hour(int): count(int)} — 주문 없는 시간대는 키 없음."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")
    cursor.execute("""
        SELECT CAST(strftime('%H', order_time) AS INTEGER) AS hour, COUNT(*)
        FROM orders
        WHERE DATE(order_time) = ?
        GROUP BY hour
    """, (today,))
    rows = cursor.fetchall()
    conn.close()
    return {hour: count for hour, count in rows}

def get_monthly_sales(year, month):
    """해당 연/월의 일자별 매출 합계. {day(int): total_price(int)} — 매출 없는 날은 키 없음."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    ym = f"{year:04d}-{month:02d}"
    cursor.execute("""
        SELECT CAST(strftime('%d', order_time) AS INTEGER) AS day, SUM(total_price)
        FROM orders
        WHERE strftime('%Y-%m', order_time) = ?
        GROUP BY day
    """, (ym,))
    rows = cursor.fetchall()
    conn.close()
    return {day: total for day, total in rows}

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

def get_order_menu_names(order_id):
    """해당 주문에 포함된 메뉴 이름 목록. wok_integrate 실행 시 dish 파라미터
    (fried_rice/jeon) 결정에 쓰인다."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT menu_name FROM order_items WHERE order_id=?", (order_id,))
    rows = cursor.fetchall()
    conn.close()
    return [row[0] for row in rows]

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

def get_orders_by_table(table_number):
    """해당 테이블(홀)의 주문 내역 — 테이블 오더 화면의 '주문내역' 팝업이 새로고침 후에도
    조회할 수 있도록 서버(DB) 기준으로 반환한다."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT order_id, order_time, total_price, status
        FROM orders
        WHERE table_number=?
        ORDER BY order_id DESC
    """, (str(table_number),))
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

def save_staff_call(table_number):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO staff_calls(table_number, call_time, status)
        VALUES (?, ?, ?)
    """, (str(table_number), now, "PENDING"))
    call_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return call_id

def get_pending_staff_calls():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT call_id, table_number, call_time
        FROM staff_calls
        WHERE status='PENDING'
        ORDER BY call_id ASC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [{"call_id": r[0], "table_number": r[1], "call_time": r[2]} for r in rows]

def resolve_staff_call(call_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("UPDATE staff_calls SET status='DONE' WHERE call_id=?", (call_id,))
    conn.commit()
    conn.close()