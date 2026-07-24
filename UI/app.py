from flask import Flask, render_template, request, jsonify, redirect
from database import (
    create_database,
    save_order,
    get_orders,
    get_today_sales,
    get_inventory,
    update_inventory,
    check_stock,
    update_order_status
)

create_database()

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("jumak_order.html")

@app.route("/api/orders", methods=["POST"])
def order():

    data = request.get_json()

    ok, message = check_stock(data)

    if not ok:
        return jsonify({
            "status":"error",
            "message":message
        })

    order_id = save_order(data)

    return jsonify({
        "status":"success",
        "orderId":order_id
    })

@app.route("/admin")
def admin():

    orders = get_orders()

    today_sales = get_today_sales()

    inventory = get_inventory()

    return render_template(
        "admin.html",
        orders=orders,
        today_sales=today_sales,
        inventory=inventory
    )

@app.route("/update_stock", methods=["POST"])
def update_stock():
    ingredient_name = request.form["ingredient_name"]
    stock = int(request.form["stock"])
    
    update_inventory(ingredient_name, stock)
    
    return redirect("/admin")

@app.route("/api/inventory")
def inventory_api():
    from database import RECIPES
    
    inventory = get_inventory()
    stock_dict = {item[0]: item[1] for item in inventory}

    menu_ids = {
        "볶음밥": "menu1",
        "부침개": "menu2",
        "막걸리": "menu3"
    }

    result = []
    for menu_name, m_id in menu_ids.items():
        if menu_name in RECIPES:
            max_portions = float('inf')
            for ing_name, req_amount in RECIPES[menu_name]:
                ing_stock = stock_dict.get(ing_name, 0)
                portions = ing_stock // req_amount
                if portions < max_portions:
                    max_portions = portions
            
            result.append({
                "id": m_id,
                "name": menu_name,
                "stock": max_portions if max_portions != float('inf') else 0
            })

    return jsonify(result)

@app.route("/update_status", methods=["POST"])
def update_status():

    order_id = request.form["order_id"]
    status = request.form["status"]

    update_order_status(order_id, status)

    return redirect("/admin")

robot_status = {
    "state": "대기",
    "task": "없음",
    "order_id": "-"
}

# 상태 조회
@app.route("/api/robot_status")
def robot_state():
    return jsonify(robot_status)


# 상태 변경
@app.route("/api/robot_status", methods=["POST"])
def update_robot_status():

    data = request.get_json()

    robot_status["state"] = data.get("state", robot_status["state"])
    robot_status["task"] = data.get("task", robot_status["task"])
    robot_status["order_id"] = data.get("order_id", robot_status["order_id"])

    return jsonify({
        "status": "success"
    })

if __name__ == "__main__":
    app.run(debug=True)