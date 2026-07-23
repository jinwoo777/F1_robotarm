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

    menu_name = request.form["menu_name"]

    stock = int(request.form["stock"])

    max_stock = int(request.form["max_stock"])

    update_inventory(
        menu_name,
        stock,
        max_stock
    )

    return redirect("/admin")

@app.route("/api/inventory")
def inventory_api():

    inventory = get_inventory()

    menu_ids = {
        "볶음밥":"menu1",
        "파전":"menu2",
        "막걸리":"menu3"
    }

    result=[]

    for item in inventory:
        result.append({
            "id": menu_ids[item[0]],
            "name": item[0],
            "stock": item[1],
            "max_stock": item[2]
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