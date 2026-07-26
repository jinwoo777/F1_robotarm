from flask import Flask, render_template, request, jsonify, redirect
import threading
import subprocess
import time
import os
from datetime import datetime
from collections import deque
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String, Float32
from geometry_msgs.msg import Point

from database import (
    create_database,
    save_order,
    get_orders,
    get_orders_by_table,
    get_today_sales,
    get_hourly_orders_today,
    get_monthly_sales,
    get_inventory,
    update_inventory,
    check_stock,
    update_order_status,
    get_order_menu_names,
    save_staff_call,
    get_pending_staff_calls,
    resolve_staff_call,
)

create_database()

app = Flask(__name__, template_folder="../frontend/templates", static_folder="../frontend/static")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WOK_INTEGRATE_LOG_PATH = os.path.join(BASE_DIR, "wok_integrate.log")

# 메뉴 이름 → wok_integrate.py의 dish ROS 파라미터. 목록에 없는 메뉴(예: 막걸리)는
# 로봇 조리가 필요 없으므로 매핑하지 않는다 — 그런 주문만 있으면 launch_wok_integrate가
# 호출되지 않아도 되지만, 현재 status 변경 흐름은 항상 로봇을 띄우므로 기본값 fried_rice로 진행한다.
MENU_TO_DISH = {
    "부침개": "jeon",
    "볶음밥": "fried_rice",
}


def is_wok_integrate_running():
    """실제 OS 프로세스 목록에서 wok_integrate 실행 여부를 직접 확인한다.

    파이썬 메모리 안에 "지금 실행 중"이라는 플래그만 두면 안 되는 이유: Flask가
    debug=True라 코드를 수정할 때마다 워커 프로세스가 재시작되면서 그 플래그는
    사라지는데, 그 전에 띄워둔 wok_integrate 서브프로세스는 안 죽고 계속 남는다.
    (wok_test4 시절 실제로 이 문제로 프로세스가 5개까지 동시에 떠서 같은 로봇을
    제어한 적이 있었다.) 그래서 메모리 상태가 아니라 매번 실제 프로세스 테이블을 확인한다.

    ★ pgrep -f는 검색어 문자열을 자기 자신의 실행 인자로도 가지고 있어서, 이 환경의
      procps 버전에서는 pgrep 프로세스 자기 자신까지 결과에 잡히는 self-match 버그가
      있었다(확인됨) — 그래서 항상 "실행 중"으로 오판해 아무도 조리를 시작 못 하게 될
      뻔했다. Popen으로 직접 띄워서 그 pgrep 프로세스의 pid를 알아낸 뒤, 결과에서
      그 pid는 명시적으로 제외한다."""
    try:
        proc = subprocess.Popen(
            ["pgrep", "-f", "rokey/lib/rokey/wok_integrate"],
            stdout=subprocess.PIPE, text=True,
        )
        stdout, _ = proc.communicate(timeout=3)
        pids = [int(p) for p in stdout.split() if p.strip()]
        real_pids = [p for p in pids if p != proc.pid]
        return bool(real_pids)
    except Exception:
        return False   # 확인 자체가 실패하면 막지 않는다(과잉 차단보다는 낫다고 판단)


def kill_wok_integrate_processes():
    """wok_integrate 프로세스가 남아있으면 정리한다("조리 완료"를 눌러도 로봇 스크립트가 멈춰
    있으면 다음 주문이 "이미 조리 중"으로 막혔던 문제 — DB 상태와 실제 로봇 프로세스는 서로
    다른 상태라 완료 버튼이 후자를 건드리지 않으면 둘이 어긋난다). SIGINT로 정상 종료를
    먼저 시도하고, 살아있으면 SIGKILL로 강제 종료한다."""
    try:
        proc = subprocess.Popen(
            ["pgrep", "-f", "rokey/lib/rokey/wok_integrate"],
            stdout=subprocess.PIPE, text=True,
        )
        stdout, _ = proc.communicate(timeout=3)
        pids = [int(p) for p in stdout.split() if p.strip() and int(p) != proc.pid]
        if not pids:
            return
        print(f"조리 완료 처리 — 남아있는 wok_integrate 프로세스 정리: {pids}")
        for pid in pids:
            try:
                os.kill(pid, 2)   # SIGINT
            except ProcessLookupError:
                pass
        time.sleep(2)
        for pid in pids:
            try:
                os.kill(pid, 9)   # SIGKILL — 아직 살아있으면 강제 종료
            except ProcessLookupError:
                pass
    except Exception as e:
        print(f"wok_integrate 프로세스 정리 실패(무시하고 진행): {e}")


# 로봇 실시간 상태 — wok_integrate가 발행하는 토픽(/wok_stage, /gripper_width, /tcp_pose)을
# FlaskRosNode가 구독해 여기에 반영하고, /api/robot_status로 admin.html에 그대로 전달한다.
# tcp_history: 관리자 대시보드의 TCP 궤적 스캐터/속도 그래프용 최근 위치 이력.
robot_status = {
    "state": "대기",
    "task": "없음",
    "order_id": "-",
    "gripper_width_mm": None,
    "tcp_x": None,
    "tcp_y": None,
    "speed_mm_s": None,
    "tcp_history": [],   # [{"x":.., "y":..}, ...] 최근 순
}
_TCP_HISTORY_MAX = 15
_last_tcp_sample = {"t": None, "x": None, "y": None}

# "웍질 시작/세트/닫기" 등은 조리중, "완료"는 완료, 그 외 안 받으면 대기(기본값 유지)
_STAGE_DONE = {"완료"}


# ROS 2 Node setup for E-Stop + 텔레메트리 구독
class FlaskRosNode(Node):
    def __init__(self):
        super().__init__('flask_ui_node')
        self.estop_pub = self.create_publisher(Bool, '/estop', 10)
        # robot_command_bridge(wok_exception_handling)가 구독 — "HOME"이면 안전정지 해제 후
        # 홈 자세로 복귀시킨다. "조리 완료" 시 로봇을 다음 주문을 받을 수 있는 상태로 되돌리는 데 쓴다.
        self.reset_robot_pub = self.create_publisher(String, '/reset_robot', 10)
        self.create_subscription(String, '/wok_stage', self._on_stage, 10)
        self.create_subscription(Float32, '/gripper_width', self._on_gripper_width, 10)
        self.create_subscription(Point, '/tcp_pose', self._on_tcp_pose, 10)

    def _on_stage(self, msg):
        robot_status["task"] = msg.data
        robot_status["state"] = "완료" if msg.data in _STAGE_DONE else "조리중"

        # wok_integrate가 "완료"를 발행하면(정상 종료 직전, 홈 복귀까지 끝난 시점) 그
        # 주문을 자동으로 DONE 처리한다 — 관리자가 "조리 완료" 버튼을 매번 수동으로 누를
        # 필요가 없다. robot_status["order_id"]는 /update_status가 COOKING을 실행시킬 때
        # 채워둔다. 이미 "-"(처리 완료로 리셋됨)면 중복 처리를 건너뛴다.
        order_id = robot_status["order_id"]
        if msg.data in _STAGE_DONE and order_id != "-":
            print(f"wok_integrate 완료 신호 수신 — 주문 {order_id} 자동 조리완료 처리")
            update_order_status(order_id, "DONE")
            robot_status["order_id"] = "-"

    def _on_gripper_width(self, msg):
        robot_status["gripper_width_mm"] = round(float(msg.data), 1)

    def _on_tcp_pose(self, msg):
        now = time.time()
        x, y = float(msg.x), float(msg.y)

        prev = _last_tcp_sample
        if prev["t"] is not None:
            dt = now - prev["t"]
            if dt > 0:
                dist = ((x - prev["x"]) ** 2 + (y - prev["y"]) ** 2) ** 0.5
                robot_status["speed_mm_s"] = round(dist / dt, 1)
        prev["t"], prev["x"], prev["y"] = now, x, y

        robot_status["tcp_x"] = round(x, 1)
        robot_status["tcp_y"] = round(y, 1)
        history = robot_status["tcp_history"]
        history.append({"x": round(x, 1), "y": round(y, 1)})
        if len(history) > _TCP_HISTORY_MAX:
            del history[0]


ros_node = None

def ros_spin_thread():
    global ros_node
    rclpy.init()
    ros_node = FlaskRosNode()
    rclpy.spin(ros_node)
    ros_node.destroy_node()
    rclpy.shutdown()

threading.Thread(target=ros_spin_thread, daemon=True).start()

@app.route("/")
def index():
    # 테이블 QR코드가 /?table=1 형태로 들어온다 — 없거나(예: 관리자가 직접 접속) 숫자가
    # 아니면 1번으로 취급한다. 이 값이 그대로 <script> 안 문자열로 렌더링되므로,
    # 숫자만 허용해서 URL 조작을 통한 스크립트 주입을 막는다.
    raw_table = request.args.get("table", "1")
    table_number = raw_table if raw_table.isdigit() else "1"
    return render_template("jumak_order.html", table_number=table_number)

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

@app.route("/api/orders")
def orders_by_table():
    table_number = request.args.get("table", "1")
    return jsonify(get_orders_by_table(table_number))

@app.route("/api/call_staff", methods=["POST"])
def call_staff():
    data = request.get_json()
    table_number = data.get("tableNumber", "1")
    call_id = save_staff_call(table_number)
    return jsonify({"status": "success", "callId": call_id})

@app.route("/api/staff_calls")
def staff_calls():
    return jsonify(get_pending_staff_calls())

@app.route("/api/staff_calls/resolve", methods=["POST"])
def staff_calls_resolve():
    data = request.get_json()
    resolve_staff_call(data["call_id"])
    return jsonify({"status": "success"})

@app.route("/api/sales_calendar")
def sales_calendar():
    now = datetime.now()
    year = request.args.get("year", type=int, default=now.year)
    month = request.args.get("month", type=int, default=now.month)
    if month < 1 or month > 12:
        return jsonify({"status": "error", "message": "month must be 1-12"}), 400

    daily = get_monthly_sales(year, month)
    return jsonify({
        "year": year,
        "month": month,
        "total": sum(daily.values()),
        "daily": daily,   # {"5": 2234000, ...} — 매출 없는 날은 없음
    })

@app.route("/admin")
def admin():

    orders = get_orders()

    today_sales = get_today_sales()

    inventory = get_inventory()

    # 시간대별 주문 처리 차트 — 오늘 실제 주문 건수(DB 기반). 영업시간대만 표시(9~21시).
    hourly = get_hourly_orders_today()
    hourly_labels = [f"{h}시" for h in range(9, 22)]
    hourly_counts = [hourly.get(h, 0) for h in range(9, 22)]

    return render_template(
        "admin.html",
        orders=orders,
        today_sales=today_sales,
        inventory=inventory,
        hourly_labels=hourly_labels,
        hourly_counts=hourly_counts,
        warning=request.args.get("warning"),
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

    if status == "COOKING":
        if is_wok_integrate_running():
            print(f"이미 조리 중인 로봇 프로세스가 있습니다 — 주문 {order_id} 중복 실행 방지")
            return redirect("/admin?warning=already_cooking")

        # 주문에 담긴 메뉴로 dish를 고른다. 부침개가 하나라도 섞여 있으면 jeon을 우선한다 —
        # 한 주문에 볶음밥+부침개가 같이 들어와도 로봇 세션은 하나뿐이라 둘을 동시에 조리할
        # 수 없고, 재료투입/레버가 없는 jeon 쪽이 실수로 fried_rice 취급되면 웍에 이미 있는
        # 반죽 위로 재료 투입 동작이 얹히는 쪽이 더 위험하다고 판단했다.
        menu_names = get_order_menu_names(order_id)
        dish = "fried_rice"
        for menu_name in menu_names:
            if MENU_TO_DISH.get(menu_name) == "jeon":
                dish = "jeon"
                break

        print(f"주문 {order_id} 조리 시작! dish={dish} — ROS 2 노드 실행... (로그: {WOK_INTEGRATE_LOG_PATH})")
        cmd = f"""
        source /opt/ros/humble/setup.bash
        source /home/rokey/wok_wark/ws_cobot_pjt/ws_dsr/install/setup.bash
        export ROS_DOMAIN_ID=50
        export PYTHONPATH=$PYTHONPATH:~/ros2_ws/install/dsr_common2/lib/dsr_common2/imp
        export PYTHONPATH=$PYTHONPATH:~/Desktop/Cobot1/ws_cobot_pjt/ws_dsr/install/dsr_common2/lib/dsr_common2/imp
        export PYTHONPATH=$PYTHONPATH:~/wok_wark/ws_cobot_pjt/ws_dsr/install/dsr_common2/lib/dsr_common2/imp
        ros2 run rokey wok_integrate --ros-args -p dish:={dish}
        """
        # stdout/stderr를 DEVNULL로 버리면 나중에 프로세스가 멈춰도 원인을 볼 수 없다
        # (실제로 이 때문에 왜 멈췄는지 못 보고 "좀비 프로세스"로만 보였던 적이 있다).
        # 이어쓰기(append) — 여러 조리 기록이 순서대로 로그 파일에 쌓인다.
        log_file = open(WOK_INTEGRATE_LOG_PATH, "a")
        subprocess.Popen(
            cmd,
            shell=True,
            executable="/bin/bash",
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True
        )
        # FlaskRosNode._on_stage가 "완료" 신호를 받았을 때 어느 주문을 자동으로 DONE
        # 처리할지 알아야 하므로 여기서 채워둔다.
        robot_status["order_id"] = str(order_id)

    if status == "DONE":
        robot_status["order_id"] = "-"   # 수동 완료 처리 — 자동완료 중복 트리거 방지
        # 남아있는(특히 멈춰버린) wok_integrate 프로세스를 정리하고, robot_command_bridge에
        # HOME을 보내 안전정지 해제 + 홈 복귀까지 시켜서 다음 주문을 바로 받을 수 있는
        # 상태로 되돌린다. 프로세스가 이미 정상 종료돼 있어도 HOME 자체는 안전한 동작이라
        # 무해하다(로봇이 이미 홈이면 movej가 짧게 끝날 뿐).
        kill_wok_integrate_processes()
        if ros_node:
            ros_node.reset_robot_pub.publish(String(data="HOME"))
            print(f"주문 {order_id} 조리 완료 — 로봇 HOME 복귀 신호 발행")

    update_order_status(order_id, status)
    return redirect("/admin")

@app.route("/api/estop", methods=["POST"])
def api_estop():
    if ros_node:
        msg = Bool()
        msg.data = True
        ros_node.estop_pub.publish(msg)
        print("🚨 긴급정지 신호 퍼블리시 완료!")
    return jsonify({"status": "success"})

@app.route("/api/resume", methods=["POST"])
def api_resume():
    if ros_node:
        msg = Bool()
        msg.data = False
        ros_node.estop_pub.publish(msg)
        print("▶️ 작업 재개 신호 퍼블리시 완료!")
    return jsonify({"status": "success"})

# 상태 조회 (robot_status는 파일 상단에서 정의되고 FlaskRosNode가 실시간으로 채운다)
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
    # host="0.0.0.0" — 같은 매장 와이파이에 붙은 손님 휴대폰이 테이블 QR코드로
    # 이 서버에 접속할 수 있어야 하므로 localhost로만 열면 안 된다.
    app.run(host="0.0.0.0", port=5000, debug=True)