"""estop_button_io_integrate — estop_button_io.py + estop_button_io(1).py 병합.

기존 두 파일은 그대로 두고 이 파일에서만 합친다.

  - DI13(정지)/DI16(재개) 처리, move_stop 즉시정지, 단일 스레드 spin 제약은
    estop_button_io.py(base) 그대로 유지한다.
  - DI14(RG2 닫기)/DI15(RG2 열기) 그리퍼 제어, RG2 ROS 파라미터, 워커 스레드 +
    우선순위 큐, E-STOP 중 그리퍼 명령 게이팅, graceful shutdown은
    estop_button_io(1).py에서 그대로 이식한다.

★ (1)에는 DI13 처리에 `move_stop` 호출이 없었다(MoveStop import 자체가 없음) — 그 상태로는
  버튼을 눌러도 로봇 모션이 즉시 안 멈추고, wok_integrate 쪽 체크포인트(movel/movej 완료
  시점)까지 계속 진행된 뒤에야 멈춘다. base가 이 기능을 만든 이유(헤더 참고: dsr_controller2는
  공유 서비스라 이 프로세스가 move_stop을 직접 부르면 다른 프로세스가 movel/movej로 블로킹
  중이어도 그 즉시 멈춘다)를 그대로 살려, DI13 상승엣지에서 move_stop을 먼저 호출한 뒤
  그리퍼 stop을 큐잉한다.

★ 이 노드는 독립된 프로세스(자체 rclpy.init())로 실행되며, DI 폴링은 메인 스레드 하나에서만
  spin한다 — rclpy가 같은 프로세스 안 여러 executor의 동시 spin을 안전하게 지원하지 않아서
  wok_test4가 `IndexError: wait set index too big`로 죽은 적이 있기 때문(base 헤더와 동일 근거).
  RG2 그리퍼 워커 스레드는 rclpy spin을 전혀 하지 않는 순수 Modbus 소켓 스레드라 이 제약과
  무관하게 안전하다.

실행 예시
  ros2 run wok_exception_handling estop_button_io_integrate --ros-args \
    -p rg2_host:=192.168.1.1 \
    -p rg2_unit_id:=65 \
    -p rg2_close_width_mm:=0.0 \
    -p rg2_open_width_mm:=110.0 \
    -p rg2_force_n:=20.0
"""

from __future__ import annotations

import itertools
import queue
import threading
import time
from typing import Any, Optional

import rclpy
from dsr_msgs2.srv import GetCtrlBoxDigitalInput, MoveStop
from rclpy.node import Node
from std_msgs.msg import Bool, Empty, String

# onrobot_rg2.py는 wok_exception_handling 패키지 안에 없고 rokey 패키지
# (rokey/rokey/onrobot_rg2/)에만 있다. estop_button_io(1)의 원래 import 순서
# (설치된 wheel → 같은 패키지 내부 상대 import)는 이 워크스페이스에 둘 다 없어서 항상
# 실패했다 — wok_integrate.py가 이미 쓰고 있는 `rokey.onrobot_rg2` 절대 import로 바꾼다
# (rokey는 develop 모드로 설치돼 있어 워크스페이스를 source하면 다른 패키지에서도
# import 가능하다 — wok_integrate.py에서 실제로 동작 확인됨).
_RG2_IMPORT_ERROR: Optional[ImportError] = None
try:
    from rokey.onrobot_rg2 import RG2Client, RG2Error
except ImportError as import_error:
    RG2Client = None  # type: ignore[assignment,misc]
    RG2Error = Exception  # type: ignore[assignment,misc]
    _RG2_IMPORT_ERROR = import_error


ROBOT_ID = "dsr01"

DI_ESTOP = 13
DI_GRIP_CLOSE = 14
DI_GRIP_OPEN = 15
DI_RESUME = 16

# IO 서비스는 한 번에 index 하나씩만 읽는다(연속으로 여러 index를 빠르게 읽으면 값이 밀림 —
# base/​(1) 공통 확인 사항). DI13은 안전을 위해 가장 자주, DI14/15도 DI16보다 우선해서 읽는다.
POLL_SEQUENCE = (
    DI_ESTOP,
    DI_GRIP_CLOSE,
    DI_ESTOP,
    DI_GRIP_OPEN,
    DI_ESTOP,
    DI_GRIP_CLOSE,
    DI_ESTOP,
    DI_GRIP_OPEN,
    DI_RESUME,
)
CYCLE_DT = 0.25
READ_TIMEOUT = 1.0
DEBUG_LOG_PERIOD_SEC = 3.0

# 기본 RG2 설정. ROS 파라미터로 실행 시 변경할 수 있다.
DEFAULT_RG2_HOST = "192.168.1.1"
DEFAULT_RG2_PORT = 502
DEFAULT_RG2_UNIT_ID = 65
DEFAULT_RG2_CLOSE_WIDTH_MM = 0.0
DEFAULT_RG2_OPEN_WIDTH_MM = 110.0
DEFAULT_RG2_FORCE_N = 20.0
DEFAULT_RG2_SOCKET_TIMEOUT_SEC = 1.0
DEFAULT_INCLUDE_FINGERTIP_OFFSET = True


class EstopButtonIOIntegrate(Node):
    """DI13~16 버튼을 읽어 E-STOP(즉시 move_stop)/재개 및 RG2 그리퍼 동작을 처리한다."""

    def __init__(self) -> None:
        super().__init__("estop_button_io", namespace=ROBOT_ID)

        # 컨트롤박스 DI 읽기 서비스. 상대이름 → /dsr01/io/get_ctrl_box_digital_input
        self._di_client = self.create_client(
            GetCtrlBoxDigitalInput,
            "io/get_ctrl_box_digital_input",
        )
        # DI13 감지 즉시 진행 중 동작을 멈추는 서비스. 상대이름 → /dsr01/motion/move_stop
        # (base 기능 — (1)에는 없었다. 반드시 유지.)
        self._stop_cli = self.create_client(MoveStop, "motion/move_stop")

        self.estop_pub = self.create_publisher(Bool, "/estop", 10)
        self.alarm_pub = self.create_publisher(String, "/alarm", 10)
        self.resume_pub = self.create_publisher(Empty, "/manual_resume", 10)

        monitored_dis = (DI_ESTOP, DI_GRIP_CLOSE, DI_GRIP_OPEN, DI_RESUME)
        self._prev = {idx: False for idx in monitored_dis}
        self._last = {idx: -9 for idx in monitored_dis}
        self._poll_position = 0
        self._last_debug_log_time = time.monotonic()
        self._manual_estop_latched = False

        self._declare_rg2_parameters()
        self._load_rg2_parameters()

        self._gripper: Optional[Any] = None
        self._gripper_queue: queue.PriorityQueue[tuple[int, int, str]] = (
            queue.PriorityQueue(maxsize=8)
        )
        self._gripper_queue_counter = itertools.count()
        self._gripper_worker_stop = threading.Event()
        self._gripper_worker: Optional[threading.Thread] = None

        self._initialize_gripper_worker()

        self.get_logger().info(
            "estop_button_io_integrate 시작 — "
            f"DI{DI_ESTOP}=정지(move_stop), DI{DI_GRIP_CLOSE}=RG2 닫기, "
            f"DI{DI_GRIP_OPEN}=RG2 열기, DI{DI_RESUME}=재개"
        )
        self.get_logger().info(
            "RG2 설정 — "
            f"host={self._rg2_host}:{self._rg2_port}, unit_id={self._rg2_unit_id}, "
            f"닫기={self._rg2_close_width_mm:.1f} mm, "
            f"열기={self._rg2_open_width_mm:.1f} mm, "
            f"force={self._rg2_force_n:.1f} N"
        )

    # ── RG2 파라미터 ((1) 그대로) ───────────────────────────────────────

    def _declare_rg2_parameters(self) -> None:
        self.declare_parameter("rg2_host", DEFAULT_RG2_HOST)
        self.declare_parameter("rg2_port", DEFAULT_RG2_PORT)
        self.declare_parameter("rg2_unit_id", DEFAULT_RG2_UNIT_ID)
        self.declare_parameter("rg2_close_width_mm", DEFAULT_RG2_CLOSE_WIDTH_MM)
        self.declare_parameter("rg2_open_width_mm", DEFAULT_RG2_OPEN_WIDTH_MM)
        self.declare_parameter("rg2_force_n", DEFAULT_RG2_FORCE_N)
        self.declare_parameter(
            "rg2_socket_timeout_sec",
            DEFAULT_RG2_SOCKET_TIMEOUT_SEC,
        )
        self.declare_parameter(
            "rg2_include_fingertip_offset",
            DEFAULT_INCLUDE_FINGERTIP_OFFSET,
        )

    def _load_rg2_parameters(self) -> None:
        self._rg2_host = str(self.get_parameter("rg2_host").value)
        self._rg2_port = int(self.get_parameter("rg2_port").value)
        self._rg2_unit_id = int(self.get_parameter("rg2_unit_id").value)
        self._rg2_close_width_mm = float(
            self.get_parameter("rg2_close_width_mm").value
        )
        self._rg2_open_width_mm = float(
            self.get_parameter("rg2_open_width_mm").value
        )
        self._rg2_force_n = float(self.get_parameter("rg2_force_n").value)
        self._rg2_socket_timeout_sec = float(
            self.get_parameter("rg2_socket_timeout_sec").value
        )
        self._rg2_include_fingertip_offset = bool(
            self.get_parameter("rg2_include_fingertip_offset").value
        )

        if not 0.0 <= self._rg2_close_width_mm <= 110.0:
            raise ValueError("rg2_close_width_mm must be between 0 and 110 mm")
        if not 0.0 <= self._rg2_open_width_mm <= 110.0:
            raise ValueError("rg2_open_width_mm must be between 0 and 110 mm")
        if not 3.0 <= self._rg2_force_n <= 40.0:
            raise ValueError("rg2_force_n must be between 3 and 40 N")
        if self._rg2_port <= 0 or self._rg2_port > 65535:
            raise ValueError("rg2_port must be between 1 and 65535")
        if self._rg2_socket_timeout_sec <= 0.0:
            raise ValueError("rg2_socket_timeout_sec must be greater than 0")

    def _initialize_gripper_worker(self) -> None:
        if RG2Client is None:
            self.get_logger().error(
                "OnRobot RG2 모듈을 불러오지 못했습니다. "
                "E-STOP/재개 버튼은 계속 동작하지만 DI14/15는 사용할 수 없습니다. "
                f"import error={_RG2_IMPORT_ERROR}"
            )
            return

        self._gripper = RG2Client(
            host=self._rg2_host,
            port=self._rg2_port,
            unit_id=self._rg2_unit_id,
            timeout=self._rg2_socket_timeout_sec,
        )
        self._gripper_worker = threading.Thread(
            target=self._gripper_worker_loop,
            name="rg2-command-worker",
            daemon=True,
        )
        self._gripper_worker.start()

    # ── DI 폴링 (base 그대로) ───────────────────────────────────────────

    def read_di(self, idx: int) -> Optional[int]:
        """DI 한 채널을 읽어 0/1을 반환하고 실패 시 None을 반환한다."""
        if not self._di_client.service_is_ready():
            return None

        request = GetCtrlBoxDigitalInput.Request()
        request.index = idx
        future = self._di_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=READ_TIMEOUT)

        if not future.done():
            return None

        response = future.result()
        if response is None or not response.success:
            return None

        return int(response.value)

    def poll_once(self) -> None:
        """POLL_SEQUENCE의 DI 하나를 읽고 상승엣지를 처리한다."""
        idx = POLL_SEQUENCE[self._poll_position]
        self._poll_position = (self._poll_position + 1) % len(POLL_SEQUENCE)

        value = self.read_di(idx)
        if value is not None:
            self._last[idx] = value
            pressed = bool(value)

            if pressed and not self._prev[idx]:
                self._handle_rising_edge(idx)

            self._prev[idx] = pressed

        self._log_debug_state_if_due()

    def _handle_rising_edge(self, idx: int) -> None:
        if idx == DI_ESTOP:
            self._manual_estop_latched = True
            self.get_logger().warn(
                f"DI{DI_ESTOP} 버튼 눌림 — move_stop으로 즉시 정지 + 수동 E-STOP"
            )

            # base 기능: dsr_controller2는 공유 서비스라서, wok_integrate가 movel/movej로
            # 블로킹돼 있어도 이 프로세스가 독립적으로 move_stop을 호출해 그 즉시 멈출 수 있다.
            # /estop 토픽만 발행하면 wok_integrate의 다음 체크포인트까지 정지가 지연된다.
            if self._stop_cli.service_is_ready():
                stop_req = MoveStop.Request()
                stop_req.stop_mode = 1  # DR_QSTOP: Quick stop (Stop Category 2)
                stop_future = self._stop_cli.call_async(stop_req)
                rclpy.spin_until_future_complete(self, stop_future, timeout_sec=0.5)

            self.alarm_pub.publish(String(data="MANUAL_ESTOP"))
            self.estop_pub.publish(Bool(data=True))

            # E-STOP 전에 대기 중이던 열기/닫기 명령을 제거하고,
            # RG2 정지는 별도 작업 스레드에서 best-effort로 처리한다.
            self._discard_pending_gripper_commands()
            self._queue_gripper_command("stop", priority=0)
            return

        if idx == DI_RESUME:
            self.get_logger().info(f"DI{DI_RESUME} 버튼 눌림 — 관리자 재개 요청")
            self.resume_pub.publish(Empty())
            self._manual_estop_latched = False
            return

        if self._manual_estop_latched:
            self.get_logger().warn(
                f"DI{idx} 입력 무시 — 수동 E-STOP이 해제되지 않았습니다. "
                f"DI{DI_RESUME}으로 재개한 뒤 사용하세요."
            )
            return

        if idx == DI_GRIP_CLOSE:
            self.get_logger().info(
                f"DI{DI_GRIP_CLOSE} 버튼 눌림 — RG2 닫기 "
                f"({self._rg2_close_width_mm:.1f} mm)"
            )
            self._queue_gripper_command("close", priority=10)
            return

        if idx == DI_GRIP_OPEN:
            self.get_logger().info(
                f"DI{DI_GRIP_OPEN} 버튼 눌림 — RG2 열기 "
                f"({self._rg2_open_width_mm:.1f} mm)"
            )
            self._queue_gripper_command("open", priority=10)

    # ── RG2 그리퍼 명령 큐/워커 ((1) 그대로) ─────────────────────────────

    def _discard_pending_gripper_commands(self) -> None:
        """아직 실행되지 않은 RG2 명령을 모두 제거한다."""
        discarded = 0
        while True:
            try:
                self._gripper_queue.get_nowait()
            except queue.Empty:
                break
            else:
                self._gripper_queue.task_done()
                discarded += 1

        if discarded:
            self.get_logger().warn(
                f"E-STOP으로 대기 중인 RG2 명령 {discarded}개를 폐기했습니다."
            )

    def _queue_gripper_command(self, command: str, priority: int) -> None:
        if self._gripper is None or self._gripper_worker is None:
            self.get_logger().error(
                f"RG2 {command} 명령 실패 — RG2 모듈/클라이언트가 준비되지 않았습니다."
            )
            return

        item = (priority, next(self._gripper_queue_counter), command)
        try:
            self._gripper_queue.put_nowait(item)
        except queue.Full:
            self.get_logger().error(
                f"RG2 {command} 명령 실패 — 명령 큐가 가득 찼습니다."
            )

    def _gripper_worker_loop(self) -> None:
        while not self._gripper_worker_stop.is_set():
            try:
                _, _, command = self._gripper_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                if command == "stop":
                    self._stop_gripper()
                elif command == "close":
                    if self._manual_estop_latched:
                        self.get_logger().warn(
                            "RG2 닫기 명령 폐기 — 수동 E-STOP이 활성 상태입니다."
                        )
                    else:
                        self._move_gripper(self._rg2_close_width_mm, "닫기")
                elif command == "open":
                    if self._manual_estop_latched:
                        self.get_logger().warn(
                            "RG2 열기 명령 폐기 — 수동 E-STOP이 활성 상태입니다."
                        )
                    else:
                        self._move_gripper(self._rg2_open_width_mm, "열기")
                else:
                    self.get_logger().error(f"알 수 없는 RG2 명령: {command}")
            finally:
                self._gripper_queue.task_done()

    def _move_gripper(self, target_width_mm: float, action_name: str) -> None:
        gripper = self._gripper
        if gripper is None:
            return

        try:
            status = gripper.get_status()
            if status.safety_error:
                self.get_logger().error(
                    f"RG2 {action_name} 거부 — RG2 safety_error가 활성화되어 있습니다."
                )
                return

            # 반대 버튼이 이동 중 눌린 경우 현재 동작을 멈춘 뒤 새 목표를 보낸다.
            if status.busy:
                gripper.stop()
                time.sleep(0.1)

            command = gripper.start_move_to_width(
                target_width_mm=target_width_mm,
                force_n=self._rg2_force_n,
                include_fingertip_offset=self._rg2_include_fingertip_offset,
                check_ready=False,
            )
            self.get_logger().info(
                f"RG2 {action_name} 명령 전송 완료 — "
                f"target={command.target_width_mm:.1f} mm, "
                f"force={command.force_n:.1f} N"
            )
        except (RG2Error, OSError, ValueError) as exc:
            self.get_logger().error(f"RG2 {action_name} 명령 실패: {exc}")
            self._reset_gripper_connection()
        except Exception as exc:  # 예상하지 못한 오류도 노드를 종료시키지 않는다.
            self.get_logger().error(
                f"RG2 {action_name} 처리 중 예상하지 못한 오류: "
                f"{type(exc).__name__}: {exc}"
            )
            self._reset_gripper_connection()

    def _stop_gripper(self) -> None:
        gripper = self._gripper
        if gripper is None:
            return

        try:
            gripper.stop()
            self.get_logger().warn("RG2 STOP 명령 전송 완료")
        except (RG2Error, OSError, ValueError) as exc:
            self.get_logger().error(f"RG2 STOP 명령 실패: {exc}")
            self._reset_gripper_connection()
        except Exception as exc:
            self.get_logger().error(
                "RG2 STOP 처리 중 예상하지 못한 오류: "
                f"{type(exc).__name__}: {exc}"
            )
            self._reset_gripper_connection()

    def _reset_gripper_connection(self) -> None:
        gripper = self._gripper
        if gripper is None:
            return
        try:
            gripper.close()
        except Exception:
            pass

    def _log_debug_state_if_due(self) -> None:
        now = time.monotonic()
        if now - self._last_debug_log_time < DEBUG_LOG_PERIOD_SEC:
            return

        self._last_debug_log_time = now
        self.get_logger().info(
            "[디버그] "
            f"DI{DI_ESTOP}={self._last[DI_ESTOP]}, "
            f"DI{DI_GRIP_CLOSE}={self._last[DI_GRIP_CLOSE]}, "
            f"DI{DI_GRIP_OPEN}={self._last[DI_GRIP_OPEN]}, "
            f"DI{DI_RESUME}={self._last[DI_RESUME]}"
        )

    def shutdown(self) -> None:
        """작업 스레드와 RG2 연결을 안전하게 정리한다."""
        self._gripper_worker_stop.set()

        if self._gripper_worker is not None:
            self._gripper_worker.join(timeout=self._rg2_socket_timeout_sec + 0.5)

        self._reset_gripper_connection()


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = EstopButtonIOIntegrate()

    try:
        while rclpy.ok():
            node.poll_once()
            time.sleep(CYCLE_DT)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
