"""estop_button_io — Doosan 컨트롤박스 DI 버튼 + OnRobot RG2 제어.

버튼 기능
  - DI 13 상승엣지 -> /estop=True, /alarm(MANUAL_ESTOP) 발행 + RG2 정지 요청
  - DI 14 상승엣지 -> RG2 닫기(RG2_CLOSE_WIDTH_MM)
  - DI 15 상승엣지 -> RG2 열기(RG2_OPEN_WIDTH_MM)
  - DI 16 상승엣지 -> /manual_resume 발행

RG2 통신
  - OnRobot Compute Box/Eye Box의 Modbus TCP를 사용한다.
  - 설치형 모듈: ``from onrobot_rg2 import RG2Client``
  - ROS 2 패키지 내부에 onrobot_rg2.py를 둘 경우 상대 import도 지원한다.

실행 예시
  ros2 run wok_exception_handling estop_button_io --ros-args \
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
from dsr_msgs2.srv import GetCtrlBoxDigitalInput
from rclpy.node import Node
from std_msgs.msg import Bool, Empty, String

# 설치된 wheel/패키지와 ROS 2 패키지 내부 모듈 두 방식을 모두 지원한다.
_RG2_IMPORT_ERROR: Optional[ImportError] = None
try:
    from onrobot_rg2 import RG2Client, RG2Error
except ImportError as absolute_import_error:
    try:
        from .onrobot_rg2 import RG2Client, RG2Error
    except ImportError as relative_import_error:
        RG2Client = None  # type: ignore[assignment,misc]
        RG2Error = Exception  # type: ignore[assignment,misc]
        _RG2_IMPORT_ERROR = relative_import_error
    else:
        _RG2_IMPORT_ERROR = None
else:
    _RG2_IMPORT_ERROR = None


ROBOT_ID = "dsr01"

DI_ESTOP = 13
DI_GRIP_CLOSE = 14
DI_GRIP_OPEN = 15
DI_RESUME = 16

# IO 서비스는 한 번에 index 하나씩만 읽는다.
# DI13은 안전을 위해 가장 자주 읽고, DI14/15도 짧은 버튼 입력을 놓치지 않도록
# DI16보다 우선해서 읽는다. CYCLE_DT=0.25초 기준 대략:
#   DI13 약 1.78 Hz, DI14/15 약 0.89 Hz, DI16 약 0.44 Hz
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


class EstopButtonRG2IO(Node):
    """DI13~16 버튼을 읽어 E-STOP/재개 및 RG2 동작을 처리한다."""

    def __init__(self) -> None:
        super().__init__("estop_button_io", namespace=ROBOT_ID)

        # 컨트롤박스 DI 읽기 서비스.
        # 상대 이름이므로 최종 서비스는 /dsr01/io/get_ctrl_box_digital_input 이다.
        self._di_client = self.create_client(
            GetCtrlBoxDigitalInput,
            "io/get_ctrl_box_digital_input",
        )

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
            "estop_button_io 시작 — "
            f"DI{DI_ESTOP}=정지, DI{DI_GRIP_CLOSE}=RG2 닫기, "
            f"DI{DI_GRIP_OPEN}=RG2 열기, DI{DI_RESUME}=재개"
        )
        self.get_logger().info(
            "RG2 설정 — "
            f"host={self._rg2_host}:{self._rg2_port}, unit_id={self._rg2_unit_id}, "
            f"닫기={self._rg2_close_width_mm:.1f} mm, "
            f"열기={self._rg2_open_width_mm:.1f} mm, "
            f"force={self._rg2_force_n:.1f} N"
        )

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
            self.get_logger().warn(f"DI{DI_ESTOP} 버튼 눌림 — 수동 E-STOP")

            # 로봇 정지 신호를 가장 먼저 발행한다.
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
    node = EstopButtonRG2IO()

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
