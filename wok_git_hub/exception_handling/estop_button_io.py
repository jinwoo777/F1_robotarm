"""estop_button_io — 컨트롤박스 디지털 입력 13/16번을 이용한 물리 버튼 E-STOP/재개.

  - DI 13 상승엣지 → move_stop으로 로봇을 '즉시' 정지시킨 뒤 /estop=True, /alarm(MANUAL_ESTOP)
    발행 (자동복구 금지, 관리자 대기)
  - DI 16 상승엣지 → /manual_resume 발행 (관리자 재개 확인)

★ 이 노드는 독립된 프로세스(자체 rclpy.init())로 실행되며, 단일 스레드에서만 spin한다.
  이렇게 해야 안전하다 — 한때 이 DI13/16 감시 로직을 wok_test4 프로세스 안 별도 스레드로
  합쳐서 돌린 적이 있는데, rclpy는 같은 프로세스 안에서 서로 다른 스레드가 동시에
  spin(같은 컨텍스트를 공유하는 executor를 여러 개 동시 사용)하는 걸 안전하게 지원하지
  않아서 `IndexError: wait set index too big`로 wok_test4 자체가 죽는 버그가 있었다.
  그래서 별도 '프로세스'로 분리한다 — dsr_controller2는 공유 서비스라서, 어느 프로세스가
  move_stop을 호출하든 로봇이 그 즉시 멈춘다(=wok_test4가 movel/movej로 블로킹돼 있어도
  이 프로세스가 독립적으로 move_stop을 호출해 '진짜' 즉시정지를 구현할 수 있다).

  ros2 run wok_exception_handling estop_button_io
"""
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String, Empty
from dsr_msgs2.srv import GetCtrlBoxDigitalInput, MoveStop

ROBOT_ID = "dsr01"

DI_ESTOP = 13     # 정지 버튼
DI_RESUME = 16    # 재개 버튼

# ★ get_ctrl_box_digital_input 서비스는 고속/동시 폴링을 못 버틴다(값이 0으로 밀리거나
#   timeout). 실제로 안정적으로 읽힌 조건은 "index 하나씩, index당 ~2Hz, timeout 1초"였다.
#   그래서 매 사이클 한 index만 읽고(13↔16 번갈아), 사이클을 CYCLE_DT 간격으로 돈다.
CYCLE_DT = 0.25       # 사이클 간격(초). 두 index 번갈아 → index당 실질 ~2Hz
READ_TIMEOUT = 1.0    # 한 번 읽기 대기 최대 시간(초)


class EstopButtonIO(Node):
    def __init__(self):
        super().__init__('estop_button_io', namespace=ROBOT_ID)

        # 컨트롤박스 DI 읽기 서비스 (raw). 상대이름 → /dsr01/io/get_ctrl_box_digital_input
        self._cli = self.create_client(GetCtrlBoxDigitalInput, 'io/get_ctrl_box_digital_input')
        # DI13 감지 즉시 진행 중 동작을 멈추는 서비스. 상대이름 → /dsr01/motion/move_stop
        self._stop_cli = self.create_client(MoveStop, 'motion/move_stop')

        self.estop_pub = self.create_publisher(Bool, '/estop', 10)
        self.alarm_pub = self.create_publisher(String, '/alarm', 10)
        self.resume_pub = self.create_publisher(Empty, '/manual_resume', 10)

        self._prev = {DI_ESTOP: False, DI_RESUME: False}   # 상승엣지 판정용 이전값
        self._last = {DI_ESTOP: -9, DI_RESUME: -9}         # 디버그용 마지막 읽은 값
        self._idx_toggle = 0                                # 사이클마다 읽을 index를 번갈아
        self._hb = 0

        self.get_logger().info(
            f'estop_button_io 시작 — DI{DI_ESTOP}=정지, DI{DI_RESUME}=재개 '
            f'(index당 ~{1.0/(2*CYCLE_DT):.1f}Hz, 번갈아 읽기)'
        )

    def read_di(self, idx):
        """DI 한 채널을 읽어 0/1 반환(실패 시 None). 메인 스레드에서 직접 spin으로 완료시킴."""
        if not self._cli.service_is_ready():
            return None
        req = GetCtrlBoxDigitalInput.Request()
        req.index = idx
        fut = self._cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=READ_TIMEOUT)
        if not fut.done():
            return None
        res = fut.result()
        if res is None or not res.success:
            return None
        return int(res.value)

    def poll_once(self):
        # ★ 한 사이클에 index '하나만' 읽는다(13↔16 번갈아). 이 IO 서비스는 연속으로
        #   두 index를 빠르게 읽으면 값이 밀리므로, di13_track에서 검증된 단일읽기 방식을 따른다.
        idx = DI_ESTOP if self._idx_toggle == 0 else DI_RESUME
        self._idx_toggle ^= 1

        val = self.read_di(idx)
        if val is not None:
            self._last[idx] = val
            pressed = bool(val)
            if pressed and not self._prev[idx]:
                if idx == DI_ESTOP:
                    self.get_logger().warn(f'DI{DI_ESTOP} 버튼 눌림 — move_stop으로 즉시 정지 + 수동 E-STOP')
                    if self._stop_cli.service_is_ready():
                        sreq = MoveStop.Request()
                        sreq.stop_mode = 1  # DR_QSTOP: Quick stop (Stop Category 2)
                        fut = self._stop_cli.call_async(sreq)
                        rclpy.spin_until_future_complete(self, fut, timeout_sec=0.5)
                    self.alarm_pub.publish(String(data='MANUAL_ESTOP'))
                    self.estop_pub.publish(Bool(data=True))
                else:
                    self.get_logger().info(f'DI{DI_RESUME} 버튼 눌림 — 관리자 재개 요청')
                    self.resume_pub.publish(Empty())
            self._prev[idx] = pressed

        # 디버그: 약 2초마다 현재 읽은 값 출력 (동작 확인되면 지워도 됨)
        self._hb += 1
        if self._hb >= int(2.0 / CYCLE_DT):
            self._hb = 0
            self.get_logger().info(
                f'[디버그] DI{DI_ESTOP}={self._last[DI_ESTOP]}, DI{DI_RESUME}={self._last[DI_RESUME]}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = EstopButtonIO()
    try:
        while rclpy.ok():
            node.poll_once()
            time.sleep(CYCLE_DT)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
