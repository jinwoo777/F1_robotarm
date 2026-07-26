"""OnRobot RG2 control over Modbus TCP.

The client communicates directly with an OnRobot Compute Box/Eye Box and uses
only the Python standard library. It can read the current gripper width and
command the RG2 to move to a target width with a target gripping force.
"""

from __future__ import annotations

import math
import socket
import struct
import threading
import time
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from enum import IntEnum
from numbers import Integral
from typing import Optional, Sequence


class RG2Error(Exception):
    """Base exception for this package."""


class RG2ConnectionError(RG2Error):
    """Raised when the Modbus TCP connection cannot be used."""


class ModbusProtocolError(RG2Error):
    """Raised when a malformed or unexpected Modbus response is received."""


class ModbusExceptionResponse(RG2Error):
    """Raised when the Modbus server returns an exception response."""

    EXCEPTION_NAMES = {
        1: "Illegal Function",
        2: "Illegal Data Address",
        3: "Illegal Data Value",
        4: "Server Device Failure",
        5: "Acknowledge",
        6: "Server Device Busy",
        8: "Memory Parity Error",
        10: "Gateway Path Unavailable",
        11: "Gateway Target Device Failed to Respond",
    }

    def __init__(self, function_code: int, exception_code: int) -> None:
        self.function_code = function_code
        self.exception_code = exception_code
        name = self.EXCEPTION_NAMES.get(exception_code, "Unknown Modbus exception")
        super().__init__(
            f"Modbus exception: function=0x{function_code:02X}, "
            f"code=0x{exception_code:02X} ({name})"
        )


class RG2BusyError(RG2Error):
    """Raised when a new motion is requested while the RG2 is already busy."""


class RG2SafetyError(RG2Error):
    """Raised when an RG2 safety fault prevents motion."""

    def __init__(self, status: "RG2Status") -> None:
        self.status = status
        super().__init__(
            "RG2 safety fault is active "
            f"(status=0x{status.raw:04X}); clear the obstruction and power-cycle "
            "the gripper if a safety circuit was triggered"
        )


class RG2MotionTimeout(RG2Error):
    """Raised when a commanded motion does not finish within the timeout."""


class UnitId(IntEnum):
    """Common OnRobot device/unit identifiers."""

    QUICK_CHANGER = 65
    DUAL_PRIMARY = 66
    DUAL_SECONDARY = 67


class GripControl(IntEnum):
    """Values accepted by the RG2 control register."""

    GRIP = 0x0001
    STOP = 0x0008
    GRIP_WITH_OFFSET = 0x0010


@dataclass(frozen=True)
class RG2Status:
    """Decoded RG2 status register (address 0x010C)."""

    raw: int
    busy: bool
    grip_detected: bool
    safety_switch_1_pushed: bool
    safety_switch_1_triggered: bool
    safety_switch_2_pushed: bool
    safety_switch_2_triggered: bool
    safety_error: bool

    @classmethod
    def from_register(cls, raw: int) -> "RG2Status":
        if not 0 <= raw <= 0xFFFF:
            raise ValueError("status register must be an unsigned 16-bit value")
        return cls(
            raw=raw,
            busy=bool(raw & (1 << 0)),
            grip_detected=bool(raw & (1 << 1)),
            safety_switch_1_pushed=bool(raw & (1 << 2)),
            safety_switch_1_triggered=bool(raw & (1 << 3)),
            safety_switch_2_pushed=bool(raw & (1 << 4)),
            safety_switch_2_triggered=bool(raw & (1 << 5)),
            safety_error=bool(raw & (1 << 6)),
        )

    @property
    def has_safety_fault(self) -> bool:
        """Return True when a latched/active safety condition blocks motion."""

        return (
            self.safety_switch_1_pushed
            or self.safety_switch_1_triggered
            or self.safety_switch_2_pushed
            or self.safety_switch_2_triggered
            or self.safety_error
        )


@dataclass(frozen=True)
class RG2MotionCommand:
    """Quantized motion command sent to the gripper."""

    target_width_mm: float
    force_n: float
    include_fingertip_offset: bool
    target_width_raw: int
    force_raw: int
    control_raw: int


@dataclass(frozen=True)
class RG2MotionResult:
    """Result returned after a blocking move finishes."""

    command: RG2MotionCommand
    initial_width_mm: float
    final_width_mm: float
    status: RG2Status
    elapsed_seconds: float
    tolerance_mm: float

    @property
    def target_width_mm(self) -> float:
        """Quantized target width that was sent to the RG2."""

        return self.command.target_width_mm

    @property
    def force_n(self) -> float:
        """Quantized target force that was sent to the RG2."""

        return self.command.force_n

    @property
    def target_reached(self) -> bool:
        """Whether the final width is within the requested tolerance."""

        return abs(self.final_width_mm - self.target_width_mm) <= self.tolerance_mm

    @property
    def reached_target(self) -> bool:
        """Backward-friendly alias for :attr:`target_reached`."""

        return self.target_reached

    @property
    def grip_detected(self) -> bool:
        """Whether the gripper detected an internal or external grip."""

        return self.status.grip_detected

    @property
    def stopped_by_object(self) -> bool:
        """Whether force was detected before the target width was reached."""

        return self.grip_detected and not self.target_reached


class RG2Client:
    """Control an OnRobot RG2 through an OnRobot Modbus TCP gateway.

    Args:
        host: Compute Box/Eye Box IP address.
        port: Modbus TCP port. OnRobot uses 502 by default.
        unit_id: OnRobot device address. Common values are 65 for a single
            Quick Changer, 66 for Dual Quick Changer primary, and 67 for
            Dual Quick Changer secondary.
        timeout: Socket timeout in seconds for each Modbus transaction.

    The client keeps one TCP connection open. This matches OnRobot's documented
    limit of one concurrent Modbus TCP connection.
    """

    READ_HOLDING_REGISTERS = 0x03
    WRITE_SINGLE_REGISTER = 0x06
    WRITE_MULTIPLE_REGISTERS = 0x10

    TARGET_FORCE_REGISTER = 0x0000
    TARGET_WIDTH_REGISTER = 0x0001
    CONTROL_REGISTER = 0x0002

    FINGERTIP_OFFSET_REGISTER = 0x0102
    ACTUAL_WIDTH_REGISTER = 0x010B
    STATUS_REGISTER = 0x010C
    ACTUAL_WIDTH_WITH_OFFSET_REGISTER = 0x0113

    WIDTH_SCALE_MM = 0.1
    FORCE_SCALE_N = 0.1

    MIN_WIDTH_MM = 0.0
    MAX_WIDTH_MM = 110.0
    MIN_FORCE_N = 3.0
    MAX_FORCE_N = 40.0

    def __init__(
        self,
        host: str,
        port: int = 502,
        unit_id: int = int(UnitId.QUICK_CHANGER),
        timeout: float = 1.0,
    ) -> None:
        if not host or not host.strip():
            raise ValueError("host must be a non-empty IP address or hostname")
        if not 1 <= port <= 65535:
            raise ValueError("port must be in the range 1..65535")
        if not 0 <= int(unit_id) <= 255:
            raise ValueError("unit_id must be in the range 0..255")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a finite number greater than zero")

        self.host = host.strip()
        self.port = int(port)
        self.unit_id = int(unit_id)
        self.timeout = float(timeout)

        self._socket: Optional[socket.socket] = None
        self._transaction_id = 0
        self._lock = threading.RLock()

    @property
    def connected(self) -> bool:
        """Return True when a socket is currently open."""

        return self._socket is not None

    def connect(self) -> None:
        """Open the Modbus TCP connection if it is not already open."""

        with self._lock:
            if self._socket is not None:
                return
            try:
                sock = socket.create_connection(
                    (self.host, self.port), timeout=self.timeout
                )
                sock.settimeout(self.timeout)
            except OSError as exc:
                raise RG2ConnectionError(
                    f"Cannot connect to {self.host}:{self.port}: {exc}"
                ) from exc
            self._socket = sock

    def close(self) -> None:
        """Close the Modbus TCP connection."""

        with self._lock:
            sock, self._socket = self._socket, None
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                sock.close()

    def __enter__(self) -> "RG2Client":
        self.connect()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def get_width_mm(self, include_fingertip_offset: bool = True) -> float:
        """Return the current gripper gap in millimetres.

        Args:
            include_fingertip_offset: When True, read register 0x0113 and
                include the fingertip offset stored in the gripper. When False,
                read register 0x010B, which measures between the inner faces of
                the aluminium fingers.
        """

        return self.get_width_raw(include_fingertip_offset) * self.WIDTH_SCALE_MM

    def get_width_raw(self, include_fingertip_offset: bool = True) -> int:
        """Return the raw width register value in 0.1 mm units."""

        register = (
            self.ACTUAL_WIDTH_WITH_OFFSET_REGISTER
            if include_fingertip_offset
            else self.ACTUAL_WIDTH_REGISTER
        )
        return self.read_holding_registers(register, 1)[0]

    def get_fingertip_offset_mm(self) -> float:
        """Return the configured fingertip offset as a signed millimetre value."""

        raw = self.read_holding_registers(self.FINGERTIP_OFFSET_REGISTER, 1)[0]
        signed = raw - 0x10000 if raw & 0x8000 else raw
        return signed * self.WIDTH_SCALE_MM

    def get_status(self) -> RG2Status:
        """Read and decode the RG2 status register."""

        raw = self.read_holding_registers(self.STATUS_REGISTER, 1)[0]
        return RG2Status.from_register(raw)

    def start_move_to_width(
        self,
        target_width_mm: float,
        force_n: float = 20.0,
        *,
        include_fingertip_offset: bool = True,
        check_ready: bool = True,
    ) -> RG2MotionCommand:
        """Start a non-blocking motion to ``target_width_mm``.

        The target width and force are written together with the control value
        using Modbus function code 0x10. The returned command contains the
        values quantized to the RG2's 0.1 mm / 0.1 N register resolution.

        Args:
            target_width_mm: Desired finger gap, 0.0 to 110.0 mm.
            force_n: Desired gripping force, 3.0 to 40.0 N.
            include_fingertip_offset: When True, target width is interpreted
                between the configured fingertip contact surfaces. When False,
                it is interpreted between the aluminium finger faces.
            check_ready: Read status first and raise if the gripper is busy or
                a safety fault is active.
        """

        command = self._build_motion_command(
            target_width_mm,
            force_n,
            include_fingertip_offset=include_fingertip_offset,
        )

        if check_ready:
            self._ensure_ready_for_motion(self.get_status())

        self._send_motion_command(command)
        return command

    def move_to_width(
        self,
        target_width_mm: float,
        force_n: float = 20.0,
        *,
        include_fingertip_offset: bool = True,
        motion_timeout: float = 5.0,
        poll_interval: float = 0.05,
        tolerance_mm: float = 0.5,
        stop_on_timeout: bool = True,
    ) -> RG2MotionResult:
        """Move to a target width and wait until the motion finishes.

        A motion can finish either because the target width was reached or
        because the target force was reached against a workpiece. Check
        ``result.target_reached`` and ``result.grip_detected`` to distinguish
        the two cases.

        Args:
            target_width_mm: Desired finger gap, 0.0 to 110.0 mm.
            force_n: Desired gripping force, 3.0 to 40.0 N.
            include_fingertip_offset: Interpret target and measured width with
                the configured fingertip offset when True.
            motion_timeout: Maximum time to wait for completion, in seconds.
            poll_interval: Status/width polling period, in seconds.
            tolerance_mm: Width error accepted as target reached.
            stop_on_timeout: Best-effort STOP command before raising timeout.
        """

        self._validate_positive_finite(motion_timeout, "motion_timeout")
        self._validate_positive_finite(poll_interval, "poll_interval")
        if not math.isfinite(tolerance_mm) or tolerance_mm < 0:
            raise ValueError("tolerance_mm must be a finite number >= 0")

        command = self._build_motion_command(
            target_width_mm,
            force_n,
            include_fingertip_offset=include_fingertip_offset,
        )
        initial_status = self.get_status()
        self._ensure_ready_for_motion(initial_status)
        initial_width = self.get_width_mm(include_fingertip_offset)

        started_at = time.monotonic()
        self._send_motion_command(command)
        deadline = started_at + motion_timeout

        saw_busy = False
        stable_idle_samples = 0
        last_width = initial_width

        while True:
            status = self.get_status()
            self._raise_if_safety_fault(status)
            width = self.get_width_mm(include_fingertip_offset)
            now = time.monotonic()

            if status.busy:
                saw_busy = True
                stable_idle_samples = 0
            else:
                target_reached = (
                    abs(width - command.target_width_mm) <= tolerance_mm
                )
                width_changed = (
                    abs(width - initial_width) >= self.WIDTH_SCALE_MM
                )
                grip_event = (
                    status.grip_detected and not initial_status.grip_detected
                )

                # Normal completion after observing BUSY, or an extremely fast
                # command that completed before the first status poll.
                if saw_busy or target_reached or grip_event:
                    return RG2MotionResult(
                        command=command,
                        initial_width_mm=initial_width,
                        final_width_mm=width,
                        status=status,
                        elapsed_seconds=now - started_at,
                        tolerance_mm=tolerance_mm,
                    )

                # Some firmware/motion combinations can complete between polls
                # without BUSY being observed. A changed width that remains
                # stable for two idle samples is treated as completed.
                if width_changed and abs(width - last_width) <= self.WIDTH_SCALE_MM:
                    stable_idle_samples += 1
                    if stable_idle_samples >= 2:
                        return RG2MotionResult(
                            command=command,
                            initial_width_mm=initial_width,
                            final_width_mm=width,
                            status=status,
                            elapsed_seconds=now - started_at,
                            tolerance_mm=tolerance_mm,
                        )
                else:
                    stable_idle_samples = 0

            if now >= deadline:
                stop_error: Optional[Exception] = None
                if stop_on_timeout:
                    try:
                        self.stop()
                    except RG2Error as exc:
                        stop_error = exc

                message = (
                    f"RG2 motion to {command.target_width_mm:.1f} mm did not "
                    f"finish within {motion_timeout:.3f} s"
                )
                if stop_error is not None:
                    message += f"; STOP command also failed: {stop_error}"
                raise RG2MotionTimeout(message)

            last_width = width
            remaining = deadline - now
            time.sleep(min(poll_interval, max(0.0, remaining)))

    def wait_until_idle(
        self,
        timeout: float = 5.0,
        poll_interval: float = 0.05,
    ) -> RG2Status:
        """Wait until the gripper's BUSY bit is low and return final status."""

        self._validate_positive_finite(timeout, "timeout")
        self._validate_positive_finite(poll_interval, "poll_interval")
        deadline = time.monotonic() + timeout

        while True:
            status = self.get_status()
            self._raise_if_safety_fault(status)
            if not status.busy:
                return status
            now = time.monotonic()
            if now >= deadline:
                raise RG2MotionTimeout(
                    f"RG2 remained busy for longer than {timeout:.3f} s"
                )
            time.sleep(min(poll_interval, max(0.0, deadline - now)))

    def stop(self) -> None:
        """Request an immediate stop of the current RG2 motion."""

        self.write_single_register(self.CONTROL_REGISTER, int(GripControl.STOP))

    def read_holding_registers(self, address: int, count: int = 1) -> Sequence[int]:
        """Read one or more holding registers using function code 0x03."""

        self._validate_address(address)
        if isinstance(count, bool) or not isinstance(count, Integral):
            raise TypeError("count must be an integer")
        if not 1 <= int(count) <= 125:
            raise ValueError("count must be in the range 1..125")

        request_pdu = struct.pack(
            ">BHH", self.READ_HOLDING_REGISTERS, address, count
        )
        response_pdu = self._transact(request_pdu)

        if len(response_pdu) < 2:
            raise ModbusProtocolError("truncated register response")
        byte_count = response_pdu[1]
        expected_byte_count = count * 2
        if byte_count != expected_byte_count:
            raise ModbusProtocolError(
                "register byte count mismatch: "
                f"expected {expected_byte_count}, got {byte_count}"
            )
        if len(response_pdu) != 2 + byte_count:
            raise ModbusProtocolError(
                f"PDU length mismatch: expected {2 + byte_count}, "
                f"got {len(response_pdu)}"
            )

        return struct.unpack(f">{count}H", response_pdu[2:])

    def write_single_register(self, address: int, value: int) -> None:
        """Write one holding register using function code 0x06."""

        self._validate_address(address)
        self._validate_register_value(value)

        request_pdu = struct.pack(
            ">BHH", self.WRITE_SINGLE_REGISTER, address, int(value)
        )
        response_pdu = self._transact(request_pdu)
        if len(response_pdu) != 5:
            raise ModbusProtocolError(
                f"write-single response length must be 5, got {len(response_pdu)}"
            )
        _, response_address, response_value = struct.unpack(">BHH", response_pdu)
        if response_address != address or response_value != int(value):
            raise ModbusProtocolError(
                "write-single echo mismatch: "
                f"expected address=0x{address:04X}, value=0x{int(value):04X}; "
                f"got address=0x{response_address:04X}, "
                f"value=0x{response_value:04X}"
            )

    def write_multiple_registers(
        self, address: int, values: Sequence[int]
    ) -> None:
        """Write consecutive holding registers using function code 0x10."""

        self._validate_address(address)
        registers = tuple(values)
        if not 1 <= len(registers) <= 123:
            raise ValueError("values must contain 1..123 registers")
        if address + len(registers) - 1 > 0xFFFF:
            raise ValueError("register range exceeds address 0xFFFF")
        for value in registers:
            self._validate_register_value(value)

        count = len(registers)
        payload = struct.pack(f">{count}H", *(int(value) for value in registers))
        request_pdu = struct.pack(
            ">BHHB",
            self.WRITE_MULTIPLE_REGISTERS,
            address,
            count,
            len(payload),
        ) + payload
        response_pdu = self._transact(request_pdu)

        if len(response_pdu) != 5:
            raise ModbusProtocolError(
                f"write-multiple response length must be 5, got {len(response_pdu)}"
            )
        _, response_address, response_count = struct.unpack(">BHH", response_pdu)
        if response_address != address or response_count != count:
            raise ModbusProtocolError(
                "write-multiple acknowledgement mismatch: "
                f"expected address=0x{address:04X}, count={count}; "
                f"got address=0x{response_address:04X}, count={response_count}"
            )

    def _build_motion_command(
        self,
        target_width_mm: float,
        force_n: float,
        *,
        include_fingertip_offset: bool,
    ) -> RG2MotionCommand:
        target_width_raw = self._quantize_to_register(
            target_width_mm,
            scale=self.WIDTH_SCALE_MM,
            minimum=self.MIN_WIDTH_MM,
            maximum=self.MAX_WIDTH_MM,
            name="target_width_mm",
        )
        force_raw = self._quantize_to_register(
            force_n,
            scale=self.FORCE_SCALE_N,
            minimum=self.MIN_FORCE_N,
            maximum=self.MAX_FORCE_N,
            name="force_n",
        )
        control = (
            GripControl.GRIP_WITH_OFFSET
            if include_fingertip_offset
            else GripControl.GRIP
        )
        return RG2MotionCommand(
            target_width_mm=target_width_raw * self.WIDTH_SCALE_MM,
            force_n=force_raw * self.FORCE_SCALE_N,
            include_fingertip_offset=include_fingertip_offset,
            target_width_raw=target_width_raw,
            force_raw=force_raw,
            control_raw=int(control),
        )

    def _send_motion_command(self, command: RG2MotionCommand) -> None:
        # Force, width, and control are adjacent. A single FC16 request ensures
        # the control trigger is delivered with the matching force/width pair.
        self.write_multiple_registers(
            self.TARGET_FORCE_REGISTER,
            (command.force_raw, command.target_width_raw, command.control_raw),
        )

    @staticmethod
    def _ensure_ready_for_motion(status: RG2Status) -> None:
        RG2Client._raise_if_safety_fault(status)
        if status.busy:
            raise RG2BusyError(
                "RG2 is already moving; wait until status.busy is False or call stop()"
            )

    @staticmethod
    def _raise_if_safety_fault(status: RG2Status) -> None:
        if status.has_safety_fault:
            raise RG2SafetyError(status)

    @staticmethod
    def _quantize_to_register(
        value: float,
        *,
        scale: float,
        minimum: float,
        maximum: float,
        name: str,
    ) -> int:
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{name} must be finite")
        if not minimum <= numeric <= maximum:
            raise ValueError(
                f"{name} must be in the range {minimum:.1f}..{maximum:.1f}"
            )
        # Decimal(str(...)) avoids binary floating-point edge cases at exact
        # half steps while keeping a predictable ROUND_HALF_UP policy.
        scaled = Decimal(str(numeric)) / Decimal(str(scale))
        return int(scaled.to_integral_value(rounding=ROUND_HALF_UP))

    @staticmethod
    def _validate_positive_finite(value: float, name: str) -> None:
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite number greater than zero")

    @staticmethod
    def _validate_address(address: int) -> None:
        if isinstance(address, bool) or not isinstance(address, Integral):
            raise TypeError("address must be an integer")
        if not 0 <= int(address) <= 0xFFFF:
            raise ValueError("address must be in the range 0..65535")

    @staticmethod
    def _validate_register_value(value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("register value must be an integer")
        if not 0 <= int(value) <= 0xFFFF:
            raise ValueError("register value must be in the range 0..65535")

    def _transact(self, request_pdu: bytes) -> bytes:
        if not request_pdu:
            raise ValueError("request_pdu must not be empty")

        with self._lock:
            self.connect()
            sock = self._socket
            if sock is None:  # Defensive; connect() either sets it or raises.
                raise RG2ConnectionError("connection is not open")

            transaction_id = self._next_transaction_id()
            request = struct.pack(
                ">HHHB",
                transaction_id,
                0,  # Protocol identifier: Modbus
                1 + len(request_pdu),  # Unit ID + PDU
                self.unit_id,
            ) + request_pdu

            try:
                sock.sendall(request)
                mbap = self._recv_exact(sock, 7)
                (
                    response_transaction_id,
                    protocol_id,
                    length,
                    response_unit_id,
                ) = struct.unpack(">HHHB", mbap)

                if not 2 <= length <= 260:
                    raise ModbusProtocolError(
                        f"invalid Modbus length field: {length}"
                    )
                response_pdu = self._recv_exact(sock, length - 1)
            except ModbusProtocolError:
                self.close()
                raise
            except (OSError, EOFError) as exc:
                self.close()
                raise RG2ConnectionError(
                    f"Communication with {self.host}:{self.port} failed: {exc}"
                ) from exc

            try:
                if response_transaction_id != transaction_id:
                    raise ModbusProtocolError(
                        "transaction ID mismatch: "
                        f"expected {transaction_id}, got {response_transaction_id}"
                    )
                if protocol_id != 0:
                    raise ModbusProtocolError(
                        f"unexpected protocol identifier: {protocol_id}"
                    )
                if response_unit_id != self.unit_id:
                    raise ModbusProtocolError(
                        f"unit ID mismatch: expected {self.unit_id}, "
                        f"got {response_unit_id}"
                    )
                if not response_pdu:
                    raise ModbusProtocolError("empty Modbus PDU")

                request_function = request_pdu[0]
                response_function = response_pdu[0]
                if response_function & 0x80:
                    if len(response_pdu) < 2:
                        raise ModbusProtocolError(
                            "truncated Modbus exception response"
                        )
                    raise ModbusExceptionResponse(
                        response_function & 0x7F, response_pdu[1]
                    )
                if response_function != request_function:
                    raise ModbusProtocolError(
                        "unexpected function code: "
                        f"expected 0x{request_function:02X}, "
                        f"got 0x{response_function:02X}"
                    )
            except ModbusProtocolError:
                self.close()
                raise

            return response_pdu

    def _next_transaction_id(self) -> int:
        self._transaction_id = (self._transaction_id + 1) & 0xFFFF
        return self._transaction_id

    @staticmethod
    def _recv_exact(sock: socket.socket, size: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < size:
            chunk = sock.recv(size - len(chunks))
            if not chunk:
                raise EOFError(
                    f"connection closed after {len(chunks)} of {size} bytes"
                )
            chunks.extend(chunk)
        return bytes(chunks)
