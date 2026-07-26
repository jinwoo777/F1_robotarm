"""Command-line interface for :mod:`onrobot_rg2`."""

from __future__ import annotations

import argparse
import sys
import time

from .client import RG2Client, RG2Error, UnitId


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read or control an OnRobot RG2 gripper through an OnRobot "
            "Compute Box/Eye Box using Modbus TCP."
        )
    )
    parser.add_argument("host", help="Compute Box/Eye Box IP address")
    parser.add_argument("--port", type=int, default=502, help="Modbus TCP port")
    parser.add_argument(
        "--unit-id",
        type=int,
        default=int(UnitId.QUICK_CHANGER),
        help="65=single QC, 66=dual primary, 67=dual secondary",
    )
    parser.add_argument(
        "--timeout", type=float, default=1.0, help="socket timeout in seconds"
    )
    parser.add_argument(
        "--without-offset",
        action="store_true",
        help=(
            "measure/command between aluminium finger faces instead of the "
            "configured fingertip contact surfaces"
        ),
    )
    parser.add_argument(
        "--watch",
        type=float,
        metavar="SECONDS",
        help="read the current width repeatedly at this interval",
    )
    parser.add_argument(
        "--target-width",
        type=float,
        metavar="MM",
        help="move the gripper to this target gap in millimetres",
    )
    parser.add_argument(
        "--force",
        type=float,
        default=20.0,
        metavar="N",
        help="target gripping force for movement (default: 20 N)",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="send the target command and return without waiting for completion",
    )
    parser.add_argument(
        "--motion-timeout",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="maximum wait for motion completion (default: 5.0)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.05,
        metavar="SECONDS",
        help="status polling interval while waiting (default: 0.05)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.5,
        metavar="MM",
        help="target-width tolerance (default: 0.5 mm)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    include_offset = not args.without_offset

    if args.target_width is not None and args.watch is not None:
        parser.error("--target-width and --watch cannot be used together")
    if args.target_width is None and args.no_wait:
        parser.error("--no-wait requires --target-width")

    try:
        with RG2Client(
            host=args.host,
            port=args.port,
            unit_id=args.unit_id,
            timeout=args.timeout,
        ) as gripper:
            if args.target_width is not None:
                if args.no_wait:
                    command = gripper.start_move_to_width(
                        args.target_width,
                        force_n=args.force,
                        include_fingertip_offset=include_offset,
                    )
                    print(
                        "command accepted: "
                        f"target={command.target_width_mm:.1f} mm, "
                        f"force={command.force_n:.1f} N"
                    )
                    return 0

                result = gripper.move_to_width(
                    args.target_width,
                    force_n=args.force,
                    include_fingertip_offset=include_offset,
                    motion_timeout=args.motion_timeout,
                    poll_interval=args.poll_interval,
                    tolerance_mm=args.tolerance,
                )
                print(
                    f"target={result.command.target_width_mm:.1f} mm, "
                    f"final={result.final_width_mm:.1f} mm, "
                    f"target_reached={result.target_reached}, "
                    f"grip_detected={result.grip_detected}, "
                    f"elapsed={result.elapsed_seconds:.3f} s"
                )
                return 0

            if args.watch is None:
                print(f"{gripper.get_width_mm(include_offset):.1f} mm")
                return 0

            if args.watch <= 0:
                raise ValueError("--watch must be greater than zero")

            while True:
                width_mm = gripper.get_width_mm(include_offset)
                print(f"{width_mm:.1f} mm", flush=True)
                time.sleep(args.watch)
    except KeyboardInterrupt:
        return 130
    except (RG2Error, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
