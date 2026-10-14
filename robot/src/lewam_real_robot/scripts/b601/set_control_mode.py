#!/usr/bin/env python
"""Force the B601 follower motors into the correct Damiao control mode.

lerobot's ensure_mode() writes the runtime CTRL_MODE register but the change is
not persisted, so a motor that has a different mode saved (e.g. POS_VEL left over
from an earlier run) stays in that mode and connect() fails with
"control mode verify failed". This writes CTRL_MODE directly and persists it with
store_parameters, matching what lerobot's configure() expects to find.

Arm joints go to MIT or POS_VEL (per --mode); the gripper goes to FORCE_POS or
MIT (per --gripper-mode), matching the follower's configure() logic.

Usage:
  python scripts/b601/set_control_mode.py --port /dev/tty.usbmodemXXXX --mode mit
"""

from __future__ import annotations

import argparse
import time

from motorbridge import (
    Controller,
    MODE_FORCE_POS,
    MODE_MIT,
    MODE_POS_VEL,
    RID_CTRL_MODE,
)

CAN_IDS = {
    "shoulder_pan": (0x01, 0x11),
    "shoulder_lift": (0x02, 0x12),
    "elbow_flex": (0x03, 0x13),
    "wrist_flex": (0x04, 0x14),
    "wrist_yaw": (0x05, 0x15),
    "wrist_roll": (0x06, 0x16),
    "gripper": (0x07, 0x17),
}
MODELS = {
    "shoulder_pan": "4340P",
    "shoulder_lift": "4340P",
    "elbow_flex": "4340P",
    "wrist_flex": "4310",
    "wrist_yaw": "4310",
    "wrist_roll": "4310",
    "gripper": "4310",
}
ARM_MODES = {"mit": MODE_MIT, "pos_vel": MODE_POS_VEL}
GRIPPER_MODES = {"force_pos": MODE_FORCE_POS, "mit": MODE_MIT}


def set_mode(motor, name: str, target: int) -> None:
    motor.disable()
    time.sleep(0.3)
    mode = None
    for _ in range(5):
        try:
            mode = motor.damiao_get_param_u32(RID_CTRL_MODE)
        except Exception:
            mode = None
        if mode == target:
            break
        motor.disable()
        time.sleep(0.3)
        motor.damiao_write_param_u32(RID_CTRL_MODE, target)
        motor.store_parameters()
        time.sleep(0.4)
    if mode != target:
        raise RuntimeError(f"{name}: could not set mode {target} (stuck at {mode})")
    print(f"  {name}: mode {mode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--mode", choices=list(ARM_MODES), default="mit")
    ap.add_argument("--gripper-mode", choices=list(GRIPPER_MODES), default="force_pos")
    args = ap.parse_args()

    arm_target = ARM_MODES[args.mode]
    gripper_target = GRIPPER_MODES[args.gripper_mode]

    controller = Controller.from_dm_serial(serial_port=args.port, baud=args.baud)
    motors = {n: controller.add_damiao_motor(s, r, MODELS[n]) for n, (s, r) in CAN_IDS.items()}
    controller.disable_all()
    time.sleep(0.5)

    print(f"Setting arm joints to {args.mode}, gripper to {args.gripper_mode}...")
    try:
        for name, motor in motors.items():
            target = gripper_target if name == "gripper" else arm_target
            set_mode(motor, name, target)
    finally:
        try:
            controller.disable_all()
        except Exception:
            pass
        for motor in motors.values():
            try:
                motor.disable()
                motor.close()
            except Exception:
                pass
        try:
            controller.close()
        except Exception:
            pass
    print("Control modes set.")


if __name__ == "__main__":
    main()
