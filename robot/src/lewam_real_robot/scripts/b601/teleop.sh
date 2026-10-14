#!/usr/bin/env bash
# Teleoperate the reBot B601-DM follower with the reBot Arm 102 leader.
#
# Usage:   ./src/lewam_real_robot/scripts/b601/teleop.sh [--recalibrate]
#
# On first run (or whenever the saved ports don't exist anymore), this walks
# you through lerobot-find-port once per arm and saves the detected ports to
# src/lewam_real_robot/scripts/b601/.env, so future runs skip detection entirely.
#
# --recalibrate  Re-run zero-pose calibration for the follower and the leader
#                before teleop starts (move each arm to its zero pose when
#                prompted). Overwrites the saved calibration files.
# --profile-log=PATH  Write a per-step timing breakdown (get_observation, teleop
#                read, send_action, ...) for every teleop_loop() iteration to
#                PATH, comparable to record.py's --profile-log output.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate

RECALIBRATE=0
PROFILE_LOG=""
for arg in "$@"; do
    case "$arg" in
        --recalibrate) RECALIBRATE=1 ;;
        --profile-log=*) PROFILE_LOG="${arg#--profile-log=}" ;;
        *) echo "Unknown argument: $arg (usage: teleop.sh [--recalibrate] [--profile-log=PATH])" >&2; exit 1 ;;
    esac
done
if [[ -n "$PROFILE_LOG" ]]; then
    export LEROBOT_PROFILE_LOG="$PROFILE_LOG"
fi

ENV_FILE="$REPO_ROOT/src/lewam_real_robot/scripts/b601/.env"
CALLER_NAME="teleop.sh"
source "$REPO_ROOT/src/lewam_real_robot/scripts/b601/common.sh"

FOLLOWER_ID="${FOLLOWER_ID:-b601_follower}"
LEADER_ID="${LEADER_ID:-b601_leader}"
CONTROL_MODE="${CONTROL_MODE:-mit}"

resolve_ports

echo
echo "Follower: $FOLLOWER_PORT (id=$FOLLOWER_ID)"
echo "Leader:   $LEADER_PORT (id=$LEADER_ID)"
echo "Control mode: $CONTROL_MODE"
echo

# lerobot's ensure_mode does not persist a CTRL_MODE change, so a motor left in
# the wrong mode by an earlier run makes connect() fail. Force + store the modes
# the follower's configure() expects before starting teleop.
echo "Resetting follower control modes..."
python src/lewam_real_robot/scripts/b601/set_control_mode.py --port "$FOLLOWER_PORT" --mode "$CONTROL_MODE"
echo

if [[ "$RECALIBRATE" -eq 1 ]]; then
    echo "=== Recalibration requested ==="
    recalibrate_device robot rebot_b601_follower robots \
        "$FOLLOWER_PORT" "$FOLLOWER_ID" \
        "--robot.can_adapter=damiao --robot.control_mode=$CONTROL_MODE"
    recalibrate_device teleop rebot_102_leader teleoperators \
        "$LEADER_PORT" "$LEADER_ID" ""
    echo
fi

lerobot-teleoperate \
    --robot.type=rebot_b601_follower \
    --robot.port="$FOLLOWER_PORT" \
    --robot.id="$FOLLOWER_ID" \
    --robot.can_adapter=damiao \
    --robot.control_mode="$CONTROL_MODE" \
    --teleop.type=rebot_102_leader \
    --teleop.port="$LEADER_PORT" \
    --teleop.id="$LEADER_ID"
