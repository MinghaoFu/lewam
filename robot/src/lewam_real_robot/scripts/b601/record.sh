#!/usr/bin/env bash
# Record a LeRobot dataset by teleoperating the reBot B601-DM follower with the reBot
# Arm 102 leader. Detects and saves the USB ports, resets the control mode, then runs
# record.py, a Hydra app with one config per dataset under configs/:
#   ./src/lewam_real_robot/scripts/b601/record.sh --config-name=<dataset> [key=value ...]
#
# --help         show the resolved config
# --recalibrate  re-run zero-pose calibration for the follower and leader first
# --update-mode  reset the follower's control mode first (after a crash)
# --dry-run      teleoperate under the config without writing a dataset
set -euo pipefail

RECALIBRATE=0
UPDATE_MODE=0
# Hydra's CLI requires its own flags before any key=value override, or it errors
# with "unrecognized arguments". Split them so callers can type either order;
# a space-separated `--config-name foo` keeps its value glued to it.
HYDRA_FLAGS=()
OVERRIDE_ARGS=()
args=("$@")
i=0
while [[ $i -lt ${#args[@]} ]]; do
    arg="${args[$i]}"
    case "$arg" in
        --recalibrate) RECALIBRATE=1 ;;
        --update-mode) UPDATE_MODE=1 ;;
        # passed to Hydra as an override, since its CLI rejects any --flag it does not define
        --dry-run) OVERRIDE_ARGS+=("dry_run=true") ;;
        --config-name)
            i=$((i + 1))
            HYDRA_FLAGS+=("$arg" "${args[$i]}")
            ;;
        --*) HYDRA_FLAGS+=("$arg") ;;
        *) OVERRIDE_ARGS+=("$arg") ;;
    esac
    i=$((i + 1))
done

# fail fast before touching the venv, ports or motors: repo_id has no default
HAS_CONFIG=0
for arg in "${HYDRA_FLAGS[@]}" "${OVERRIDE_ARGS[@]}"; do
    case "$arg" in
        --config-name=*|--config-name|repo_id=*) HAS_CONFIG=1 ;;
    esac
done
if [[ "$HAS_CONFIG" -eq 0 ]]; then
    echo "Missing dataset config: pass --config-name=<dataset> (see src/lewam_real_robot/scripts/b601/configs/) or repo_id=<hf_username>/<dataset_name> directly." >&2
    echo "Usage: ./src/lewam_real_robot/scripts/b601/record.sh --config-name=<dataset> [key=value ...]" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate

# read by `datasets` at import time, and it outranks disable_progress_bars(),
# which the Hub client undoes while resuming a dataset
export HF_DATASETS_DISABLE_PROGRESS_BARS=1

ENV_FILE="$REPO_ROOT/src/lewam_real_robot/scripts/b601/.env"
CALLER_NAME="record.sh"
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
# the wrong mode by an earlier run makes connect() fail
if [[ "$UPDATE_MODE" -eq 1 ]]; then
    echo "Resetting follower control modes..."
    python src/lewam_real_robot/scripts/b601/set_control_mode.py --port "$FOLLOWER_PORT" --mode "$CONTROL_MODE"
    echo
fi

if [[ "$RECALIBRATE" -eq 1 ]]; then
    echo "=== Recalibration requested ==="
    recalibrate_device robot rebot_b601_follower robots \
        "$FOLLOWER_PORT" "$FOLLOWER_ID" \
        "--robot.can_adapter=damiao --robot.control_mode=$CONTROL_MODE"
    recalibrate_device teleop rebot_102_leader teleoperators \
        "$LEADER_PORT" "$LEADER_ID" ""
    echo
fi

python src/lewam_real_robot/scripts/b601/record.py \
    "${HYDRA_FLAGS[@]}" \
    follower_port="$FOLLOWER_PORT" \
    leader_port="$LEADER_PORT" \
    follower_id="$FOLLOWER_ID" \
    leader_id="$LEADER_ID" \
    control_mode="$CONTROL_MODE" \
    "${OVERRIDE_ARGS[@]}"
