#!/usr/bin/env bash
# Wait for each extended GIP arm to finish training, then run its eval matrix.
# warm-start (gip_*) = 40 ep -> bc/guided/planning ; from-scratch (gip_scratch_*) = 60 ep -> bc/guided
cd /home/minghao.fu/workspace/le-wm-repro
LOGD=/mnt/data_nvme1/minghao.fu/le-wm-repro-logs
OUT=$LOGD/gip_extended_eval.log

wait_pids(){ for e in "$@"; do while ps -p "$(cat /tmp/$e.pid 2>/dev/null)" >/dev/null 2>&1; do sleep 120; done; done; }

wait_pids gip_pusht gip_tworoom gip_reacher
echo "WARM(40ep) TRAINING DONE $(date -Iseconds)" >> "$OUT"
for e in pusht tworoom reacher; do
  bash run_gip_evals.sh 6 "$e" "gip_$e" 50 '' 'bc guided planning' >> "$OUT" 2>&1
done
echo WARM_EVAL_DONE >> "$OUT"

wait_pids gip_scratch_pusht gip_scratch_tworoom gip_scratch_reacher
echo "SCRATCH(60ep) TRAINING DONE $(date -Iseconds)" >> "$OUT"
for e in pusht tworoom reacher; do
  bash run_gip_evals.sh 0 "$e" "gip_scratch_$e" 50 '' 'bc guided' >> "$OUT" 2>&1
done
echo "ALL_EXTENDED_EVAL_DONE $(date -Iseconds)" >> "$OUT"
touch /tmp/gip_extended_eval_done
