# Merlin / Arnold — how to submit a GPU job (battle-tested)

Everything here was actually run during the LeWAM campaigns (June–July 2026). Two paths:
**A** interactive worker (debugging), **B** `mlx job submitv2` (unattended — the default).

## Path B: unattended job (use this)

```bash
mlx job submitv2 -p my_job.yaml        # -p is REQUIRED unless the file is ./mlx_config.yaml
```

Working YAML template (this exact shape launched merged_wam/baseline_wam; adjust gpu count + caption):

```yaml
caption: 'my experiment'
jobDefVersion:
  entrypointMode: FULL_SCRIPT
  gitRepo: {mnt: /opt/tiger/lewam_job}
  imageMeta: {imageSource: url, imageUrl: aliyun-va-hub.byted.org/arnold/modelchef-gpu:1.0.0.48}
  lazyDownloadCode: false
  name: 'my experiment'
  resource:
    arnoldConfig:
      clusterId: 4
      elasticTraining: {}
      groupIds: [62]                    # numeric id ONLY here; CLI flags use the NAME bi_algorithm
      keepMins: 30
      preemptible: false
      quotaPool: default
      hdfsVolumes:
        - {accessMode: RW, mnt: /mnt/hdfs/byte_ad_audit/bi_algorithm, path: hdfs://harunava/home/byte_ad_audit/bi_algorithm}
        - {accessMode: RO, mnt: /mnt/hdfs/bi_algo_a2f, path: hdfs://harunava/home/byte_arnold_va_ssd/bi_algo_a2f}
      roles:
        - {cpu: 48, gpu: 4, gpuv: A100-SXM-80GB, memory: 400000, name: worker, num: 1, ports: 1,
           queueName: compute-334-aliyun.va-cloudnative-ai-bi.algorithm-guarantee}
    backend: ARNOLD
jobRunParams:
  entrypointFullScript: |
    set -ex
    mkdir -p /opt/tiger/lewam_job && cd /opt/tiger/lewam_job
    cp /mnt/hdfs/byte_ad_audit/bi_algorithm/minghao.fu/lewam/code/my_entry.sh .
    bash my_entry.sh
  envsList: {}
namespace: /user/minghao.fu
```

## Entry-script pattern (survives everything)

See `merlin/example_entry.sh` (= the real `merged_wam_v2.sh`). The load-bearing pieces:

1. **Code**: keep a repo tarball on HDFS (`code/lewam_repo_current.tar.gz`), untar into the pod,
   `export PYTHONPATH`. Refresh the tarball before submitting:
   `cd ~/lewam && tar czf /tmp/t.tgz lewam scripts configs requirements.txt pyproject.toml all_tasks.json && cp /tmp/t.tgz <HDFS>/code/lewam_repo_current.tar.gz`
2. **Data**: symlink the pre-decompressed SSD `.h5` (HDD fallback). NEVER decompress.
3. **Per-task heartbeat FILES** (`hb_<task>.log`), not one shared file — concurrent appends to a
   single HDFS-fuse file lose lines.
4. **Background checkpoint sync** every 5 min to HDFS (`ckpts_live/`) — pod-local checkpoints are
   lost on kill; with the sync loop you can always eval the latest synced epoch.
5. **Failure isolation**: run tasks as `run_task ... &` + `wait`, each returning 0 on failure so one
   task crashing never disturbs the others.

## Monitoring (what actually works)

- `mlx job list | grep <id8>` — status (STARTED/RUNNING/STOPPED/DONE). This is the reliable signal.
- `mlx job log/describe <id>` — **broken from this CLI** (websocket); do not rely on it.
  Monitor through your own HDFS heartbeats + `ckpts_live/` epoch files instead.
- `mlx job` has no kill subcommand, but formal jobs CAN be stopped from the devbox:
  `yes | merlin-cli --control-plane i18n-tt job-v2 runs stop --json '{"sid":"<mlx job id>","stop_reason":"<why>"}'`
  (status becomes `killed` within a minute; confirm with `mlx job get <id>`). The web UI works too.

## Runtime facts (measured)

- **No hard 4h wall** — jobs have run 6.5–7.6h fine. (An early 4h stop was coincidence.)
- **HDFS-fuse is the bottleneck** for per-step DataLoader reads (~7–10ms/read): per-epoch time
  scales with dataset size; page cache warms after epoch 1. RAM-preload trainers
  (`train_lewam_gc.py`) avoid this but need ~1×dataset of CPU RAM (reacher/cube ≈ 100GB each;
  max 2 big preloads per 384GB node).
- CPU contention matters: 4 training tasks + evals on one pod starve DataLoader workers.

## Path A: interactive worker (debug only)

```bash
mlx worker launch --type A100-SXM-80GB --gpu 1 --resourcetype arnold \
  --usergroup bi_algorithm --cluster cloudnative-maliva \
  --queuename compute-334-aliyun.va-cloudnative-ai-bi.algorithm-guarantee -- bash
# mlx worker list / login <id> / kill <id>
```
Needs a real TTY: launched from a background/no-TTY context it dies at "Login Worker".

## Common errors

| error | fix |
|---|---|
| `user is not in group XX` | `--usergroup bi_algorithm` (name, not number) |
| `gpu type X is not available for this cluster` | add `--resourcetype arnold` |
| `read mlx_config.yaml ... failed` | pass `-p your.yaml` |
| queue stuck pending | try Path A worker, or the H100 GCP queue |

## Job-launch hard rules — learned the hard way (2026-07-21, a whole session lost to these)

**1. `mlx job submitv2` jobs are killable, but only through merlin-cli.**
- `mlx job` has NO stop/kill/cancel/delete subcommand and Path-B pods do NOT appear in `mlx worker list`. The
  stop is `yes | merlin-cli --control-plane i18n-tt job-v2 runs stop --json '{"sid":"<mlx job id>","stop_reason":"<why>"}'`
  (see Monitoring above). Kill only your own superseded or wrong jobs.
- **Still: never launch casually.** Every submitv2 job is a contended H100 committed for hours, and a killed job
  leaves stale locks and half-written outputs behind. Think the whole pipeline through BEFORE submitting. Read
  logs/checkpoints to diagnose FIRST; launch only when you know exactly what the result buys.

**2. Build instrumentation INTO the entry script before the first launch.**
- Mirror the training stdout to HDFS DURING training (background `cp /tmp/train.log $CK/train_$TAG.log` every ~60s), not only at the end. Otherwise you fly blind on multi-hour runs — no loss curves, can't tell training from hung. (I launched five 5h jobs with metrics synced only at the end. Inexcusable.)
- Sync `latest.pt` periodically, not just `best.pt`-on-improvement — best.pt frequently freezes on an early checkpoint (see rule 6), so if HDFS only has best.pt you cannot eval the actual current model.
- Emit a per-10-epoch heartbeat (`grep "ep N/EP" log | say`). Estimate epoch time from the REAL first-epoch line, never a guess (I assumed 6 min/ep; it was ~25).

**3. Logging hygiene: one job = one log file, synchronous appends.**
- Never let two jobs write the same heartbeat/log. They interleave, and with backgrounded appends they RACE and DROP lines — I read two duplicate jobs' interleaved epoch lines as a single "diverging" trajectory and falsely declared a training collapse. Use per-run paths (`hb_<tag>.log`).
- Use SYNCHRONOUS `echo >> LOG`, not backgrounded `( echo >> LOG ) &` — the bg write is lost when the container exits before it flushes (dropped final summary lines).

**4. Resubmitting a "stuck" job creates duplicates.**
- A job with no heartbeat is usually QUEUED for a card, not dead. Resubmitting risks a duplicate that ALSO schedules → two jobs, same tag, colliding on checkpoints. Only resubmit after `mlx job list` shows the original FAILED, or after stopping the original with merlin-cli.

**5. Monitor for FAILURE, not just success — never stall.**
- `mlx job list` shows RUNNING/FAILED. Poll it. A job can FAIL in ~2 min (e.g. a 256GB pod that can't schedule) and the heartbeat shows nothing. Watch best.pt mtime as a liveness signal. Set monitors whose grep catches FAILED / no-progress, not only the happy-path result line. Don't wait 20 min hoping — probe status actively.

**6. Don't misread early training as failure (this cost the most).**
- best.pt frozen for ~6-7 early epochs is NORMAL: it's the val_dyn transient. The screening run that CONVERGED to 52 SR had combined_val spike to 105 at ep4 and best.pt frozen 6 epochs before recovering at ep7. Full-data spikes are smaller. ALWAYS compare a suspected stall against a known-good run's per-epoch trajectory before alarming. Eval of an early-epoch best.pt giving low SR = "undertrained," not "broken."

**7. GPU / pod facts.**
- V100 UNUSABLE: image torch has no sm_70 kernels → `cudaErrorNoKernelImage` on any CUDA op. Target A100/H100. aigcp/H100 queue gives H100s reliably; the `ai` queue is mixed and hands out V100s despite an A100 request.
- 256GB pods can FAIL to schedule (nodes/quota full) while 48GB schedule instantly — never assume a big pod lands; check `mlx job list` for fast FAILED.
- Full-data STREAM ≈ 20-25 min/epoch (HDFS-read-bound); PRELOAD ≈ 3-4x faster (~6 min/ep). Preload full pusht in a 256GB pod (§10-proven; OOMs at 240GB). Do NOT stream full-data if preload fits — it triples wall-time. (Streaming full pusht = ~16-20h for 50ep and blew the deadline.)
