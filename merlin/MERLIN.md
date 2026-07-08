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
- **There is no `mlx job kill/stop/cancel` subcommand in this CLI** — stop jobs from the web UI
  (the job link printed at submit time).

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
