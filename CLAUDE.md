# LeWAM Project Instructions (devbox working copy)

> Repo is PRIVATE; cluster/infra details are kept in-repo by the owner's choice (convenience
> first). Companion how-to: `merlin/MERLIN.md`. The original handoff doc is `docs/CLAUDE.md`.

## Working space map (this machine, minghao4)

- **Repo / code**: `/home/tiger/lewam` (this dir). Branch `main`; a background watcher mirrors the
  working tree to GitHub branch `live` every 30s (`~/lewam_project/sync_ctl.sh status|stop|start`;
  never merge `live` into `main` — it is a moving snapshot).
- **Project hub**: `~/lewam_project/` — symlinks `code`→repo, `data`→SSD datasets,
  `outputs`→HDFS results root; `jobs/` = Merlin YAMLs + entry scripts; `logs/`.
- **Results**: `outputs/ckpts/` — §10 campaign `lewam_gc_v1/v2/v3`, baselines `gcidm_scratch_v2`,
  `lewm_repro`, merged-backbone series `merged_wam(2)/baseline_wam/merged_mode_wam`.
  Dashboard: `docs/results/dashboard/index.html`.
- **Paper (Overleaf)**: edited LOCALLY on the Mac (`local_projects/LeWAM/overleaf/`), NOT kept on this machine — https://git.overleaf.com/6a4dc8458ed014fea0608a21 (owner rule 2026-07-08).
  (branch `main`; username literally `git`, password = Overleaf token from the credential store).

## Git identity — HARD RULE

GitHub work uses **Minghao Fu <isminghaofu@gmail.com>**, repo `git@github.com:MinghaoFu/lewam.git`
(PRIVATE). NEVER the bytedance identity, NEVER code.byted.org for GitHub-facing work.
Before committing: `git config user.name "Minghao Fu" && git config user.email isminghaofu@gmail.com`.

## GPU 申请方法 (Merlin/Arnold) — full how-to in `merlin/MERLIN.md`

**Read our Merlin rules before launching GPUs, placing files, or reading data in a job:
`/mnt/hdfs/bi_algo_a2f/minghao.fu/merlin-docs/`** (`merlin-kraken-gpu-ops.md`,
`merlin-worker-interactive.md`, `merlin-job-formal.md`).

### 用户组信息
- **usergroup**: `bi_algorithm`
- **cluster**: `cloudnative-maliva`
- **Arnold groupId**: 62 (仅用于 `mlx job submitv2` YAML 的 `groupIds` 字段)

### 可用 GPU 资源
| GPU 类型 | 数量 | Queue |
|----------|------|-------|
| A100-SXM-80GB (稳定) | 50 | `compute-334-aliyun.va-cloudnative-ai-bi.algorithm-guarantee` |
| V100-SXM2-32GB (稳定) | 43 | `compute-334-aliyun.va-cloudnative-ai-bi.algorithm-guarantee` |
| H100-SXM-80GB (GCP) | 16 | `compute-334-aliyun.va-cloudnative-aigcp-bi.algorithm-guarantee` |

### 路径 A: 交互式 Worker (`mlx worker launch`) — 调试/少量实验
```bash
mlx worker launch --type A100-SXM-80GB --gpu 1 --resourcetype arnold \
  --usergroup bi_algorithm --cluster cloudnative-maliva \
  --queuename compute-334-aliyun.va-cloudnative-ai-bi.algorithm-guarantee -- bash
```
管理: `mlx worker list` / `login <id>` / `kill <id>`;
配额: `mlx worker quota --resourcetype=arnold --usergroup=bi_algorithm`
**注意**: `--usergroup` 用名称 `bi_algorithm`, 不是数字; 数字 62 只进 YAML `groupIds`。
路径 A 需要 TTY — 从无终端环境发起会在 "Login Worker" 阶段失败, 无人值守一律用路径 B。

### 路径 B: 正式任务 (`mlx job submitv2`) — 无人值守
```bash
mlx job submitv2 -p <config>.yaml     # -p 指定 YAML(默认找 ./mlx_config.yaml)
```
YAML 模板与 entry-script 模式见 `merlin/MERLIN.md` (含实测经验: 无 kill 子命令、job log 不可用、
心跳/断点续传模式)。

### 常见错误
- `user is not in group XX` → `--usergroup` 用了数字, 改 `bi_algorithm`
- `gpu type X is not available` → 缺 `--resourcetype arnold`
- 队列排队 → 改用 `mlx worker launch` 直接拿卡

### HDFS 数据路径
| 用途 | 路径 | 说明 |
|------|------|------|
| **SSD 数据** | `/mnt/hdfs/bi_algo_a2f/minghao.fu/lewam/data/` | 解压好的 .h5, entry script 直接 symlink, 无需解压 |
| **HDD 数据** | `/mnt/hdfs/byte_ad_audit/bi_algorithm/minghao.fu/lewam/datasets/` | .h5 备份, SSD 不可用时 fallback |
| **HDD 压缩包** | `/mnt/hdfs/byte_ad_audit/bi_algorithm/minghao.fu/lewam/data/` | .tar.zst 原始归档 |
| **代码** | `/mnt/hdfs/byte_ad_audit/bi_algorithm/minghao.fu/lewam/code/` | entry script + repo tarball |
| **Checkpoints** | `/mnt/hdfs/byte_ad_audit/bi_algorithm/minghao.fu/lewam/ckpts/` | 实验产出 |

SSD 上已解压可直接读: `tworoom.h5`(12G) `pusht_expert_train.h5`(44G) `reacher.h5`(93G)
`ogbench/cube_single_expert.h5`(95G)。**禁止解压流程**: 一律 symlink SSD 的 .h5。

### 存储性能 (实测 2026-07-01)
SSD ≈ HDD ≈ 本地 NVMe, 瓶颈在 HDFS fuse 层 (~7-10ms/episode)。SSD 的价值是省去解压, 不是读取更快。
`cp` 到本地再读反而更慢。Entry script 数据加载模式:
```bash
HSSD=/mnt/hdfs/bi_algo_a2f/minghao.fu/lewam/data
if [ ! -d "$HSSD" ]; then HSSD=$HROOT/datasets; fi
ln -sf "$HSSD/tworoom.h5" "$DS/tworoom.h5"
```

## Reproduce the headline numbers (§10, "many 100%")

Train: `scripts/train_lewam_gc.py --dataset_name <h5> --run_name <task>_lewam_gc --epochs 50
--H_max <25 for tworoom | 50 others> --w_cyc 0.0` (v3: `--w_cyc 1.0`).
Eval: `bash scripts/eval_lewam_gc.sh <task>` (in the repo; remaps ckpt → gcidm format,
3 seeds × N=50). Traps: the loader reads the prebuilt frames_cache (~1× the frame tensor in RAM:
cube/reacher ~140GB, pusht ~160GB, tworoom ~70GB); reacher eval needs `dm_control==1.0.43`;
headless needs `MUJOCO_GL=egl`.
