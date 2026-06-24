# Machine Transfer Manifest — LeWAM / le-wm-repro workspace

How to move the **entire** LeWAM workspace (code + data + checkpoints + env + assets) to a new machine fast and completely. Keep this in sync when assets change.

**Current home:** L40S — `exx@64.62.194.205`, run work as `sudo -u minghao.fu`, venv `lewm`. Two disks: `/mnt/minghao_data` (3 TB xfs, data+ckpts) and `/var/lib/docker` (7.3 TB xfs, code+venv+robomimic+HF). The operative code is `le-wm-repro` (the verified fork of lucas-maes/le-wm).

---

## 1. Asset inventory (everything the project needs)

| asset | source path on L40S | disk | size | needed for |
|---|---|---|---|---|
| **datasets** | `/mnt/minghao_data/.stable-wm/datasets/` | minghao_data | **425 G** | all training/eval (see breakdown ↓) |
| **checkpoints** | `/mnt/minghao_data/.stable-wm/checkpoints/` | minghao_data | **102 G** | converged models + live `ov_*` campaign |
| **SIGReg bases / init weights** | `/mnt/minghao_data/.stable-wm/decoders/*_ours_lewm_weights.pt` | minghao_data | 621 M | `init_from` warm-start (cube/pusht/reacher/tworoom) |
| **eval results** | `/mnt/minghao_data/.stable-wm/gip_eval/` | minghao_data | 54 M | accumulated SR numbers |
| **overnight campaign** | `/mnt/minghao_data/overnight/` | minghao_data | 2 M | orchestrator `run_overnight.sh` + SUMMARY + results.jsonl + logs |
| **CODE** (`le-wm-repro`) | `/var/lib/docker/data/minghao_home/workspace/le-wm-repro/` | docker | 116 M | `train.py`, `jepa.py`, `eval_*`, `config/` |
| **venv `lewm`** | `/var/lib/docker/data/minghao_home/lewm/` | docker | 7.2 G | Python runtime (torch + stable_worldmodel + stable_pretraining) — **may need recreate, see §3** |
| **HF cache** | `/var/lib/docker/data/minghao_home/.cache/huggingface/` | docker | 9.2 G | DINOv2 + CLIP weights (else re-downloads with internet) |
| **robomimic raw data** | `/var/lib/docker/data/minghao_home/robomimic/` | docker | 18 G | lift/can/square/transport/tool_hang eval |
| **robosuite + robomimic env code** | `/var/lib/docker/data/minghao_home/workspace/{robosuite,robomimic}/` | docker | ~690 M | registers the sim env for robomimic eval |

**Datasets breakdown** (the 425 G): `xinyue_mix` 146 G (robomimic abs-action, task #59 — OPTIONAL for the 4-env run), `ogbench` 114 G (cube), `reacher.h5` 93 G, `pusht_expert_train.h5` 44 G, `tworoom.h5` 12 G, `transport.h5` 8.1 G, `tool_hang.h5` 5.7 G, `can.h5` 2 G (+ lift/square).

**Minimum to run the 4-env LeWM campaign** (~260 G): ogbench + reacher + pusht + tworoom + the SIGReg bases + le-wm-repro + venv + HF cache. Skip `xinyue_mix` (146 G) and robomimic unless you need those tasks.

---

## 2. Transfer commands (L40S → `<NEW>`)

Run **on L40S** pushing to the new host (rsync as the right user; mtimes preserved). Replace `<NEW>` with the destination host/path.

```bash
# data + checkpoints + bases + results (the big one — over a fast link)
ssh L40S "rsync -aLP /mnt/minghao_data/.stable-wm/  <NEW>:/mnt/<data>/.stable-wm/"
# overnight campaign (orchestrator + results)
ssh L40S "rsync -aLP /mnt/minghao_data/overnight/   <NEW>:/mnt/<data>/overnight/"
# code + robomimic + robosuite env + HF cache (on the docker disk)
ssh L40S "rsync -aLP /var/lib/docker/data/minghao_home/workspace/le-wm-repro/ <NEW>:<home>/workspace/le-wm-repro/"
ssh L40S "rsync -aLP /var/lib/docker/data/minghao_home/robomimic/             <NEW>:<home>/robomimic/"
ssh L40S "rsync -aLP /var/lib/docker/data/minghao_home/workspace/robosuite/   <NEW>:<home>/workspace/robosuite/"
ssh L40S "rsync -aLP /var/lib/docker/data/minghao_home/.cache/huggingface/    <NEW>:<home>/.cache/huggingface/"
# venv ONLY if same arch+CUDA, else recreate (§3)
ssh L40S "rsync -aLP /var/lib/docker/data/minghao_home/lewm/  <NEW>:<home>/lewm/"
```

Tip (from the 174→L40S hop): if the source can `sudo -u <user> rsync`, push with `--rsync-path='sudo -n -u <user> rsync'` so files land owned correctly. To skip the 146 G `xinyue_mix`, add `--exclude 'datasets/xinyue_mix'`.

---

## 3. Destination setup (do this BEFORE first run)

**a) The codebase hardcodes the H100-174 path layout** `/mnt/data_nvme1/minghao.fu/{.stable-wm, robomimic, le-wm-repro-logs}`. Two ways to satisfy it:
- Set env vars to centralize (preferred): `STABLEWM_HOME=<data>/.stable-wm` (governs `get_cache_dir`), `ROBOMIMIC_RAW=<home>/robomimic`. Most paths flow from `STABLEWM_HOME`.
- OR recreate the **symlink tree** (zero-copy) so the hardcoded paths resolve — exactly what L40S does:
  ```bash
  sudo mkdir -p /mnt/data_nvme1/minghao.fu
  sudo ln -sfn <data>/.stable-wm        /mnt/data_nvme1/minghao.fu/.stable-wm
  sudo ln -sfn <home>/robomimic         /mnt/data_nvme1/minghao.fu/robomimic
  sudo ln -sfn <data>/le-wm-repro-logs  /mnt/data_nvme1/minghao.fu/le-wm-repro-logs
  ```

**b) Per-run disk-safe env redirects (MANDATORY — a disk-full crash killed runs before).** Export ALL of these to a writable disk with space, per run:
`STABLEWM_HOME`, `XDG_CACHE_HOME`, `TMPDIR`, `MPLCONFIGDIR`, `HF_HOME`, and **critically `SPT_CACHE_DIR`** — stable-pretraining writes its ~855 MB Lightning `.ckpt` + `metrics.csv` to `SPT_CACHE_DIR` (default `~/.cache`), which the OTHER vars do NOT govern. Missing it = `FileNotFoundError metrics.csv` / disk-full at ep1. Also `HF_HUB_OFFLINE=0` at eval, `MUJOCO_GL=egl`.

**c) venv.** Rsync the `lewm` venv only if the new box is the same arch + CUDA. Otherwise recreate: `python -m venv lewm && lewm/bin/pip install torch <cuda> stable_worldmodel stable_pretraining` plus the repo deps. Then `lewm/bin/python -c "import torch, stable_worldmodel, stable_pretraining"` must pass.

**d) Mount + fstab.** Put the data disk's mount in `/etc/fstab` so a reboot keeps it (on L40S the data disk was NOT in fstab → it vanished after a reboot and the run broke). Example: `<device> /mnt/<data> xfs rw,auto 0 0`.

---

## 4. Gotchas (learned the hard way — see memory `project_l40s_path_consolidation`, `project_workspace_l40s_lewmrepro`)

- **Hardcoded paths** — the whole codebase assumes `/mnt/data_nvme1/minghao.fu/...`; without the env vars or the symlink tree you get `FileNotFoundError`.
- **`SPT_CACHE_DIR` is separate** — easy to miss; it's the #1 disk-full cause.
- **fstab** — non-fstab data mounts disappear on reboot.
- **Permissions** — `minghao_home` is `drwxr-x---` (owner-only); checking files as another login gives false "missing" — use `sudo` / the right user.
- **HF weights** — DINOv2/CLIP download on demand (needs internet) if the HF cache isn't transferred.
- **GPU policy** — co-locate many jobs per GPU but watch CPU/disk (the real bottleneck); concurrency too high starves data-loading. Never kill other users' GPU jobs.

---

*Generated 2026-06-22 from the L40S workspace survey. Update the sizes/paths when assets change.*
