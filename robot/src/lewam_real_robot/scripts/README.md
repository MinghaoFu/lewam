# scripts/

Entry points that don't belong in `main.py`'s Hydra flow. Each takes `--help`.

```bash
source .venv/bin/activate

# teleoperate the follower with the Arm 102 leader
./src/lewam_real_robot/scripts/b601/teleop.sh

# record a LeRobot dataset, cameras attached; one config per dataset
./src/lewam_real_robot/scripts/b601/record.sh --config-name=<dataset>

# answer action-chunk requests from a GPU box, for on-robot rollouts
./src/lewam_real_robot/scripts/serve.sh
```

The b601 scripts need the `b601` extra (`uv pip install -e ".[b601]"`). They detect the
arms' USB ports on first run and cache them in `src/lewam_real_robot/scripts/b601/.env`. Delete it
to force re-detection after a cable swap.
