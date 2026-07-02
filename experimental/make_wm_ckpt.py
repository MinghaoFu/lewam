import json, shutil, sys
from pathlib import Path
CK = "/mnt/data_nvme1/minghao.fu/.stable-wm/checkpoints"
DEC = "/mnt/data_nvme1/minghao.fu/.stable-wm/decoders"
task = sys.argv[1]                       # lift / can / square
adim = int(sys.argv[2]) if len(sys.argv) > 2 else 35   # frameskip(5)*action_dim(7)
cfg = json.loads(Path(f"{CK}/gip_pusht/config.json").read_text())   # same LeWM model architecture
cfg["action_encoder"]["input_dim"] = adim
out = Path(f"{CK}/robomimic_{task}_wm"); out.mkdir(parents=True, exist_ok=True)
(out / "config.json").write_text(json.dumps(cfg, indent=2))
shutil.copy(f"{DEC}/{task}_lewm_weights.pt", out / "weights_epoch_1.pt")
print("wrote", out, "action_encoder.input_dim=", adim)
