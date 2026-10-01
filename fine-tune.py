







from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import sys


os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
sys.dont_write_bytecode = True


def _shared_module():
    name = "_epi_shared"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().with_name("latent space construction.py")
    if not path.is_file():
        raise FileNotFoundError(f"缺少共享代码：{path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载共享代码：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module




MODE = "C2"
EPOCHS = 1
BATCH_SIZE = 1
DENOISER_LR = 1e-5
CONDITION_LATE_LR = 5e-6
SEED = 20260930
AMP = True
AUGMENT = False
CHECKPOINT_EVERY = 250
PRINT_EVERY = 25


def main(
    dataset: str | Path | None = None,
    checkpoint: str | Path | None = None,
    output_root: str | Path | None = None,
    *,
    mode: str = MODE,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    denoiser_lr: float = DENOISER_LR,
    condition_late_lr: float = CONDITION_LATE_LR,
    amp: bool = AMP,
    seed: int = SEED,
    resume_checkpoint: str | Path | None = None,
    device: str = "cuda",
):
    shared = _shared_module()
    selected_dataset = Path(dataset).expanduser().resolve() if dataset else shared.choose_dataset_v13("fine_tune")
    selected_checkpoint = Path(checkpoint).expanduser().resolve() if checkpoint else shared.choose_checkpoint_v13()
    run_home = Path(output_root).expanduser().resolve() if output_root else shared.default_v13_run_home()
    settings = {
        "dataset": str(selected_dataset),
        "checkpoint": str(selected_checkpoint),
        "run_home": str(run_home),
        "mode": str(mode).upper(),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "denoiser_lr": float(denoiser_lr),
        "condition_late_lr": float(condition_late_lr),
        "seed": int(seed),
        "device": device,
        "amp": bool(amp),
        "augment": AUGMENT,
        "checkpoint_every": CHECKPOINT_EVERY,
        "print_every": PRINT_EVERY,
        "resume_checkpoint": str(Path(resume_checkpoint).expanduser().resolve()) if resume_checkpoint else None,
    }
    return shared.run_fine_tune_v13(settings)


def _arguments():
    parser = argparse.ArgumentParser(description="v13 simulation-gradient C1/C2 fine-tuning")
    parser.add_argument("--dataset", help="Known dataset root/stage or generic input plus gt/target folder")
    parser.add_argument("--checkpoint", help="Completed v13 pretraining checkpoint")
    parser.add_argument("--output", help="Output root; omitted uses adjacent training_outputs_v13")
    parser.add_argument("--mode", choices=("C1", "C2", "c1", "c2"), default=MODE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--denoiser-lr", type=float, default=DENOISER_LR)
    parser.add_argument("--condition-late-lr", type=float, default=CONDITION_LATE_LR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision")
    parser.add_argument("--resume", help="Hidden resume_fine_tune_v13.pt from the same run")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = _arguments()
    main(
        dataset=args.dataset,
        checkpoint=args.checkpoint,
        output_root=args.output,
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        denoiser_lr=args.denoiser_lr,
        condition_late_lr=args.condition_late_lr,
        amp=not args.no_amp,
        seed=args.seed,
        resume_checkpoint=args.resume,
        device=args.device,
    )
