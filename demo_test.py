






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


SEED = 20260930
AMP = True
PHASE_EXAMPLES = 3
PRINT_EVERY = 25


def main(
    dataset: str | Path | None = None,
    checkpoint: str | Path | None = None,
    output_root: str | Path | None = None,
    *,
    device: str = "cuda",
    phase_examples: int = PHASE_EXAMPLES,
    amp: bool = AMP,
    seed: int = SEED,
):
    shared = _shared_module()
    selected_dataset = Path(dataset).expanduser().resolve() if dataset else shared.choose_dataset_v13("test")
    selected_checkpoint = Path(checkpoint).expanduser().resolve() if checkpoint else shared.choose_checkpoint_v13()
    run_home = Path(output_root).expanduser().resolve() if output_root else shared.default_v13_run_home()
    settings = {
        "dataset": str(selected_dataset),
        "checkpoint": str(selected_checkpoint),
        "run_home": str(run_home),
        "device": device,
        "seed": int(seed),
        "amp": bool(amp),
        "phase_examples": int(phase_examples),
        "print_every": PRINT_EVERY,
    }
    return shared.run_test_v13(settings)


def _arguments():
    parser = argparse.ArgumentParser(description="v13 condition-only demo/test")
    parser.add_argument("--dataset", help="Known root/stage, generic input plus gt/target, or input-only folder")
    parser.add_argument("--checkpoint", help="Completed v13 pretraining or fine-tuning checkpoint")
    parser.add_argument("--output", help="Output root; omitted uses adjacent training_outputs_v13")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--phase-examples", type=int, default=PHASE_EXAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision")
    return parser.parse_args()


if __name__ == "__main__":
    args = _arguments()
    main(
        dataset=args.dataset,
        checkpoint=args.checkpoint,
        output_root=args.output,
        device=args.device,
        phase_examples=args.phase_examples,
        amp=not args.no_amp,
        seed=args.seed,
    )
