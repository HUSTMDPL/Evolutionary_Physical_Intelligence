







from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True


def deployment_helpers():
    name = "_epi_experiment"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().with_name("experiment deployment.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _parser():
    parser = argparse.ArgumentParser(
        description="Deployed v13 condition-only experimental test"
    )
    parser.add_argument("--config", type=Path, required=True, help="Calibrated initialization JSON")
    parser.add_argument(
        "--dataset", type=Path, required=True,
        help="Test input folder or paired input plus gt/target folder",
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Completed deployed/transferred checkpoint")
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent.parent / "experimental_outputs_v13",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--mock", action="store_true",
        help="Software validation only; results are explicitly marked mock",
    )
    return parser


def _validated_frame(c, frame):
    array = c.np.asarray(frame)
    if array.dtype != c.np.float32:
        raise ValueError("HardwareSession.measure_rgb must return float32")
    if array.shape != (3, 256, 256) or not c.np.isfinite(array).all():
        raise ValueError("HardwareSession.measure_rgb must return finite RGB 3x256x256")
    return array


def _validated_phase(c, value, name):
    array = c.np.asarray(value, dtype=c.np.float32)
    if array.shape != (3, 2048, 2048) or not c.np.isfinite(array).all():
        raise ValueError(f"{name} must be finite RGB 3x2048x2048 phase radians")
    return array


def _metrics(c, image, target):
    mse = (image.float() - target.float()).square().mean().clamp_min(1e-12)
    psnr = float(-10.0 * c.torch.log10(mse))
    ms_ssim = float(c.ms_ssim_index(image.float(), target.float()))
    if not math.isfinite(psnr) or not math.isfinite(ms_ssim):
        raise FloatingPointError("Non-finite evaluation metric")
    return psnr, ms_ssim


def _write_rows(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.print_every <= 0:
        raise ValueError("print-every must be positive")

    h = deployment_helpers()
    c, hw = h.shared(), h.hardware_module()
    config = hw.load_config(args.config)
    selection = c.discover_selected_pairs_v13(args.dataset, purpose="test")
    device = c.transfer_device(args.device)
    model, optics, payload = h.load_experiment_checkpoint(args.checkpoint, device)

    
    
    if payload.get("role") not in {"hardware_deployed", "hardware_transfer"}:
        raise ValueError("Run experiment deployment.py before experimental testing")
    model.requires_grad_(False).eval()
    optics.requires_grad_(False).eval()
    before_model = h.tensor_state_hash(model.state_dict())
    before_optics = h.tensor_state_hash(optics.state_dict())
    fixed_odr = _validated_phase(
        c, optics.export_odr_phase(cpu=True).detach().numpy(), "SLM2 phase"
    )

    rows = []
    acquisitions = []
    config_identity = None

    
    
    session = hw.HardwareSession(config, mock=args.mock)
    h.require_deployed(payload, session, args.mock)
    label = "hardware_test_mock" if args.mock else "hardware_test"
    run = h.make_output(args.output, label)
    c.np.save(run / "selected_slm2_phase_v13.npy", fixed_odr)
    with session:
        config_identity = session.config_identity
        with c.torch.inference_mode():
            for index, pair in enumerate(selection.pairs, 1):
                
                
                session.prepare_scene(pair.key)
                phase, condition = h.generate_cws(
                    model, pair, device, args.seed, not args.no_amp,
                    gradients=False,
                )
                cws = _validated_phase(
                    c, phase[0].detach().float().cpu().numpy(), "SLM1 phase"
                )
                frame = _validated_frame(c, session.measure_rgb(cws, fixed_odr))

                
                
                target = None
                values = {
                    "index": index,
                    "stage": pair.stage,
                    "family": pair.family,
                    "domain": pair.domain,
                    "key": pair.key,
                    "group": pair.group,
                    "mock": bool(args.mock),
                }
                if pair.target is not None:
                    target = c.load_rgb_tensor(pair.target).unsqueeze(0)
                    measured = c.torch.from_numpy(frame.copy()).unsqueeze(0)
                    input_cpu = condition.detach().float().cpu()
                    values["input_psnr"], values["input_ms_ssim"] = _metrics(
                        c, input_cpu, target
                    )
                    values["camera_psnr"], values["camera_ms_ssim"] = _metrics(
                        c, measured, target
                    )
                    rows.append(values)

                acquisition_metadata = getattr(
                    session, "last_measurement_metadata", None
                )
                h.save_measurement(
                    run, index, pair, frame, condition, target,
                    metadata=acquisition_metadata,
                )
                digest = hashlib.sha256(pair.key.encode()).hexdigest()[:10]
                prefix = f"{index:05d}_{digest}"
                acquisitions.append({
                    "index": index,
                    "key": pair.key,
                    "stage": pair.stage,
                    "family": pair.family,
                    "domain": pair.domain,
                    "group": pair.group,
                    "condition": str(pair.condition),
                    "target": str(pair.target) if pair.target is not None else "",
                    "camera_npy": f"measurements/{prefix}_camera.npy",
                    "camera_png": f"measurements/{prefix}_camera.png",
                    "comparison_png": (
                        f"measurements/{prefix}_comparison.png"
                        if pair.target is not None else ""
                    ),
                    "acquisition_json": (
                        f"measurements/{prefix}_acquisition.json"
                        if acquisition_metadata is not None else ""
                    ),
                    "mock": bool(args.mock),
                })
                _write_rows(run / "acquisition_manifest_v13.csv", acquisitions)
                if index % args.print_every == 0 or index == len(selection.pairs):
                    print(f"experimental test: {index}/{len(selection.pairs)}", flush=True)

    after_model = h.tensor_state_hash(model.state_dict())
    after_optics = h.tensor_state_hash(optics.state_dict())
    if after_model != before_model or after_optics != before_optics:
        raise AssertionError("Experimental test changed model or optical parameters")

    _write_rows(run / "metrics_v13.csv", rows)
    metric_names = ("input_psnr", "input_ms_ssim", "camera_psnr", "camera_ms_ssim")
    means = {
        name: float(c.np.mean([float(row[name]) for row in rows]))
        for name in metric_names
    } if rows else {}
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    c.write_json(
        run / "experiment_test_summary_v13.json",
        {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "checkpoint_role": payload.get("role"),
            "checkpoint_workflow": payload.get("workflow"),
            "config_identity": config_identity,
            "mock": bool(args.mock),
            "real_hardware_executed": not args.mock,
            "conditions": len(selection.pairs),
            "acquisition_manifest": "acquisition_manifest_v13.csv",
            "targets_available": len(rows),
            "metrics": means,
            "selected_data": selection.summary(),
            "target_used_by_generator": False,
            "scene_pairing": "prepare_scene(pair.key)",
            "slm2_source": "fixed phase exported from checkpoint optics",
            "parameter_updates": 0,
            "parameters_verified_unchanged": True,
            "independent_test_claimed": False,
        },
        internal=False,
    )
    print(f"OUTPUT: {run}", flush=True)
    return run


if __name__ == "__main__":
    main()
