






from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True


def _load_local(name, filename):
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def shared():
    return _load_local("_epi_shared", "latent space construction.py")


def hardware_module():
    return _load_local("_epi_hardware", "initialization.py")


def tensor_state_hash(state):

    c = shared()
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"{name}|{tensor.dtype}|{tuple(tensor.shape)}".encode())
        digest.update(tensor.reshape(-1).view(c.torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_experiment_checkpoint(path, device):

    c = shared()
    payload = c.torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("role") in {"pre_train", "fine_tune"}:
        return c.load_inference_checkpoint_v13(path, device)
    if payload.get("role") not in {"hardware_deployed", "hardware_transfer"}:
        raise ValueError("Expected a completed v13 simulation or hardware checkpoint")
    if payload.get("format") != c.V13_FORMAT or payload.get("progress", {}).get("status") != "completed":
        raise ValueError("Incomplete or incompatible experimental checkpoint")
    if (payload.get("model_config") != c.ModelConfig().to_dict()
            or payload.get("diffusion_config") != c.DiffusionConfig().to_dict()
            or payload.get("optics_spec") != c.OpticsSpec().to_dict()):
        raise ValueError("Experimental checkpoint architecture or geometry differs")
    meta = payload.get("hardware", {})
    if not isinstance(meta.get("mock"), bool) or not isinstance(meta.get("config_identity"), str):
        raise ValueError("Missing hardware calibration identity/provenance")
    if type(meta.get("real_hardware_executed")) is not bool or meta["real_hardware_executed"] != (not meta["mock"]):
        raise ValueError("Inconsistent real/mock hardware provenance")
    if payload.get("frozen_verification", {}).get("verified_equal") is not True:
        raise ValueError("Experimental checkpoint has no verified frozen-parameter record")
    if tensor_state_hash(payload["model_state"]) != meta.get("model_sha256"):
        raise ValueError("Digital checkpoint integrity check failed")
    if tensor_state_hash(payload["optics_state"]) != meta.get("optics_sha256"):
        raise ValueError("Optical checkpoint integrity check failed")
    model = c.EPIModel().to(device)
    optics = c.OpticalSystem(c.OpticsSpec()).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    optics.load_state_dict(payload["optics_state"], strict=True)
    if not bool(model.latent_stats_fitted):
        raise ValueError("Experimental checkpoint has no fitted latent statistics")
    return model, optics, payload


def require_deployed(payload, session, mock):

    if payload.get("role") not in {"hardware_deployed", "hardware_transfer"}:
        raise ValueError("Run experiment deployment.py before physical transfer/testing")
    meta = payload["hardware"]
    if type(meta.get("mock")) is not bool or type(meta.get("real_hardware_executed")) is not bool:
        raise ValueError("Missing explicit real/mock hardware provenance")
    if meta["real_hardware_executed"] != (not meta["mock"]):
        raise ValueError("Inconsistent real/mock hardware provenance")
    if meta["config_identity"] != session.config_identity:
        raise ValueError("Device/calibration configuration differs from the deployed checkpoint")
    if bool(meta["mock"]) != bool(mock):
        raise ValueError("Mock and real hardware checkpoints cannot be interchanged")


def save_experiment_checkpoint(path, model, optics, source, source_path, role,
                               session, mock, settings, progress, frozen_check):
    c = shared()
    model_state, optics_state = c._cpu_state(model), c._cpu_state(optics)
    payload = {
        "format": c.V13_FORMAT, "role": role,
        "workflow": "camera_feedback_deployment" if role == "hardware_deployed" else "camera_feedback_digital_transfer",
        "model_config": model.config.to_dict(),
        "diffusion_config": model.diffusion_config.to_dict(),
        "optics_spec": optics.spec.to_dict(),
        "model_state": model_state, "optics_state": optics_state,
        "progress": dict(progress), "settings": dict(settings),
        "source_checkpoint_sha256": hashlib.sha256(Path(source_path).read_bytes()).hexdigest(),
        "source_role": source["role"], "frozen_verification": frozen_check,
        "hardware": {"config_identity": session.config_identity, "mock": bool(mock),
                     "real_hardware_executed": not mock,
                     "model_sha256": tensor_state_hash(model_state),
                     "optics_sha256": tensor_state_hash(optics_state)},
    }
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    c.torch.save(payload, temporary)
    c._atomic_replace(temporary, path)


def camera_image_loss(frame, target):

    c = shared()
    array = c.np.asarray(frame, dtype=c.np.float32)
    if array.shape != (3, 256, 256) or not c.np.isfinite(array).all():
        raise ValueError("Camera adapter must return finite RGB 3x256x256 intensities")
    with c.torch.no_grad():
        measured = c.torch.from_numpy(array.copy()).unsqueeze(0)
        value = float(c.image_objective(measured, target.detach().float().cpu()))
    if not math.isfinite(value):
        raise FloatingPointError("Non-finite camera-domain loss")
    return value


def measured_phase_direction(session, cws, odr, target, *, which, epsilon, directions, rng):







    c = shared()
    if not session.supports_feedback:
        raise ValueError("Feedback requires a static or verified replayable scene")
    if which not in {"cws", "odr"}:
        raise ValueError("which must be cws or odr")
    if directions != 4 or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Invalid perturbation settings")
    reference = cws if which == "cws" else odr
    result = c.np.zeros_like(reference, dtype=c.np.float32)
    records = []
    for index in range(directions):
        delta = rng.integers(0, 2, size=reference.shape, dtype=c.np.int8).astype(c.np.float32)
        delta *= 2.0
        delta -= 1.0
        plus = c.np.remainder(reference + epsilon * delta, 2 * math.pi)
        minus = c.np.remainder(reference - epsilon * delta, 2 * math.pi)
        frame_plus = session.measure_rgb(plus, odr) if which == "cws" else session.measure_rgb(cws, plus)
        metadata_plus = copy.deepcopy(getattr(session, "last_measurement_metadata", None))
        lp = camera_image_loss(frame_plus, target)
        frame_minus = session.measure_rgb(minus, odr) if which == "cws" else session.measure_rgb(cws, minus)
        metadata_minus = copy.deepcopy(getattr(session, "last_measurement_metadata", None))
        lm = camera_image_loss(frame_minus, target)
        result += ((lp - lm) / (2.0 * epsilon * directions)) * delta
        records.append({"direction": index, "loss_plus": lp, "loss_minus": lm,
                        "measurement_plus": metadata_plus, "measurement_minus": metadata_minus})
    if not c.np.isfinite(result).all():
        raise FloatingPointError("Non-finite measured phase direction")
    return result, records


def generate_cws(model, pair, device, seed, amp, *, gradients=False):
    c = shared()
    condition = c.load_rgb_tensor(pair.condition).unsqueeze(0).to(device)
    with c.torch.set_grad_enabled(gradients), c.mixed_precision(device, amp):
        generated = model.generate_phase_logits(
            condition, initial_noise=c.stable_noise(pair.key, device, seed),
            checkpoint_denoiser=gradients, checkpoint_decoder=gradients)
        phase = c.phase_from_logits(generated.phase_logits)
    return phase, condition


def make_output(parent, label):
    c = shared()
    parent = Path(parent).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1000000:06d}_{label}_v13"
    path.mkdir()
    (path / "measurements").mkdir()
    (path / "support").mkdir()
    c.hide_internal(path / "support")
    return path


def save_measurement(run, step, pair, frame, condition, target=None, metadata=None):
    c = shared()
    stem = f"{step:05d}_" + hashlib.sha256(pair.key.encode()).hexdigest()[:10]
    dest = run / "measurements"
    c.np.save(dest / f"{stem}_camera.npy", c.np.asarray(frame, dtype=c.np.float32))
    tensor = c.torch.from_numpy(c.np.asarray(frame).copy())
    c.Image.fromarray(c._rgb8(tensor)).save(dest / f"{stem}_camera.png")
    if metadata is not None:
        c.write_json(dest / f"{stem}_acquisition.json", metadata, internal=False)
    if target is not None:
        mock = bool(metadata and metadata.get("mock"))
        c.visual_sheet([(pair.key, condition[0].detach().cpu(), tensor, target[0].cpu())],
                       dest / f"{stem}_comparison.png",
                       "Mock device output (software check)" if mock else "Measured camera output",
                       reconstruction_label="Mock frame" if mock else "Camera measurement")


def write_history(run, rows):
    if not rows:
        return
    path = run / "camera_loss_history_v13.csv"
    temporary = run / "support" / "camera_loss_history_v13.csv.writing"
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    shared()._atomic_replace(temporary, path)


def common_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", type=Path, required=True, help="Calibrated initialization JSON")
    p.add_argument("--dataset", type=Path, required=True, help="Paired acquisition dataset; keys must match configured scene poses")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / "experimental_outputs_v13")
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=20, help="Camera-feedback updates, cycling through selected pairs")
    p.add_argument("--directions", type=int, choices=(4,), default=4, help="Four paired measurements per reference-method update")
    p.add_argument("--epsilon", type=float, default=0.08, help="Phase perturbation amplitude, radians")
    p.add_argument("--seed", type=int, default=20260930)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--mock", action="store_true", help="Software exercise; output is explicitly marked mock")
    return p


def validate_feedback_args(args):
    for name in ("steps", "directions"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.directions != 4:
        raise ValueError("The reference hardware method uses four paired directions")
    if not math.isfinite(args.epsilon) or not 0 < args.epsilon < math.pi:
        raise ValueError("epsilon must be finite and between 0 and pi radians")


def main(argv=None):
    parser = common_parser("B0 camera-feedback deployment: SLM2 only")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args(argv)
    validate_feedback_args(args)
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning-rate must be finite and positive")
    c, hw = shared(), hardware_module()
    config = hw.load_config(args.config)
    selection = c.discover_selected_pairs_v13(args.dataset, purpose="fine_tune")
    device = c.transfer_device(args.device)
    model, optics, source = load_experiment_checkpoint(args.checkpoint, device)
    model.requires_grad_(False).eval()
    optics.requires_grad_(False).eval()
    optics.odr_logits.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    optics.zero_grad(set_to_none=True)
    frozen_digital = tensor_state_hash(model.state_dict())
    frozen_optics = tensor_state_hash({k: v for k, v in optics.state_dict().items() if k != "odr_logits"})
    optimizer = c.torch.optim.Adam([optics.odr_logits], lr=args.learning_rate)
    rng = c.np.random.default_rng(args.seed)
    settings = vars(args).copy()
    rows = []
    session = hw.HardwareSession(config, mock=args.mock)
    if source["role"] in {"hardware_deployed", "hardware_transfer"}:
        require_deployed(source, session, args.mock)
    run = make_output(args.output, "hardware_deployment_mock" if args.mock else "hardware_deployment")
    with session:
        if not session.supports_feedback:
            raise ValueError("B0 needs a static or verified replayable scene")
        for step in range(args.steps):
            pair = selection.pairs[step % len(selection.pairs)]
            session.prepare_scene(pair.key)
            phase, condition = generate_cws(model, pair, device, args.seed, not args.no_amp)
            cws = phase[0].detach().float().cpu().numpy()
            target = c.load_rgb_tensor(pair.target).unsqueeze(0)
            odr = optics.export_odr_phase(cpu=True).numpy()
            direction, probes = measured_phase_direction(session, cws, odr, target, which="odr",
                epsilon=args.epsilon, directions=args.directions, rng=rng)
            optimizer.zero_grad(set_to_none=True)
            
            bridge = (optics.odr_phase() * c.torch.from_numpy(direction).unsqueeze(0).to(device)).sum()
            bridge.backward()
            norm = c.torch.nn.utils.clip_grad_norm_([optics.odr_logits], 1.0)
            if not bool(c.torch.isfinite(norm)):
                raise FloatingPointError("Non-finite SLM2 gradient")
            optimizer.step()
            with c.torch.no_grad():
                if not bool(c.torch.isfinite(optics.odr_logits).all()):
                    raise FloatingPointError("SLM2 update became non-finite")
            frame = session.measure_rgb(cws, optics.export_odr_phase(cpu=True).numpy())
            loss = camera_image_loss(frame, target)
            rows.append({"step": step + 1, "key": pair.key, "camera_image_loss": loss,
                         "gradient_norm": float(norm), "mock": args.mock})
            save_measurement(run, step + 1, pair, frame, condition, target,
                             metadata=session.last_measurement_metadata)
            c.write_json(run / "support" / f"probes_{step + 1:05d}.json", probes)
            write_history(run, rows)
            print(f"B0 {step + 1}/{args.steps}: measured loss={loss:.6g}", flush=True)
        after_digital = tensor_state_hash(model.state_dict())
        after_optics = tensor_state_hash({k: v for k, v in optics.state_dict().items() if k != "odr_logits"})
        if after_digital != frozen_digital or after_optics != frozen_optics:
            raise AssertionError("Deployment changed a parameter outside SLM2")
        check = {"verified_equal": True, "digital_sha256": frozen_digital,
                 "nontrainable_optics_sha256": frozen_optics, "trainable": "SLM2 odr_logits only"}
        checkpoint = run / "deployed_v13.pt"
        save_experiment_checkpoint(checkpoint, model, optics, source, args.checkpoint, "hardware_deployed",
                                   session, args.mock, settings, {"status": "completed", "updates": args.steps}, check)
        c.np.save(run / "selected_slm2_phase_v13.npy", optics.export_odr_phase(cpu=True).numpy())
        c.write_json(run / "deployment_summary_v13.json", {"checkpoint": str(checkpoint),
            "mock": args.mock, "real_hardware_executed": not args.mock, "updates": args.steps,
            "frozen_verification": check, "config_identity": session.config_identity,
            "selection": "SLM2 after the configured number of measured updates",
            "independent_evaluation": False}, internal=False)
    print(f"OUTPUT: {run}", flush=True)
    return run


if __name__ == "__main__":
    main()
