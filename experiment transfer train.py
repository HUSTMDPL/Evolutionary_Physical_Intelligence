






from pathlib import Path
import importlib.util
import math
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


def main(argv=None):
    h = deployment_helpers()
    parser = h.common_parser("C1/C2 camera-feedback digital transfer with fixed SLM2")
    parser.add_argument("--mode", choices=("C1", "C2"), default="C2")
    parser.add_argument("--denoiser-lr", type=float, default=1e-5)
    parser.add_argument("--condition-late-lr", type=float, default=5e-6)
    args = parser.parse_args(argv)
    h.validate_feedback_args(args)
    if any(not math.isfinite(v) or v <= 0 for v in (args.denoiser_lr, args.condition_late_lr)):
        raise ValueError("Learning rates must be finite and positive")
    c, hw = h.shared(), h.hardware_module()
    config = hw.load_config(args.config)
    selection = c.discover_selected_pairs_v13(args.dataset, purpose="fine_tune")
    device = c.transfer_device(args.device)
    model, optics, source = h.load_experiment_checkpoint(args.checkpoint, device)
    optimizer, trainable, counts = c._configure_otl_v13(
        model, optics, args.mode, args.denoiser_lr, args.condition_late_lr)
    reference = c._frozen_state_v13(model, optics, args.mode)
    fixed_odr = optics.export_odr_phase(cpu=True).numpy()
    rng = c.np.random.default_rng(args.seed)
    rows = []
    session = hw.HardwareSession(config, mock=args.mock)
    h.require_deployed(source, session, args.mock)
    run = h.make_output(args.output, "hardware_transfer_mock" if args.mock else "hardware_transfer")
    with session:
        if not session.supports_feedback:
            raise ValueError("Digital camera feedback needs a static or verified replayable scene")
        for step in range(args.steps):
            pair = selection.pairs[step % len(selection.pairs)]
            session.prepare_scene(pair.key)
            optimizer.zero_grad(set_to_none=True)
            
            
            phase, condition = h.generate_cws(model, pair, device, args.seed, not args.no_amp, gradients=True)
            cws = phase[0].detach().float().cpu().numpy()
            target = c.load_rgb_tensor(pair.target).unsqueeze(0)
            direction, probes = h.measured_phase_direction(session, cws, fixed_odr, target, which="cws",
                epsilon=args.epsilon, directions=args.directions, rng=rng)
            bridge = (phase * c.torch.from_numpy(direction).unsqueeze(0).to(device)).sum()
            bridge.backward()
            norm = c.torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if not bool(c.torch.isfinite(norm)):
                raise FloatingPointError("Non-finite digital feedback gradient")
            optimizer.step()
            if not all(bool(c.torch.isfinite(parameter).all()) for parameter in trainable):
                raise FloatingPointError("Digital update became non-finite")
            del phase, bridge
            updated_phase, condition = h.generate_cws(model, pair, device, args.seed, not args.no_amp)
            frame = session.measure_rgb(updated_phase[0].detach().float().cpu().numpy(), fixed_odr)
            loss = h.camera_image_loss(frame, target)
            rows.append({"step": step + 1, "key": pair.key, "camera_image_loss": loss,
                         "gradient_norm": float(norm), "mock": args.mock})
            h.save_measurement(run, step + 1, pair, frame, condition, target,
                               metadata=session.last_measurement_metadata)
            c.write_json(run / "support" / f"probes_{step + 1:05d}.json", probes)
            h.write_history(run, rows)
            print(f"{args.mode} {step + 1}/{args.steps}: measured loss={loss:.6g}", flush=True)
        check = c._assert_frozen_equal_v13(reference, model, optics, args.mode)
        checkpoint = run / "transferred_v13.pt"
        h.save_experiment_checkpoint(checkpoint, model, optics, source, args.checkpoint, "hardware_transfer",
            session, args.mock, vars(args), {"status": "completed", "updates": args.steps, "mode": args.mode}, check)
        c.write_json(run / "transfer_summary_v13.json", {"checkpoint": str(checkpoint),
            "mock": args.mock, "real_hardware_executed": not args.mock, "updates": args.steps,
            "trainable_parameter_counts": counts, "frozen_verification": check,
            "config_identity": session.config_identity, "reported_loss": "measured camera image objective",
            "independent_evaluation": False}, internal=False)
    print(f"OUTPUT: {run}", flush=True)
    return run


if __name__ == "__main__":
    main()
