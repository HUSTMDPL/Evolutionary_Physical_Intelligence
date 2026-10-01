



















from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import sys
sys.dont_write_bytecode = True
import tempfile
import time
from typing import Any, Mapping, Sequence
import uuid

import numpy as np


CHANNELS = ("red", "green", "blue")
PHASE_SHAPE = (3, 2048, 2048)
OUTPUT_SHAPE = (3, 256, 256)
MOVING_MASK = 0x000002F0
LIMIT_MASK = 0x0000000F
MOTOR_CONNECTED_MASK = 0x00000100
MOTOR_ENABLED_MASK = 0x80000000


class ConfigurationError(ValueError):
    pass


class HardwareError(RuntimeError):
    pass


class HardwareBusyError(HardwareError):
    pass


def choose_json_path(title: str = "Choose EPI hardware configuration") -> Path:


    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    try:
        selected = filedialog.askopenfilename(
            title=title,
            filetypes=(("JSON configuration", "*.json"), ("All files", "*.*")),
        )
    finally:
        root.destroy()
    if not selected:
        raise ConfigurationError("No configuration file was selected")
    return Path(selected).expanduser().resolve()


def load_shared():


    name = "_epi_shared"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).with_name("latent space construction.py")
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load numerical core: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def reference_config() -> dict[str, Any]:


    slm = {
        "model": "FSLM-2K73-P02",
        "serial": None,
        "monitor_device": None,
        "resolution": [2048, 2048],
        "phase_lut": {channel: None for channel in CHANNELS},
        "phase_correction": {channel: None for channel in CHANNELS},
    }
    return {
        "schema": "epi-hardware-config-v1",
        "setup_id": None,
        "optical_geometry": {
            "slm_spacing_m": 0.50,
            "target_conjugate": "infinity",
            "camera_plane": "lens_back_focal_plane",
        },
        "slms": [dict(slm, name="SLM1_CWS"), dict(slm, name="SLM2_ODR")],
        "rgb_source": {
            "transport": "serial_ascii",
            "device_id": None,
            "port": None,
            "baud_rate": None,
            "timeout_s": None,
            "encoding": "ascii",
            "channels": {
                channel: {
                    "level": None,
                    "set_command": None,
                    "off_command": None,
                    "ack_exact": None,
                    "ack_regex": None,
                }
                for channel in CHANNELS
            },
        },
        "camera": {
            "sdk_python_dir": None,
            "dll_directories": [],
            "serial": None,
            "model": None,
            "enumeration_timeout_ms": None,
            "frame_timeout_ms": None,
            "exposure_us": None,
            "gain_db": None,
            "pixel_format": "Mono8",
            "calibration": {
                "roi_xywh": [None, None, 256, 256],
                "dark": {channel: None for channel in CHANNELS},
                "flat": {channel: None for channel in CHANNELS},
                "radiometric_scale": {channel: None for channel in CHANNELS},
                "denominator_floor": None,
                "output_clip": None,
            },
        },
        "stage": {
            "wrapper_path": None,
            "port": None,
            "baud_rate": None,
            "axis": 0,
            "controller_serial": None,
            "stage_serial": None,
            "soft_min": None,
            "soft_max": None,
            "position_tolerance": None,
            "motion_timeout_s": None,
            "poll_interval_s": None,
            "require_homed": True,
        },
        "acquisition": {"settle_s": None},
        "scenes": {
            "*": {
                "stage_position": None,
                "static": None,
                "replayable": None,
                "operator_confirmed": False,
            }
        },
    }


def _mock_config() -> dict[str, Any]:
    config = reference_config()
    config["setup_id"] = "mock-only"
    config["acquisition"]["settle_s"] = 0.0
    config["scenes"] = {
        "mock_static": {
            "stage_position": 0.0,
            "static": True,
            "replayable": True,
            "operator_confirmed": True,
        }
    }
    return config


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:


    resolved = Path(path).expanduser().resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfigurationError("The top level of the configuration must be an object")
    value["__config_dir__"] = str(resolved.parent)
    value["__config_path__"] = str(resolved)
    return value


def _need(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in mapping or mapping[key] is None or mapping[key] == "":
        raise ConfigurationError(f"Missing {where}.{key}")
    return mapping[key]


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ConfigurationError(f"{name} has an invalid value")
    return result


def _config_base(config: Mapping[str, Any]) -> Path:
    return Path(str(config.get("__config_dir__", Path.cwd()))).expanduser().resolve()


def _resolve_file(config: Mapping[str, Any], value: Any, name: str) -> Path:
    if value is None or value == "":
        raise ConfigurationError(f"Missing {name}")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = _config_base(config) / path
    path = path.resolve()
    if not path.is_file():
        raise ConfigurationError(f"{name} does not exist: {path}")
    return path


def _resolve_directory(config: Mapping[str, Any], value: Any, name: str) -> Path:
    if value is None or value == "":
        raise ConfigurationError(f"Missing {name}")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = _config_base(config) / path
    path = path.resolve()
    if not path.is_dir():
        raise ConfigurationError(f"{name} does not exist: {path}")
    return path


def validate_config(config: Mapping[str, Any], *, real: bool) -> None:


    if config.get("schema") != "epi-hardware-config-v1":
        raise ConfigurationError("schema must be 'epi-hardware-config-v1'")
    geometry = config.get("optical_geometry")
    if not isinstance(geometry, Mapping):
        raise ConfigurationError("optical_geometry must be an object")
    spacing = _number(_need(geometry, "slm_spacing_m", "optical_geometry"), "slm_spacing_m", positive=True)
    if not math.isclose(spacing, 0.50, rel_tol=0.0, abs_tol=1e-6):
        raise ConfigurationError("The measured SLM separation for this setup must remain 0.50 m")
    if geometry.get("target_conjugate") != "infinity":
        raise ConfigurationError("target_conjugate must be 'infinity'")
    if geometry.get("camera_plane") != "lens_back_focal_plane":
        raise ConfigurationError("camera_plane must be 'lens_back_focal_plane'")

    slms = config.get("slms")
    if not isinstance(slms, list) or len(slms) != 2:
        raise ConfigurationError("Exactly two SLM entries are required")
    monitor_names: list[str] = []
    serials: list[str] = []
    for index, slm in enumerate(slms, 1):
        where = f"slms[{index - 1}]"
        if not isinstance(slm, Mapping):
            raise ConfigurationError(f"{where} must be an object")
        if slm.get("model") != "FSLM-2K73-P02":
            raise ConfigurationError(f"{where}.model must be FSLM-2K73-P02")
        if list(slm.get("resolution", [])) != [2048, 2048]:
            raise ConfigurationError(f"{where}.resolution must be [2048, 2048]")
        for field in ("phase_lut", "phase_correction"):
            values = slm.get(field)
            if not isinstance(values, Mapping) or set(values) != set(CHANNELS):
                raise ConfigurationError(f"{where}.{field} must define red, green and blue")
            if real:
                for channel in CHANNELS:
                    path = _resolve_file(config, values[channel], f"{where}.{field}.{channel}")
                    array = _load_numeric(path)
                    expected = (256,) if field == "phase_lut" else (2048, 2048)
                    if array.shape != expected:
                        raise ConfigurationError(f"{where}.{field}.{channel} must have shape {expected}")
                    if field == "phase_lut" and (np.min(array) < 0 or np.max(array) > 255):
                        raise ConfigurationError(f"{where}.{field}.{channel} must remain within [0, 255]")
        if real:
            serials.append(str(_need(slm, "serial", where)))
            monitor_names.append(str(_need(slm, "monitor_device", where)))
    if real and (len(set(serials)) != 2 or len(set(monitor_names)) != 2):
        raise ConfigurationError("The two SLM serials and monitor identities must be distinct")

    acquisition = config.get("acquisition")
    if not isinstance(acquisition, Mapping):
        raise ConfigurationError("acquisition must be an object")
    settle = acquisition.get("settle_s")
    if real or settle is not None:
        if _number(settle, "acquisition.settle_s") < 0:
            raise ConfigurationError("acquisition.settle_s cannot be negative")

    camera = config.get("camera")
    if not isinstance(camera, Mapping):
        raise ConfigurationError("camera must be an object")
    if camera.get("pixel_format") != "Mono8":
        raise ConfigurationError("camera.pixel_format must be Mono8")
    calibration = camera.get("calibration")
    if not isinstance(calibration, Mapping):
        raise ConfigurationError("camera.calibration must be an object")
    roi = calibration.get("roi_xywh")
    if not isinstance(roi, list) or len(roi) != 4 or roi[2:] != [256, 256]:
        raise ConfigurationError("camera.calibration.roi_xywh must end in [256, 256]")
    for field in ("dark", "flat", "radiometric_scale"):
        values = calibration.get(field)
        if not isinstance(values, Mapping) or set(values) != set(CHANNELS):
            raise ConfigurationError(f"camera.calibration.{field} must define red, green and blue")
    clip = calibration.get("output_clip")
    if clip is not None:
        if not isinstance(clip, list) or len(clip) != 2:
            raise ConfigurationError("camera.calibration.output_clip must be null or [minimum, maximum]")
        lo = _number(clip[0], "output_clip minimum")
        hi = _number(clip[1], "output_clip maximum")
        if hi <= lo:
            raise ConfigurationError("output_clip maximum must exceed minimum")
    if real:
        _resolve_directory(config, _need(camera, "sdk_python_dir", "camera"), "camera.sdk_python_dir")
        _need(camera, "serial", "camera")
        for name in ("enumeration_timeout_ms", "frame_timeout_ms"):
            value = _number(_need(camera, name, "camera"), f"camera.{name}", positive=True)
            if not float(value).is_integer():
                raise ConfigurationError(f"camera.{name} must be an integer")
        _number(_need(camera, "exposure_us", "camera"), "camera.exposure_us", positive=True)
        _number(_need(camera, "gain_db", "camera"), "camera.gain_db")
        if roi[0] is None or roi[1] is None or any(type(v) is not int or v < 0 for v in roi):
            raise ConfigurationError("camera.calibration.roi_xywh must contain non-negative integers")
        _number(_need(calibration, "denominator_floor", "camera.calibration"), "denominator_floor", positive=True)
        for field in ("dark", "flat"):
            for channel in CHANNELS:
                path = _resolve_file(config, calibration[field][channel], f"camera.calibration.{field}.{channel}")
                array = _load_numeric(path)
                if array.ndim != 2:
                    raise ConfigurationError(f"camera.calibration.{field}.{channel} must be two-dimensional")
        for channel in CHANNELS:
            dark = _load_numeric(_resolve_file(config, calibration["dark"][channel], f"dark.{channel}"))
            flat = _load_numeric(_resolve_file(config, calibration["flat"][channel], f"flat.{channel}"))
            if dark.shape != flat.shape:
                raise ConfigurationError(f"Dark and flat calibration shapes differ for {channel}")
            _number(calibration["radiometric_scale"][channel], f"radiometric_scale.{channel}", positive=True)
        for index, value in enumerate(camera.get("dll_directories", [])):
            _resolve_directory(config, value, f"camera.dll_directories[{index}]")

    source = config.get("rgb_source")
    if not isinstance(source, Mapping):
        raise ConfigurationError("rgb_source must be an object")
    if real:
        if source.get("transport") != "serial_ascii":
            raise ConfigurationError("Only the explicitly acknowledged serial_ascii source is supported")
        _need(source, "device_id", "rgb_source")
        _need(source, "port", "rgb_source")
        baud = _number(_need(source, "baud_rate", "rgb_source"), "rgb_source.baud_rate", positive=True)
        if not baud.is_integer():
            raise ConfigurationError("rgb_source.baud_rate must be an integer")
        _number(_need(source, "timeout_s", "rgb_source"), "rgb_source.timeout_s", positive=True)
        channels = source.get("channels")
        if not isinstance(channels, Mapping) or set(channels) != set(CHANNELS):
            raise ConfigurationError("rgb_source.channels must define red, green and blue")
        for channel in CHANNELS:
            item = channels[channel]
            if not isinstance(item, Mapping):
                raise ConfigurationError(f"rgb_source.channels.{channel} must be an object")
            _number(_need(item, "level", f"rgb_source.channels.{channel}"), f"{channel}.level")
            _need(item, "set_command", f"rgb_source.channels.{channel}")
            _need(item, "off_command", f"rgb_source.channels.{channel}")
            if not item.get("ack_exact") and not item.get("ack_regex"):
                raise ConfigurationError(f"rgb_source.channels.{channel} needs ack_exact or ack_regex")

    stage = config.get("stage")
    if not isinstance(stage, Mapping):
        raise ConfigurationError("stage must be an object")
    if real:
        _resolve_file(config, _need(stage, "wrapper_path", "stage"), "stage.wrapper_path")
        for name in ("port", "controller_serial", "stage_serial"):
            _need(stage, name, "stage")
        for name in ("baud_rate", "axis"):
            value = _number(_need(stage, name, "stage"), f"stage.{name}")
            if not value.is_integer() or value < 0:
                raise ConfigurationError(f"stage.{name} must be a non-negative integer")
        soft_min = _number(_need(stage, "soft_min", "stage"), "stage.soft_min")
        soft_max = _number(_need(stage, "soft_max", "stage"), "stage.soft_max")
        if soft_max <= soft_min:
            raise ConfigurationError("stage.soft_max must exceed stage.soft_min")
        for name in ("position_tolerance", "motion_timeout_s", "poll_interval_s"):
            _number(_need(stage, name, "stage"), f"stage.{name}", positive=True)
        if type(stage.get("require_homed")) is not bool:
            raise ConfigurationError("stage.require_homed must be true or false")

    scenes = config.get("scenes")
    if not isinstance(scenes, Mapping) or not scenes:
        raise ConfigurationError("At least one named scene is required")
    for key, scene in scenes.items():
        if not isinstance(scene, Mapping):
            raise ConfigurationError(f"scenes.{key} must be an object")
        for flag in ("static", "replayable", "operator_confirmed"):
            if type(scene.get(flag)) is not bool and (real or scene.get(flag) is not None):
                raise ConfigurationError(f"scenes.{key}.{flag} must be true or false")
        position = scene.get("stage_position")
        if real:
            if position is None and not scene.get("operator_confirmed"):
                raise ConfigurationError(
                    f"scenes.{key} needs stage_position or an explicit operator_confirmed=true"
                )
            if position is not None:
                pos = _number(position, f"scenes.{key}.stage_position")
                if not (soft_min <= pos <= soft_max):
                    raise ConfigurationError(f"scenes.{key}.stage_position exceeds the configured soft limits")
    if real:
        _need(config, "setup_id", "configuration")


def _identity_config(config: Mapping[str, Any]) -> dict[str, Any]:


    slms = []
    for slm in config.get("slms", []):
        slms.append(
            {
                "name": slm.get("name"),
                "model": slm.get("model"),
                "serial": slm.get("serial"),
                "monitor_device": slm.get("monitor_device"),
                "resolution": slm.get("resolution"),
            }
        )
    camera = config.get("camera", {})
    source = config.get("rgb_source", {})
    stage = config.get("stage", {})
    return {
        "schema": config.get("schema"),
        "setup_id": config.get("setup_id"),
        "optical_geometry": config.get("optical_geometry"),
        "slms": slms,
        "rgb_source": {
            "transport": source.get("transport"),
            "device_id": source.get("device_id"),
            "channels": source.get("channels"),
        },
        "camera": {
            "serial": camera.get("serial"),
            "model": camera.get("model"),
            "exposure_us": camera.get("exposure_us"),
            "gain_db": camera.get("gain_db"),
            "pixel_format": camera.get("pixel_format"),
            "calibration": {
                "roi_xywh": camera.get("calibration", {}).get("roi_xywh"),
                "radiometric_scale": camera.get("calibration", {}).get("radiometric_scale"),
                "denominator_floor": camera.get("calibration", {}).get("denominator_floor"),
                "output_clip": camera.get("calibration", {}).get("output_clip"),
            },
        },
        "stage": {
            "axis": stage.get("axis"),
            "controller_serial": stage.get("controller_serial"),
            "stage_serial": stage.get("stage_serial"),
            "soft_min": stage.get("soft_min"),
            "soft_max": stage.get("soft_max"),
            "position_tolerance": stage.get("position_tolerance"),
            "require_homed": stage.get("require_homed"),
        },
        "acquisition": {"settle_s": config.get("acquisition", {}).get("settle_s")},
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_identity(config: Mapping[str, Any], *, include_files: bool = True) -> str:


    payload: dict[str, Any] = {"config": _identity_config(config), "files": {}}
    if include_files:
        paths: list[tuple[str, Any]] = []
        for index, slm in enumerate(config.get("slms", [])):
            for field in ("phase_lut", "phase_correction"):
                for channel, value in slm.get(field, {}).items():
                    paths.append((f"slms.{index}.{field}.{channel}", value))
        calibration = config.get("camera", {}).get("calibration", {})
        for field in ("dark", "flat"):
            for channel, value in calibration.get(field, {}).items():
                paths.append((f"camera.calibration.{field}.{channel}", value))
        for label, value in paths:
            if value not in (None, ""):
                path = _resolve_file(config, value, label)
                payload["files"][label] = {"sha256": _sha256_file(path)}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _physical_lock_identity(config: Mapping[str, Any]) -> str:


    physical = {
        "slms": [
            {
                "serial": slm.get("serial"),
                "monitor_device": slm.get("monitor_device"),
            }
            for slm in config.get("slms", [])
        ],
        "camera_serial": config.get("camera", {}).get("serial"),
        "stage": {
            "port": config.get("stage", {}).get("port"),
            "controller_serial": config.get("stage", {}).get("controller_serial"),
            "stage_serial": config.get("stage", {}).get("stage_serial"),
        },
        "rgb_source": {
            "port": config.get("rgb_source", {}).get("port"),
            "device_id": config.get("rgb_source", {}).get("device_id"),
        },
    }
    canonical = json.dumps(physical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _hide_windows(path: Path) -> None:
    if os.name != "nt" or not path.exists():
        return
    kernel32 = ctypes.windll.kernel32
    attributes = kernel32.GetFileAttributesW(str(path))
    if attributes != 0xFFFFFFFF:
        kernel32.SetFileAttributesW(str(path), attributes | 0x2)


class _ExclusiveHardwareLock:


    def __init__(self, identity: str):
        self.path = Path(tempfile.gettempdir()) / f"epi_hardware_{identity[:24]}.lock"
        self._handle = None
        self._locked = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        _hide_windows(self.path)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            handle.close()
            raise HardwareBusyError("Another process controls this physical EPI configuration") from error
        self._handle = handle
        self._locked = True

    def release(self) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            if self._locked:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._locked = False
            handle.close()


def _load_numeric(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path, allow_pickle=False)
    elif suffix == ".json":
        array = np.asarray(json.loads(path.read_text(encoding="utf-8")))
    elif suffix in {".csv", ".txt"}:
        array = np.loadtxt(path, delimiter="," if suffix == ".csv" else None)
    else:
        raise ConfigurationError(f"Unsupported numeric calibration file: {path}")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ConfigurationError(f"Calibration contains non-finite or non-numeric values: {path}")
    return np.asarray(array)


class _SlmCalibration:
    def __init__(self, config: Mapping[str, Any], slm: Mapping[str, Any]):
        self._lut: dict[str, np.ndarray] = {}
        self._correction: dict[str, np.ndarray] = {}
        for channel in CHANNELS:
            lut = _load_numeric(_resolve_file(config, slm["phase_lut"][channel], f"phase_lut.{channel}"))
            if lut.shape != (256,):
                raise ConfigurationError(f"The {channel} phase LUT must have exactly 256 entries")
            if np.min(lut) < 0 or np.max(lut) > 255:
                raise ConfigurationError(f"The {channel} phase LUT must remain within [0, 255]")
            self._lut[channel] = np.rint(lut).astype(np.uint8)
            correction = _load_numeric(
                _resolve_file(config, slm["phase_correction"][channel], f"phase_correction.{channel}")
            )
            if correction.shape != (2048, 2048):
                raise ConfigurationError(f"The {channel} phase correction must be 2048 x 2048")
            self._correction[channel] = correction.astype(np.float32, copy=False)

    def encode(self, channel: str, phase: np.ndarray) -> np.ndarray:
        corrected = np.remainder(phase + self._correction[channel], 2.0 * np.pi)
        index = np.rint(corrected * (255.0 / (2.0 * np.pi))).astype(np.uint8)
        return self._lut[channel][index]


def _enable_per_monitor_dpi() -> None:
    if os.name != "nt":
        raise HardwareError("Extended-display SLM output is implemented for Windows")
    user32 = ctypes.windll.user32
    try:
        setter = user32.SetProcessDpiAwarenessContext
        setter.argtypes = [ctypes.c_void_p]
        setter.restype = ctypes.c_int
        success = bool(setter(ctypes.c_void_p(-4)))
        getter = user32.GetThreadDpiAwarenessContext
        getter.argtypes = []
        getter.restype = ctypes.c_void_p
        equal = user32.AreDpiAwarenessContextsEqual
        equal.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        equal.restype = ctypes.c_int
        actual = getter()
        per_monitor = bool(equal(actual, ctypes.c_void_p(-4))) or bool(equal(actual, ctypes.c_void_p(-3)))
    except (AttributeError, OSError) as error:
        raise HardwareError("Windows per-monitor DPI awareness API is unavailable") from error
    if not success and not per_monitor:
        raise HardwareError("Cannot enable per-monitor DPI awareness for exact SLM pixels")


def _windows_monitors() -> dict[str, tuple[int, int, int, int]]:
    class RECT(ctypes.Structure):
        _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long), ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

    class MONITORINFOEXW(ctypes.Structure):
        _fields_ = [
            ("cbSize", ctypes.c_ulong),
            ("rcMonitor", RECT),
            ("rcWork", RECT),
            ("dwFlags", ctypes.c_ulong),
            ("szDevice", ctypes.c_wchar * 32),
        ]

    monitors: dict[str, tuple[int, int, int, int]] = {}
    prototype = ctypes.WINFUNCTYPE(
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(RECT),
        ctypes.c_ssize_t,
    )
    user32 = ctypes.windll.user32
    user32.GetMonitorInfoW.argtypes = [ctypes.c_void_p, ctypes.POINTER(MONITORINFOEXW)]
    user32.GetMonitorInfoW.restype = ctypes.c_int
    user32.EnumDisplayMonitors.argtypes = [ctypes.c_void_p, ctypes.c_void_p, prototype, ctypes.c_ssize_t]
    user32.EnumDisplayMonitors.restype = ctypes.c_int

    def callback(handle, _hdc, _rect, _data):
        info = MONITORINFOEXW()
        info.cbSize = ctypes.sizeof(info)
        if not user32.GetMonitorInfoW(handle, ctypes.byref(info)):
            return 0
        rect = info.rcMonitor
        monitors[str(info.szDevice)] = (rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top)
        return 1

    callback_handle = prototype(callback)
    if not user32.EnumDisplayMonitors(None, None, callback_handle, 0):
        raise HardwareError("Windows monitor enumeration failed")
    return monitors


class _SlmDisplays:
    def __init__(self, config: Mapping[str, Any]):
        _enable_per_monitor_dpi()
        import tkinter as tk
        from PIL import Image, ImageTk

        self._tk = tk
        self._Image = Image
        self._ImageTk = ImageTk
        self._root = tk.Tk()
        self._root.withdraw()
        self._windows = []
        self._canvases = []
        self._window_handles: list[ctypes.c_void_p] = []
        self._window_expected: list[tuple[int, int, int, int]] = []
        self._photos: list[Any | None] = [None, None]
        monitors = _windows_monitors()
        user32 = ctypes.windll.user32
        user32.SetWindowPos.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint,
        ]
        user32.SetWindowPos.restype = ctypes.c_int

        class RECT(ctypes.Structure):
            _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long), ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

        user32.GetWindowRect.argtypes = [ctypes.c_void_p, ctypes.POINTER(RECT)]
        user32.GetWindowRect.restype = ctypes.c_int
        user32.GetParent.argtypes = [ctypes.c_void_p]
        user32.GetParent.restype = ctypes.c_void_p
        try:
            for index, slm in enumerate(config["slms"]):
                device = str(slm["monitor_device"])
                if device not in monitors:
                    raise HardwareError(f"Configured monitor is not active: {device}")
                x, y, width, height = monitors[device]
                if (width, height) != (2048, 2048):
                    raise HardwareError(f"{device} is {width} x {height}; exact 2048 x 2048 output is required")
                window = tk.Toplevel(self._root)
                window.overrideredirect(True)
                window.attributes("-topmost", True)
                window.geometry("2048x2048+0+0")
                canvas = tk.Canvas(window, width=2048, height=2048, highlightthickness=0, borderwidth=0)
                canvas.pack(fill="both", expand=False)
                self._windows.append(window)
                self._canvases.append(canvas)
                window.update_idletasks()
                child_hwnd = ctypes.c_void_p(int(window.winfo_id()))
                wrapper_hwnd = user32.GetParent(child_hwnd)
                hwnd = ctypes.c_void_p(wrapper_hwnd) if wrapper_hwnd else child_hwnd
                self._window_handles.append(hwnd)
                self._window_expected.append((x, y, 2048, 2048))
                if not user32.SetWindowPos(
                    hwnd, ctypes.c_void_p(-1), x, y, 2048, 2048, 0x0010 | 0x0040
                ):
                    raise HardwareError(f"Cannot place {device} at virtual-screen coordinate {(x, y)}")
                window.update_idletasks()
                rect = RECT()
                if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                    raise HardwareError(f"Cannot verify the SLM window for {device}")
                actual = (rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top)
                expected = (x, y, 2048, 2048)
                if actual != expected or window.winfo_width() != 2048 or window.winfo_height() != 2048:
                    raise HardwareError(f"SLM window placement/scaling mismatch for {device}: {actual}, expected {expected}")
            self._root.update_idletasks()
            self._root.update()
            for hwnd, expected in zip(self._window_handles, self._window_expected):
                rect = RECT()
                if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                    raise HardwareError("Cannot verify a final SLM window rectangle")
                actual = (rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top)
                if actual != expected:
                    raise HardwareError(f"SLM window moved or scaled after display update: {actual}, expected {expected}")
        except BaseException:
            self.close()
            raise

    def show(self, index: int, pixels: np.ndarray) -> None:
        if pixels.shape != (2048, 2048) or pixels.dtype != np.uint8:
            raise ValueError("SLM pixels must be uint8[2048, 2048]")
        photo = self._ImageTk.PhotoImage(self._Image.fromarray(pixels, mode="L"), master=self._windows[index])
        canvas = self._canvases[index]
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=photo)
        self._photos[index] = photo
        self._root.update_idletasks()
        self._root.update()

    def close(self) -> None:
        root, self._root = getattr(self, "_root", None), None
        if root is not None:
            with contextlib.suppress(Exception):
                root.destroy()


class _SerialRgbSource:
    def __init__(self, settings: Mapping[str, Any]):
        try:
            import serial
        except ImportError as error:
            raise HardwareError("pyserial is required by rgb_source.transport=serial_ascii") from error
        try:
            self._serial = serial.Serial(
                port=str(settings["port"]),
                baudrate=int(settings["baud_rate"]),
                timeout=float(settings["timeout_s"]),
                write_timeout=float(settings["timeout_s"]),
            )
        except Exception as error:
            raise HardwareError(f"Cannot open RGB source on {settings['port']}") from error
        self._settings = settings
        self._encoding = str(settings.get("encoding", "ascii"))

    def _send(self, channel: str, command: str) -> str:
        item = self._settings["channels"][channel]
        try:
            payload = command.format(channel=channel, level=item["level"]).encode(self._encoding)
            self._serial.reset_input_buffer()
            self._serial.write(payload)
            self._serial.flush()
            response = self._serial.readline().decode(self._encoding, errors="replace").strip()
        except Exception as error:
            raise HardwareError(f"RGB source communication failed for {channel}") from error
        exact = item.get("ack_exact")
        pattern = item.get("ack_regex")
        if exact and response != exact:
            raise HardwareError(f"RGB source rejected {channel}: expected {exact!r}, received {response!r}")
        if pattern and re.fullmatch(str(pattern), response) is None:
            raise HardwareError(f"RGB source returned an invalid {channel} ACK: {response!r}")
        return response

    def all_off(self) -> None:
        errors = []
        for channel in CHANNELS:
            try:
                self._send(channel, str(self._settings["channels"][channel]["off_command"]))
            except BaseException as error:
                errors.append(error)
        if errors:
            raise HardwareError("One or more RGB channels did not acknowledge the dark command") from errors[0]

    def enable(self, channel: str) -> str:
        item = self._settings["channels"][channel]
        return self._send(channel, str(item["set_command"]))

    def close(self) -> None:
        serial_port, self._serial = getattr(self, "_serial", None), None
        if serial_port is None:
            return
        try:
            self._serial = serial_port
            self.all_off()
        finally:
            self._serial = None
            serial_port.close()


class _Dcp201Stage:
    def __init__(self, config: Mapping[str, Any], settings: Mapping[str, Any]):
        wrapper = _resolve_file(config, settings["wrapper_path"], "stage.wrapper_path")
        old_directory = Path.cwd()
        name = f"_epi_dcp201_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(name, wrapper)
        if spec is None or spec.loader is None:
            raise HardwareError(f"Cannot load DCP201 wrapper: {wrapper}")
        module = importlib.util.module_from_spec(spec)
        try:
            os.chdir(wrapper.parent)
            spec.loader.exec_module(module)
        except Exception as error:
            raise HardwareError(f"Cannot load DCP201 SDK from {wrapper.parent}") from error
        finally:
            os.chdir(old_directory)
        self._controller = module.DCP201Controller()
        self._settings = settings
        self._axis = int(settings["axis"])
        try:
            self._controller.open(str(settings["port"]), int(settings["baud_rate"]))
            controller = self._controller.get_controller_info()
            stage = self._controller.get_stage_data_0x02(self._axis)
            if str(controller.SN) != str(settings["controller_serial"]):
                raise HardwareError("DCP201 controller serial does not match the configuration")
            if str(stage.SN) != str(settings["stage_serial"]):
                raise HardwareError("DCP201 stage serial does not match the configuration")
            self._hardware_min = float(stage.minPosition)
            self._hardware_max = float(stage.maxPosition)
            if float(settings["soft_min"]) < self._hardware_min or float(settings["soft_max"]) > self._hardware_max:
                raise HardwareError("Configured software limits exceed the stage limits returned by get_stage_data_0x02")
            self.assert_ready()
        except BaseException:
            self.close()
            raise

    def status(self) -> tuple[float, int]:
        try:
            position, status = self._controller.get_status(self._axis)
        except Exception as error:
            raise HardwareError("DCP201 get_status failed") from error
        return float(position), int(status)

    def assert_ready(self) -> tuple[float, int]:
        position, status = self.status()
        if status & LIMIT_MASK:
            raise HardwareError(f"DCP201 limit is active (status=0x{status:08X})")
        if not status & MOTOR_CONNECTED_MASK:
            raise HardwareError(f"DCP201 motor is not connected (status=0x{status:08X})")
        if not status & MOTOR_ENABLED_MASK:
            raise HardwareError(f"DCP201 motor is not enabled (status=0x{status:08X})")
        if self._settings.get("require_homed") and not status & 0x00000400:
            raise HardwareError(f"DCP201 stage is not homed (status=0x{status:08X})")
        return position, status

    def move_to(self, target: float) -> float:
        target = float(target)
        soft_min = float(self._settings["soft_min"])
        soft_max = float(self._settings["soft_max"])
        if not (soft_min <= target <= soft_max):
            raise ConfigurationError(f"Requested stage position {target} exceeds software limits")
        position, status = self.assert_ready()
        tolerance = float(self._settings["position_tolerance"])
        if abs(position - target) <= tolerance and not status & MOVING_MASK:
            return position
        try:
            self._controller.move_absolute_long(target, self._axis)
        except Exception as error:
            raise HardwareError("DCP201 move_absolute_long failed") from error
        deadline = time.monotonic() + float(self._settings["motion_timeout_s"])
        try:
            while time.monotonic() < deadline:
                position, status = self.status()
                if status & LIMIT_MASK:
                    raise HardwareError(f"DCP201 reached a limit (status=0x{status:08X})")
                if not status & MOVING_MASK:
                    if abs(position - target) <= tolerance:
                        return position
                    raise HardwareError(
                        f"DCP201 stopped at {position}, outside tolerance of target {target}"
                    )
                time.sleep(float(self._settings["poll_interval_s"]))
            raise HardwareError(f"DCP201 motion timed out before reaching {target}")
        except BaseException:
            with contextlib.suppress(Exception):
                self._controller.move_stop(1, self._axis)
            raise

    def assert_pose(self, expected: float) -> float:
        position, status = self.assert_ready()
        if status & MOVING_MASK:
            raise HardwareError("DCP201 stage moved during a fixed-scene measurement")
        tolerance = float(self._settings["position_tolerance"])
        if abs(position - float(expected)) > tolerance:
            raise HardwareError(f"DCP201 pose drifted from {expected} to {position}")
        return position

    def close(self) -> None:
        controller, self._controller = getattr(self, "_controller", None), None
        if controller is not None:
            with contextlib.suppress(Exception):
                controller.close()


class _GalaxyCamera:
    def __init__(self, config: Mapping[str, Any], settings: Mapping[str, Any]):
        self._dll_handles = []
        for value in settings.get("dll_directories", []):
            directory = _resolve_directory(config, value, "camera.dll_directories")
            if os.name == "nt" and hasattr(os, "add_dll_directory"):
                self._dll_handles.append(os.add_dll_directory(str(directory)))
        sdk = _resolve_directory(config, settings["sdk_python_dir"], "camera.sdk_python_dir")
        sys.path.insert(0, str(sdk))
        try:
            import gxipy as gx
        except Exception as error:
            raise HardwareError(f"Cannot import Galaxy gxipy from {sdk}") from error
        finally:
            with contextlib.suppress(ValueError):
                sys.path.remove(str(sdk))
        self._gx = gx
        self._camera = None
        self._streaming = False
        self._last_frame_id = None
        self._settings = settings
        self._calibration: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
        try:
            manager = gx.DeviceManager()
            count, devices = manager.update_device_list(int(settings["enumeration_timeout_ms"]))
            serial = str(settings["serial"])
            match = next((item for item in (devices or []) if str(item.get("sn")) == serial), None)
            if count < 1 or match is None:
                raise HardwareError(f"Galaxy camera serial {serial} was not enumerated")
            if settings.get("model") and str(match.get("model_name")) != str(settings["model"]):
                raise HardwareError("Galaxy camera model does not match the configuration")
            self._camera = manager.open_device_by_sn(serial)
            features = self._camera.get_remote_device_feature_control()
            features.get_enum_feature("AcquisitionMode").set("Continuous")
            features.get_enum_feature("TriggerSelector").set("FrameStart")
            features.get_enum_feature("TriggerMode").set("On")
            features.get_enum_feature("TriggerSource").set("Software")
            features.get_enum_feature("PixelFormat").set("Mono8")
            features.get_enum_feature("ExposureAuto").set("Off")
            features.get_enum_feature("GainAuto").set("Off")
            features.get_float_feature("ExposureTime").set(float(settings["exposure_us"]))
            features.get_float_feature("Gain").set(float(settings["gain_db"]))
            self._camera.stream_on()
            self._streaming = True
            calibration = settings["calibration"]
            for channel in CHANNELS:
                dark = _load_numeric(_resolve_file(config, calibration["dark"][channel], f"dark.{channel}"))
                flat = _load_numeric(_resolve_file(config, calibration["flat"][channel], f"flat.{channel}"))
                self._calibration[channel] = (
                    dark.astype(np.float32, copy=False),
                    flat.astype(np.float32, copy=False),
                    float(calibration["radiometric_scale"][channel]),
                )
        except BaseException:
            self.close()
            raise

    def _roi(self, array: np.ndarray) -> np.ndarray:
        x, y, width, height = map(int, self._settings["calibration"]["roi_xywh"])
        if y + height > array.shape[0] or x + width > array.shape[1]:
            raise HardwareError(f"Configured ROI {(x, y, width, height)} exceeds camera frame {array.shape}")
        return array[y : y + height, x : x + width]

    def _calibration_roi(self, array: np.ndarray, raw_shape: tuple[int, int], name: str) -> np.ndarray:
        if array.shape == raw_shape:
            return self._roi(array)
        if array.shape == (256, 256):
            return array
        raise ConfigurationError(f"{name} calibration must match the full camera frame or the 256 x 256 ROI")

    def capture(self, channel: str) -> tuple[np.ndarray, dict[str, int | None]]:
        try:
            stream = self._camera.data_stream[0]
            stream.flush_queue()
            features = self._camera.get_remote_device_feature_control()
            features.get_command_feature("TriggerSoftware").send_command()
            raw = stream.get_image(timeout=int(self._settings["frame_timeout_ms"]))
        except Exception as error:
            raise HardwareError("Galaxy software-trigger acquisition failed") from error
        if raw is None:
            raise HardwareError("Galaxy camera returned no frame before the timeout")
        if raw.get_status() != self._gx.GxFrameStatusList.SUCCESS:
            raise HardwareError(f"Galaxy camera returned frame status {raw.get_status()}")
        array = raw.get_numpy_array()
        if array is None or array.ndim != 2 or array.size == 0:
            raise HardwareError("Galaxy camera returned an empty or non-mono frame")
        if array.dtype != np.uint8:
            raise HardwareError(f"Galaxy Mono8 frame must be uint8, received {array.dtype}")
        frame_id = int(raw.get_frame_id()) if hasattr(raw, "get_frame_id") else None
        frame_timestamp = int(raw.get_timestamp()) if hasattr(raw, "get_timestamp") else None
        if frame_id is not None and self._last_frame_id is not None and frame_id <= self._last_frame_id:
            raise HardwareError(
                f"Galaxy frame counter did not advance ({frame_id} <= {self._last_frame_id}); "
                "reopen the session after a camera reconnect or counter reset"
            )
        self._last_frame_id = frame_id
        raw_float = np.asarray(array, dtype=np.float32).copy()
        roi = self._roi(raw_float)
        dark, flat, scale = self._calibration[channel]
        dark_roi = self._calibration_roi(dark, raw_float.shape, f"dark.{channel}")
        flat_roi = self._calibration_roi(flat, raw_float.shape, f"flat.{channel}")
        denominator = flat_roi - dark_roi
        floor = float(self._settings["calibration"]["denominator_floor"])
        if not np.isfinite(denominator).all() or np.any(denominator <= floor):
            raise HardwareError(f"Flat calibration denominator is invalid for {channel}")
        calibrated = (roi - dark_roi) / denominator
        calibrated = calibrated * scale
        clip = self._settings["calibration"].get("output_clip")
        if clip is not None:
            calibrated = np.clip(calibrated, float(clip[0]), float(clip[1]))
        if calibrated.shape != (256, 256) or not np.isfinite(calibrated).all():
            raise HardwareError("Fixed camera calibration produced an invalid image")
        return calibrated.astype(np.float32, copy=False), {
            "frame_id": frame_id,
            "camera_timestamp": frame_timestamp,
        }

    def close(self) -> None:
        camera, self._camera = getattr(self, "_camera", None), None
        if camera is not None:
            if getattr(self, "_streaming", False):
                with contextlib.suppress(Exception):
                    camera.stream_off()
            with contextlib.suppress(Exception):
                camera.close_device()
        self._streaming = False
        for handle in getattr(self, "_dll_handles", []):
            with contextlib.suppress(Exception):
                handle.close()
        self._dll_handles = []


class HardwareSession:


    def __init__(self, config: Mapping[str, Any] | str | os.PathLike[str], mock: bool = False):
        if isinstance(config, (str, os.PathLike)):
            config = load_config(config)
        self.config = dict(config)
        self.mock = bool(mock)
        validate_config(self.config, real=not self.mock)
        self.config_identity = config_identity(self.config, include_files=not self.mock)
        self._lock = _ExclusiveHardwareLock(_physical_lock_identity(self.config))
        self._source = None
        self._stage = None
        self._camera = None
        self._displays = None
        self._slm_calibration = None
        self._opened = False
        self._scene_key: str | None = None
        self._scene_config_key: str | None = None
        self._scene_pose: float | None = None
        self._configured_feedback = any(
            bool(scene.get("static") and scene.get("replayable"))
            for scene in self.config["scenes"].values()
        )
        self._scene_supports_feedback = self._configured_feedback
        self.last_frame_ids: list[int | None] = []
        self.last_measurement_metadata: dict[str, Any] | None = None
        self._mock_counter = 0

    def __enter__(self) -> "HardwareSession":
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        try:
            self.close()
        except BaseException as close_error:
            if exc_value is None:
                raise
            if hasattr(exc_value, "add_note"):
                exc_value.add_note(f"Hardware cleanup also failed: {close_error}")
        return False

    @property
    def supports_feedback(self) -> bool:
        return bool(self._scene_supports_feedback)

    @property
    def scene_pose(self) -> float | None:
        return self._scene_pose

    def open(self) -> None:
        if self._opened:
            return
        if self.mock:
            self._opened = True
            return
        self._lock.acquire()
        try:
            self._source = _SerialRgbSource(self.config["rgb_source"])
            self._source.all_off()
            self._stage = _Dcp201Stage(self.config, self.config["stage"])
            self._displays = _SlmDisplays(self.config)
            self._slm_calibration = [
                _SlmCalibration(self.config, self.config["slms"][0]),
                _SlmCalibration(self.config, self.config["slms"][1]),
            ]
            self._camera = _GalaxyCamera(self.config, self.config["camera"])
            self._opened = True
        except BaseException:
            self.close()
            raise

    def prepare_scene(self, key: str) -> None:
        if not self._opened:
            raise HardwareError("Open HardwareSession before prepare_scene")
        scene_config_key = key if key in self.config["scenes"] else "*"
        if scene_config_key not in self.config["scenes"]:
            raise ConfigurationError(f"Unknown scene: {key}; add the key or an explicit '*' fallback")
        scene = self.config["scenes"][scene_config_key]
        position = scene.get("stage_position")
        if self.mock:
            self._scene_pose = float(position) if position is not None else None
        elif position is not None:
            self._scene_pose = self._stage.move_to(float(position))
        elif scene.get("operator_confirmed"):
            self._scene_pose = None
        else:
            raise ConfigurationError(f"Scene {key} has neither a stage pose nor operator confirmation")
        self._scene_key = key
        self._scene_config_key = scene_config_key
        self._scene_supports_feedback = bool(scene.get("static") and scene.get("replayable"))

    @staticmethod
    def _phase(value: Any, name: str) -> np.ndarray:
        array = np.asarray(value)
        if array.shape != PHASE_SHAPE:
            raise ValueError(f"{name} must have shape {PHASE_SHAPE}, received {array.shape}")
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError(f"{name} must contain finite numeric phase values in radians")
        return array.astype(np.float32, copy=False)

    def _assert_fixed_pose(self) -> float | None:
        if not self.mock and self._scene_pose is not None:
            self._scene_pose = self._stage.assert_pose(self._scene_pose)
        return self._scene_pose

    def _mock_measure(self, cws: np.ndarray, odr: np.ndarray) -> np.ndarray:
        
        sampled = cws[:, 4::8, 4::8] + odr[:, 4::8, 4::8]
        scene_digest = hashlib.sha256(str(self._scene_key).encode("utf-8")).digest()
        offset = (scene_digest[0] / 255.0 - 0.5) * 0.1
        result = 0.10 + 0.90 * (0.5 + 0.5 * np.cos(sampled + offset))
        self._mock_counter += 1
        self.last_frame_ids = [self._mock_counter * 3 + i for i in range(3)]
        self.last_measurement_metadata = {
            "mock": True,
            "kind": "deterministic_software_mock",
            "scene_key": self._scene_key,
            "scene_config_key": self._scene_config_key,
            "stage_pose": self._scene_pose,
            "synthetic_frame_ids": list(self.last_frame_ids),
            "channel_order": list(CHANNELS),
        }
        return result.astype(np.float32, copy=False)

    def measure_rgb(self, cws_phase: Any, odr_phase: Any) -> np.ndarray:


        if not self._opened:
            raise HardwareError("Open HardwareSession before measure_rgb")
        if self._scene_key is None:
            raise HardwareError("Call prepare_scene before measure_rgb")
        self.last_frame_ids = []
        self.last_measurement_metadata = None
        cws = self._phase(cws_phase, "cws_phase")
        odr = self._phase(odr_phase, "odr_phase")
        if self.mock:
            return self._mock_measure(cws, odr)

        started_ns = time.time_ns()
        pose_before = self._assert_fixed_pose()
        images: list[np.ndarray] = []
        frame_ids: list[int | None] = []
        channel_records: list[dict[str, Any]] = []
        for channel_index, channel in enumerate(CHANNELS):
            self._source.all_off()
            first = self._slm_calibration[0].encode(channel, cws[channel_index])
            second = self._slm_calibration[1].encode(channel, odr[channel_index])
            self._displays.show(0, first)
            self._displays.show(1, second)
            enabled_ns = time.time_ns()
            acknowledgement = self._source.enable(channel)
            try:
                time.sleep(float(self.config["acquisition"]["settle_s"]))
                image, camera_record = self._camera.capture(channel)
            finally:
                self._source.all_off()
            images.append(image)
            frame_ids.append(camera_record["frame_id"])
            channel_records.append(
                {
                    "channel": channel,
                    "source_ack": acknowledgement,
                    "source_enabled_time_ns": enabled_ns,
                    "capture_complete_time_ns": time.time_ns(),
                    "settle_s": float(self.config["acquisition"]["settle_s"]),
                    **camera_record,
                }
            )
            pose_after_channel = self._assert_fixed_pose()
        measured = np.stack(images, axis=0).astype(np.float32, copy=False)
        if measured.shape != OUTPUT_SHAPE or not np.isfinite(measured).all():
            raise HardwareError(f"Measured output must be finite float32{OUTPUT_SHAPE}")
        self.last_frame_ids = frame_ids
        self.last_measurement_metadata = {
            "mock": False,
            "kind": "fresh_camera_acquisition",
            "scene_key": self._scene_key,
            "scene_config_key": self._scene_config_key,
            "stage_pose_before": pose_before,
            "stage_pose_after": pose_after_channel,
            "started_time_ns": started_ns,
            "completed_time_ns": time.time_ns(),
            "channel_order": list(CHANNELS),
            "channels": channel_records,
        }
        return measured

    def close(self) -> None:
        errors = []
        source, self._source = self._source, None
        if source is not None:
            try:
                source.close()
            except BaseException as error:
                errors.append(error)
        camera, self._camera = self._camera, None
        if camera is not None:
            try:
                camera.close()
            except BaseException as error:
                errors.append(error)
        displays, self._displays = self._displays, None
        if displays is not None:
            try:
                displays.close()
            except BaseException as error:
                errors.append(error)
        stage, self._stage = self._stage, None
        if stage is not None:
            try:
                stage.close()
            except BaseException as error:
                errors.append(error)
        self._slm_calibration = None
        self._opened = False
        self._scene_key = None
        self._scene_config_key = None
        self._scene_supports_feedback = self._configured_feedback
        self._lock.release()
        if errors:
            raise HardwareError("One or more devices failed to close cleanly") from errors[0]


def _write_reference(path: Path, force: bool) -> None:
    path = path.expanduser().resolve()
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --force to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reference_config(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote configuration template: {path}")


def _load_phase(path: str, name: str) -> np.ndarray:
    resolved = Path(path).expanduser().resolve()
    array = np.load(resolved, allow_pickle=False)
    return HardwareSession._phase(array, name)


def _append_manifest(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def _safe_sample_relative(sample_id: str) -> Path:


    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ConfigurationError("sample-id must be a non-empty relative identifier")
    normalized = sample_id.replace("\\", "/")
    if "//" in normalized:
        raise ConfigurationError("sample-id cannot contain empty path segments")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or not pure.parts:
        raise ConfigurationError("sample-id must be relative")
    parts = []
    for part in pure.parts:
        if part in {"", ".", ".."}:
            raise ConfigurationError("sample-id cannot contain empty, '.' or '..' path segments")
        if any(ord(character) < 32 or character in '<>:"|?*' for character in part):
            raise ConfigurationError(f"sample-id contains an invalid Windows filename character: {part!r}")
        if part.endswith((" ", ".")):
            raise ConfigurationError(f"sample-id path segments cannot end in a space or period: {part!r}")
        if part.split(".", 1)[0].upper() in _WINDOWS_RESERVED:
            raise ConfigurationError(f"sample-id uses a reserved Windows filename: {part!r}")
        parts.append(part)
    return Path(*parts)


def _dataset_png_path(root: Path, role: str, sample_id: str) -> Path:
    relative = _safe_sample_relative(sample_id)
    destination = root / role / relative.parent / f"{relative.name}.png"
    role_root = (root / role).resolve()
    resolved = destination.resolve()
    try:
        resolved.relative_to(role_root)
    except ValueError as error:
        raise ConfigurationError("sample-id escapes the requested dataset role directory") from error
    return resolved


def _write_dataset_png(path: Path, measured: np.ndarray) -> None:


    from PIL import Image

    clipped = np.clip(measured, 0.0, 1.0)
    rgb = np.rint(np.moveaxis(clipped, 0, -1) * 255.0).astype(np.uint8)
    temporary = path.with_name(path.name + ".writing.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        Image.fromarray(rgb, mode="RGB").save(temporary, format="PNG")
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _acquire(args: argparse.Namespace) -> None:
    if not args.config:
        raise ConfigurationError("--acquire requires --config")
    required = ("scene", "role", "sample_id", "cws_phase", "odr_phase", "output", "manifest")
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        raise ConfigurationError(f"--acquire is missing: {', '.join('--' + name.replace('_', '-') for name in missing)}")
    output = Path(args.output).expanduser().resolve()
    if output.suffix.lower() != ".npy":
        raise ConfigurationError("--output must end in .npy so the raw float calibration remains explicit")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite measured data: {output}")
    dataset_png = None
    if args.dataset_root:
        if args.role == "measurement":
            raise ConfigurationError("--dataset-root is only valid for role=input or role=gt")
        dataset_png = _dataset_png_path(Path(args.dataset_root).expanduser().resolve(), args.role, args.sample_id)
        if dataset_png.exists():
            raise FileExistsError(f"Refusing to overwrite dataset image: {dataset_png}")
    manifest_path = Path(args.manifest).expanduser().resolve()
    if manifest_path == output or (dataset_png is not None and manifest_path == dataset_png):
        raise ConfigurationError("--manifest must be distinct from the NPY and dataset PNG outputs")
    cws_path = Path(args.cws_phase).expanduser().resolve()
    odr_path = Path(args.odr_phase).expanduser().resolve()
    cws = _load_phase(str(cws_path), "cws_phase")
    odr = _load_phase(str(odr_path), "odr_phase")
    config = load_config(args.config)
    with HardwareSession(config, mock=args.mock) as session:
        session.prepare_scene(args.scene)
        measured = session.measure_rgb(cws, odr)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".writing.npy")
        np.save(temporary, measured, allow_pickle=False)
        os.replace(temporary, output)
        dataset_record = None
        if dataset_png is not None:
            _write_dataset_png(dataset_png, measured)
            dataset_record = {
                "path": str(dataset_png),
                "sha256": _sha256_file(dataset_png),
                "encoding": "lossless PNG, RGB uint8",
                "conversion": "clip fixed-calibration values to [0,1], round(value*255); no per-frame normalization",
            }
        record = {
            "schema": "epi-measurement-manifest-v1",
            "measurement_id": uuid.uuid4().hex,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "role": args.role,
            "sample_id": args.sample_id,
            "scene_key": args.scene,
            "stage_pose": session.scene_pose,
            "config_identity": session.config_identity,
            "mock": bool(args.mock),
            "measurement_metadata": session.last_measurement_metadata,
            "frame_ids": session.last_frame_ids,
            "cws_phase": {"path": str(cws_path), "sha256": _sha256_file(cws_path)},
            "odr_phase": {"path": str(odr_path), "sha256": _sha256_file(odr_path)},
            "measurement": {"path": str(output), "sha256": _sha256_file(output), "shape": list(measured.shape)},
            "dataset_png": dataset_record,
        }
        _append_manifest(manifest_path, record)
    print(f"Saved one fresh {args.role} measurement: {output}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Configure, validate and acquire data from an EPI optical setup.')
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write-config", metavar="JSON")
    action.add_argument("--check-config", action="store_true")
    action.add_argument("--mock-test", action="store_true")
    action.add_argument("--acquire", action="store_true")
    parser.add_argument("--config")
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock devices; never import a vendor SDK")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--scene")
    parser.add_argument("--role", choices=("input", "gt", "measurement"))
    parser.add_argument("--sample-id")
    parser.add_argument("--cws-phase")
    parser.add_argument("--odr-phase")
    parser.add_argument("--output")
    parser.add_argument("--manifest")
    parser.add_argument("--dataset-root", help="Also write role=input|gt as DATASET_ROOT/role/sample-id.png")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.write_config:
        _write_reference(Path(args.write_config), args.force)
        return 0
    if args.check_config:
        if not args.config:
            raise ConfigurationError("--check-config requires --config")
        config = load_config(args.config)
        validate_config(config, real=not args.mock)
        print(json.dumps({"valid": True, "real": not args.mock, "config_identity": config_identity(config, include_files=not args.mock)}, indent=2))
        return 0
    if args.mock_test:
        config = load_config(args.config) if args.config else _mock_config()
        validate_config(config, real=False)
        phase = np.zeros(PHASE_SHAPE, dtype=np.float32)
        with HardwareSession(config, mock=True) as session:
            session.prepare_scene(next(iter(config["scenes"])))
            measured = session.measure_rgb(phase, phase)
        print(json.dumps({"mock_only": True, "shape": list(measured.shape), "dtype": str(measured.dtype), "sha256": hashlib.sha256(measured.tobytes()).hexdigest()}, indent=2))
        return 0
    _acquire(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
