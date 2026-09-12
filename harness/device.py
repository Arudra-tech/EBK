"""Device telemetry probes and device fingerprint.

GB10 (and any dGPU-style box): NVML via pynvml. Note GB10's unified memory makes
nvmlDeviceGetMemoryInfo return NotSupported — every call is individually guarded.

Jetson Orin: NVML is unreliable → parse ``tegrastats``. ``temp_c`` is the junction
temperature (tj) since that is what throttling keys on. Sysfs is the fallback.

``probe.last`` is always a fully-populated DeviceSample (floats, never None for
the two fields the server requires), and /telemetry only ever reads it — the
probe thread does the polling.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import SETTINGS

log = logging.getLogger("harness.device")


@dataclass
class DeviceSample:
    gpu_util: float = 0.0  # 0..1 fraction (server contract)
    temp_c: float = 0.0
    gpu_temp_c: float | None = None
    power_w: float | None = None
    sm_clock_mhz: int | None = None
    mem_used_mb: float | None = None
    mem_total_mb: float | None = None
    cpu_util: float | None = None
    ts: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["age_s"] = round(time.time() - self.ts, 2)
        return d


def is_jetson() -> bool:
    if Path("/etc/nv_tegra_release").exists():
        return True
    dt = Path("/proc/device-tree/model")
    if dt.exists():
        try:
            m = dt.read_bytes().decode("utf-8", "ignore").lower()
            return "jetson" in m or "orin" in m or "tegra" in m
        except OSError:
            pass
    return False


class DeviceProbe(ABC):
    kind = "null"
    interval_s = 0.5

    def __init__(self):
        self.last = DeviceSample()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=f"probe-{self.kind}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                s = self.sample()
                if s is not None:
                    self.last = s
            except Exception as e:  # never let the probe die
                log.debug("probe sample failed: %s", e)
            self._stop.wait(self.interval_s)

    @abstractmethod
    def sample(self) -> DeviceSample | None: ...

    def info(self) -> dict:
        return {"probe": self.kind}

    def thermal_max_c(self) -> float:
        return SETTINGS.thermal_max_c or 80.0


class NullProbe(DeviceProbe):
    kind = "null"

    def sample(self) -> DeviceSample | None:
        return DeviceSample()


class NvmlProbe(DeviceProbe):
    kind = "nvml"

    def __init__(self):
        super().__init__()
        import pynvml

        self.nvml = pynvml
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(0)

    def _try(self, fn, *a):
        try:
            return fn(self.h, *a)
        except self.nvml.NVMLError:
            return None

    def sample(self) -> DeviceSample | None:
        n = self.nvml
        util = self._try(n.nvmlDeviceGetUtilizationRates)
        temp = self._try(n.nvmlDeviceGetTemperature, n.NVML_TEMPERATURE_GPU)
        power = self._try(n.nvmlDeviceGetPowerUsage)
        clk = self._try(n.nvmlDeviceGetClockInfo, n.NVML_CLOCK_SM)
        mem = self._try(n.nvmlDeviceGetMemoryInfo)  # NotSupported on GB10 (unified memory)
        return DeviceSample(
            gpu_util=(util.gpu / 100.0) if util is not None else 0.0,
            temp_c=float(temp) if temp is not None else 0.0,
            gpu_temp_c=float(temp) if temp is not None else None,
            power_w=(power / 1000.0) if power is not None else None,
            sm_clock_mhz=int(clk) if clk is not None else None,
            mem_used_mb=(mem.used / 2**20) if mem is not None else None,
            mem_total_mb=(mem.total / 2**20) if mem is not None else None,
        )

    def info(self) -> dict:
        n = self.nvml
        name = self._try(n.nvmlDeviceGetName)
        if isinstance(name, bytes):
            name = name.decode()
        drv = None
        try:
            drv = n.nvmlSystemGetDriverVersion()
            drv = drv.decode() if isinstance(drv, bytes) else drv
        except n.NVMLError:
            pass
        maxclk = self._try(n.nvmlDeviceGetMaxClockInfo, n.NVML_CLOCK_SM)
        plimit = self._try(n.nvmlDeviceGetPowerManagementLimit)
        return {
            "probe": self.kind,
            "gpu_name": name,
            "driver": drv,
            "sm_clock_max_mhz": maxclk,
            "power_limit_w": (plimit / 1000.0) if plimit else None,
        }


class TegrastatsProbe(DeviceProbe):
    """Reads a long-running ``tegrastats`` process. Example Orin line:

    RAM 5933/7620MB (lfb 21x1MB) SWAP 5/3810MB (cached 0MB) CPU [0%@1497,...] EMC_FREQ 0%@2133
    GR3D_FREQ 0%@[611] ... cpu@49.906C soc2@48.875C soc0@49.5C gpu@49.468C tj@50.062C soc1@50.062C
    VDD_IN 5313mW/5313mW VDD_CPU_GPU_CV 943mW/943mW VDD_SOC 1731mW/1731mW
    """

    kind = "tegrastats"
    GR3D = re.compile(r"GR3D_FREQ (\d+)%(?:@\[?(\d+)\]?)?")
    TJ = re.compile(r"\btj@([\d.]+)C")
    GPU_T = re.compile(r"\bgpu@([\d.]+)C")
    RAM = re.compile(r"RAM (\d+)/(\d+)MB")
    VDD_IN = re.compile(r"VDD_IN (\d+)mW")
    CPU = re.compile(r"CPU \[([^\]]+)\]")

    def __init__(self, interval_ms: int = 500):
        super().__init__()
        self.interval_ms = interval_ms
        self.proc: subprocess.Popen | None = None
        self.parsed_lines = 0

    def start(self) -> None:
        exe = shutil.which("tegrastats") or "/usr/bin/tegrastats"
        try:
            self.proc = subprocess.Popen(
                [exe, "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except OSError as e:
            log.warning("tegrastats unavailable (%s); using sysfs fallback", e)
            self.proc = None
        super().start()

    def _run(self) -> None:
        if self.proc is None or self.proc.stdout is None:
            # sysfs polling fallback
            while not self._stop.is_set():
                try:
                    s = self.sample()
                    if s is not None:
                        self.last = s
                except Exception as e:
                    log.debug("sysfs sample failed: %s", e)
                self._stop.wait(self.interval_s)
            return
        for line in self.proc.stdout:
            if self._stop.is_set():
                break
            try:
                s = self.parse(line)
                if s is not None:
                    self.last = s
                    self.parsed_lines += 1
            except Exception as e:
                log.debug("tegrastats parse failed: %s", e)

    def stop(self) -> None:
        super().stop()
        if self.proc is not None:
            self.proc.terminate()

    def parse(self, line: str) -> DeviceSample | None:
        m = self.GR3D.search(line)
        if not m:
            return None
        util = int(m.group(1)) / 100.0
        clk = int(m.group(2)) if m.group(2) else None
        tj = self.TJ.search(line)
        gt = self.GPU_T.search(line)
        ram = self.RAM.search(line)
        vin = self.VDD_IN.search(line)
        cpu = self.CPU.search(line)
        cpu_util = None
        if cpu:
            pcts = [int(x.split("%")[0]) for x in cpu.group(1).split(",") if "%" in x and not x.startswith("off")]
            if pcts:
                cpu_util = sum(pcts) / (100.0 * len(pcts))
        temp = float(tj.group(1)) if tj else (float(gt.group(1)) if gt else 0.0)
        return DeviceSample(
            gpu_util=util,
            temp_c=temp,
            gpu_temp_c=float(gt.group(1)) if gt else None,
            power_w=(int(vin.group(1)) / 1000.0) if vin else None,
            sm_clock_mhz=clk,
            mem_used_mb=float(ram.group(1)) if ram else None,
            mem_total_mb=float(ram.group(2)) if ram else None,
            cpu_util=cpu_util,
        )

    def sample(self) -> DeviceSample | None:
        """Sysfs fallback (only used when tegrastats can't be spawned)."""
        util = 0.0
        p = Path("/sys/devices/gpu.0/load")
        if p.exists():
            try:
                util = int(p.read_text().strip()) / 1000.0
            except (OSError, ValueError):
                pass
        temp = 0.0
        gpu_t = None
        for z in Path("/sys/class/thermal").glob("thermal_zone*"):
            try:
                t = (z / "type").read_text().strip()
                v = int((z / "temp").read_text().strip()) / 1000.0
            except (OSError, ValueError):
                continue
            if t.startswith("tj"):
                temp = v
            elif t.startswith("gpu"):
                gpu_t = v
        if temp == 0.0 and gpu_t is not None:
            temp = gpu_t
        return DeviceSample(gpu_util=util, temp_c=temp, gpu_temp_c=gpu_t)

    def info(self) -> dict:
        out: dict = {"probe": self.kind, "tegrastats_lines": self.parsed_lines}
        try:
            out["l4t"] = Path("/etc/nv_tegra_release").read_text().strip().splitlines()[0]
        except OSError:
            pass
        out["power_mode"] = _run_cmd(["nvpmodel", "-q"])
        return out

    def thermal_max_c(self) -> float:
        return SETTINGS.thermal_max_c or 85.0


def _run_cmd(cmd: list[str], timeout: float = 3.0) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = (r.stdout or r.stderr).strip()
        return out[:400] if out else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def detect() -> DeviceProbe:
    if is_jetson():
        log.info("device: Jetson detected → tegrastats probe")
        return TegrastatsProbe()
    try:
        p = NvmlProbe()
        log.info("device: NVML probe (%s)", p.info().get("gpu_name"))
        return p
    except Exception as e:
        log.warning("NVML unavailable (%s) → null probe", e)
        return NullProbe()


def default_profile(probe: DeviceProbe) -> str:
    if SETTINGS.device_profile:
        return SETTINGS.device_profile
    return "edge-lo" if probe.kind == "tegrastats" else "edge-hi"


def device_info(probe: DeviceProbe) -> dict:
    from . import artifacts

    info: dict = {
        "device_key": artifacts.device_key(),
        "profile": default_profile(probe),
        "gpu_name": artifacts.gpu_name(),
        "arch": platform.machine(),
        "hostname": platform.node(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "model": SETTINGS.model,
        "backend_map": SETTINGS.backend_overrides(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name_torch"] = torch.cuda.get_device_name(0)
            cc = torch.cuda.get_device_capability(0)
            info["compute_capability"] = f"{cc[0]}.{cc[1]}"
    except Exception as e:
        info["torch"] = f"unavailable: {e}"
    try:
        import tensorrt

        info["tensorrt"] = tensorrt.__version__
    except Exception:
        info["tensorrt"] = None
    try:
        import ultralytics

        info["ultralytics"] = ultralytics.__version__
    except Exception:
        info["ultralytics"] = None
    try:
        import onnxruntime as ort

        info["onnxruntime"] = ort.__version__
        info["ort_providers"] = ort.get_available_providers()
    except Exception:
        info["onnxruntime"] = None
    info["cpu_count"] = os.cpu_count()
    info.update(probe.info())
    if probe.kind == "tegrastats":
        info["jetson_clocks"] = _run_cmd(["jetson_clocks", "--show"])
    elif probe.kind == "nvml":
        info["nvidia_smi_clocks"] = _run_cmd(["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm", "--format=csv,noheader"])
    return info
