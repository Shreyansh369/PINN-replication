"""Hardware description and peak-memory measurement (Linux /proc)."""
import os
import platform
import re

import torch


def reset_peak_rss():
    """Reset the kernel's VmHWM counter for this process (Linux >= 4.0). Returns success."""
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


def _status_kb(field):
    try:
        with open("/proc/self/status") as f:
            m = re.search(rf"^{field}:\s+(\d+) kB", f.read(), re.M)
        return int(m.group(1)) if m else None
    except OSError:
        return None


def peak_rss_mb():
    kb = _status_kb("VmHWM")
    return kb / 1024 if kb is not None else float("nan")


def current_rss_mb():
    kb = _status_kb("VmRSS")
    return kb / 1024 if kb is not None else float("nan")


def peak_vram_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 2 ** 20
    return 0.0


def hardware_info():
    cpu = "unknown"
    try:
        with open("/proc/cpuinfo") as f:
            m = re.search(r"model name\s*:\s*(.+)", f.read())
            cpu = m.group(1).strip() if m else cpu
    except OSError:
        pass
    mem = None
    try:
        with open("/proc/meminfo") as f:
            mem = int(re.search(r"MemTotal:\s+(\d+)", f.read()).group(1)) / 2 ** 20
    except (OSError, AttributeError):
        pass
    return {"cpu": cpu, "logical_cpus": os.cpu_count(), "torch_threads": torch.get_num_threads(),
            "ram_gb": mem, "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "python": platform.python_version(), "torch": torch.__version__,
            "platform": f"{platform.system()} {platform.release()} {platform.machine()}"}
