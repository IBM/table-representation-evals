import psutil
import os
import threading
import time
from functools import wraps
import logging
import pandas as pd
import pynvml
import json

logger = logging.getLogger(__name__)

class ResourceMonitorThread:
    """
    Monitors CPU, RAM, and VRAM for a given PID and its children.
    Runs in a thread inside the same process to reliably capture GPU allocations.
    """

    def __init__(self, pid_to_monitor, sample_interval=0.1):
        self._pid = pid_to_monitor
        self._interval = sample_interval
        self._stop_event = threading.Event()

        self._cpu_samples = []
        self._memory_samples = []
        self._gpu_memory_samples = []

        # NVML setup
        try:
            pynvml.nvmlInit()
            self._nvml_available = True
        except pynvml.NVMLError:
            self._nvml_available = False
            logger.warning("NVML not available. GPU metrics will be disabled.")

    def start(self):
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()

    def _monitor_loop(self):
        parent_proc = psutil.Process(self._pid)
        while not self._stop_event.is_set():
            time.sleep(self._interval)
            try:
                # CPU and RAM
                total_cpu = parent_proc.cpu_percent(interval=None)
                total_mem = parent_proc.memory_info().rss / (1024 ** 2)

                for child in parent_proc.children(recursive=True):
                    try:
                        if child.is_running():
                            total_cpu += child.cpu_percent(interval=None)
                            total_mem += child.memory_info().rss / (1024 ** 2)
                    except psutil.NoSuchProcess:
                        pass

                self._cpu_samples.append(total_cpu)
                self._memory_samples.append(total_mem)

                # GPU VRAM
                if self._nvml_available:
                    tracked_pids = {parent_proc.pid} | {c.pid for c in parent_proc.children(recursive=True)}
                    total_gpu_mem = 0.0
                    for i in range(pynvml.nvmlDeviceGetCount()):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                            if proc.pid in tracked_pids:
                                total_gpu_mem += proc.usedGpuMemory / (1024 ** 2)
                    self._gpu_memory_samples.append(total_gpu_mem)
            except psutil.NoSuchProcess:
                break

    def get_metrics(self):
        metrics = {}
        # CPU
        if self._cpu_samples:
            metrics["peak_cpu (%)"] = max(self._cpu_samples)
            metrics["average_cpu (%)"] = sum(self._cpu_samples) / len(self._cpu_samples)
        # RAM
        if self._memory_samples:
            metrics["peak_memory (MB)"] = max(self._memory_samples)
            metrics["average_memory (MB)"] = sum(self._memory_samples) / len(self._memory_samples)
        # GPU
        if self._gpu_memory_samples:
            metrics["peak_gpu_memory (MB)"] = max(self._gpu_memory_samples)
            metrics["average_gpu_memory (MB)"] = sum(self._gpu_memory_samples) / len(self._gpu_memory_samples)
        return metrics


def monitor_resources(sample_interval=0.1, post_buffer=0.2):
    """
    Decorator to monitor CPU, RAM, and VRAM of a function.
    post_buffer: seconds to keep monitoring after function returns to catch async GPU allocations.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            monitor = ResourceMonitorThread(pid_to_monitor=os.getpid(), sample_interval=sample_interval)
            monitor.start()
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()

            # Keep monitoring a bit longer for async GPU allocations
            time.sleep(post_buffer)
            monitor.stop()

            resource_metrics = {"function": func.__name__, "execution_time (s)": end_time - start_time}
            resource_metrics.update(monitor.get_metrics())
            return result, resource_metrics
        return wrapper
    return decorator


def save_resource_metrics_to_disk(cfg, resource_metrics_setup, resource_metrics_task):
    # Label functions explicitly
    resource_metrics_setup["function"] = "model_setup"
    resource_metrics_task["function"] = "task_inference"

    all_metrics = [resource_metrics_setup, resource_metrics_task]

    # --- CSV raw ---
    df = pd.DataFrame(all_metrics)
    df["approach"] = cfg.approach.approach_name
    df.to_csv("resource_metrics.csv", index=False)

    # --- CSV formatted ---
    combined = {"approach": cfg.approach.approach_name}
    for d in all_metrics:
        fn = d["function"]
        for k, v in d.items():
            if k != "function":
                combined[(fn, k)] = v
    pd.DataFrame([combined]).to_csv("resource_metrics_formatted.csv", index=False)

    # --- JSON setup ---
    with open("resource_metrics_setup.json", "w") as f:
        json.dump(resource_metrics_setup, f, indent=4)

    # --- JSON task ---
    with open("resource_metrics_task.json", "w") as f:
        json.dump(resource_metrics_task, f, indent=4)

    logger.info("Saved resource metrics: CSV + JSON for setup and task.")
