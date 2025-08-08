import psutil
import os
import GPUtil
import multiprocessing
import threading
import time
import math
from functools import wraps
import logging
import pandas as pd

logger = logging.getLogger(__name__)

class ResourceMonitorProcess:
    def __init__(self, pid_to_monitor, metrics_queue, ready_event, stop_event, sample_interval=0.1):
        self._pid_to_monitor = pid_to_monitor
        self._metrics_queue = metrics_queue
        self._ready_event = ready_event
        self._stop_event = stop_event  
        self._sample_interval = sample_interval
        self._cpu_samples = []
        self._memory_samples = []
        self._gpu_util_samples = []
        self._gpu_memory_samples = []
        self._gpu_available = self._check_gpu_availability()

    def _check_gpu_availability(self):
        try:
            GPUtil.getGPUs()
            return True
        except:
            return False

    def run(self):
        try:
            parent_process = psutil.Process(self._pid_to_monitor)
            logger.debug(f"Have in total {len(parent_process.children(recursive=True))} children")

            # Prime cpu_percent
            parent_process.cpu_percent(interval=None)
            logger.debug(f"Sample interval: {self._sample_interval}")
            self._ready_event.set()

            while not self._stop_event.is_set():  # Check for the stop signal
                try:
                    time.sleep(self._sample_interval)
                    total_cpu_percent = parent_process.cpu_percent(interval=None)
                    total_memory_usage = parent_process.memory_info().rss / (1024 * 1024) # Or memory_info().rss

                    # Monitor child processes recursively
                    for child in parent_process.children(recursive=True):
                        try:
                            if child.is_running(): # Check if the child process is still alive
                                total_cpu_percent += child.cpu_percent(interval=None)
                                total_memory_usage += child.memory_info().rss / (1024 * 1024)
                        except psutil.NoSuchProcess:
                            pass # Child process terminated, ignore

                    self._cpu_samples.append(total_cpu_percent)
                    self._memory_samples.append(total_memory_usage)

                    # GPU usage (if available)
                    if self._gpu_available:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            # TODO
                            # Assuming you have a single GPU or want to monitor the first one
                            gpu = gpus[0]
                            self._gpu_util_samples.append(gpu.load * 100)
                            self._gpu_memory_samples.append(gpu.memoryUsed)


                except psutil.NoSuchProcess:
                    # The monitored process has terminated, so stop monitoring
                    break

            # Put the collected metrics onto the queue before stopping
            self._metrics_queue.put({
                'cpu_samples': self._cpu_samples,
                'memory_samples': self._memory_samples,
                'gpu_util_samples': self._gpu_util_samples,
                'gpu_memory_samples': self._gpu_memory_samples
            })

            self._metrics_queue.put(None) 

        except Exception as e:
            logger.error(f"Error in ResourceMonitorProcess: {e}")
            self._metrics_queue.put(None) 


def monitor_resources(sample_interval=0.1):
    def decorator(func): 
        """Decorator to measure peak and average resource usage during function execution."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            multiprocessing.set_start_method('spawn', force=True)

            metrics_queue = multiprocessing.Queue()
            ready_event = multiprocessing.Event()
            stop_event = multiprocessing.Event()

            # Create and start the monitor process
            monitor_process = multiprocessing.Process(
                target=ResourceMonitorProcess(
                    pid_to_monitor=os.getpid(),  # Monitor the current process
                    metrics_queue=metrics_queue,
                    stop_event=stop_event,
                    ready_event=ready_event,
                    sample_interval=sample_interval
                ).run
            )
            monitor_process.start()
            ready_event.wait()

            # Run the function to be monitored
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            logger.debug(f"Finished running Function '{func.__name__}'")

            # Signal the monitor process to stop and wait for it to finish
            stop_event.set()
            logger.debug(f"Sent stop event")

            # Collect metrics (and consume the queue to avoid deadlock)
            all_metrics = []
            while True:
                metric = metrics_queue.get()
                if metric is None: # Check for the sentinel value
                    break
                all_metrics.append(metric)

            if len(all_metrics) > 0:
                collected_metrics = all_metrics[0] # is a single dictionary
            else:
                logger.error("No metrics collected.")
                collected_metrics = None

            monitor_process.join(timeout=10)
            logger.debug(f"Finished joining of processes")

            # Retrieve metrics from the queue
            resource_metrics = {}
            resource_metrics["function"] = func.__name__
            resource_metrics["execution_time (s)"] = end_time - start_time
            if collected_metrics is not None:
                resource_metrics.update(get_metrics(collected_metrics))
            
                logger.info(f"Function '{func.__name__}' execution time: {resource_metrics['execution_time (s)']:.4f} seconds")
                logger.debug("Resource Metrics:")
                for key, value in resource_metrics.items():
                    if type(value) != str:
                        logger.debug(f"  {key}: {value:.4f}")

            del monitor_process

            return result, resource_metrics
        return wrapper
    return decorator

def get_metrics(collected_metrics):
        """Calculates and returns peak and average metrics."""
        metrics = {}

        # CPU metrics
        if collected_metrics["cpu_samples"]:
            cpu_samples = collected_metrics["cpu_samples"]
            metrics["peak_cpu (%)"] = max(cpu_samples)
            metrics["average_cpu (%)"] = sum(cpu_samples) / len(cpu_samples)

        # Memory metrics
        if collected_metrics["memory_samples"]:
            memory_samples = collected_metrics["memory_samples"]
            metrics["peak_memory (MB)"] = max(memory_samples)
            metrics["average_memory (MB)"] = sum(memory_samples) / len(memory_samples)

        if False:
            # GPU metrics (if available)
            if self._gpu_available and self._gpu_util_samples:
                # Ensure there are no NaN values before calculating max/average
                valid_gpu_util = [u for u in self._gpu_util_samples if not math.isnan(u)]
                if valid_gpu_util:
                    metrics["peak_gpu_utilization"] = max(valid_gpu_util)
                    metrics["average_gpu_utilization"] = sum(valid_gpu_util) / len(valid_gpu_util)

            if self._gpu_available and self._gpu_memory_samples:
                # Ensure there are no NaN values before calculating max/average
                valid_gpu_memory = [m for m in self._gpu_memory_samples if not math.isnan(m)]
                if valid_gpu_memory:
                    metrics["peak_gpu_memory_mb"] = max(valid_gpu_memory)
                    metrics["average_gpu_memory_mb"] = sum(valid_gpu_memory) / len(valid_gpu_memory)

        return metrics


def save_resource_metrics_to_disk(cfg, resource_metrics_setup: dict, resource_metrics_task: dict):
    resource_metrics_setup["function"] = "model_setup"
    resource_metrics_task["function"] = "task_inference"
    all_resource_metrics = [resource_metrics_setup, resource_metrics_task]
    resource_metrics_df = pd.DataFrame(all_resource_metrics)
    resource_metrics_df["approach"] = cfg.approach.approach_name
    resource_metrics_df.to_csv(f"resource_metrics.csv", index=False)

    resource_metrics_combined = {"approach": cfg.approach.approach_name}

    for d in all_resource_metrics:
        function = d["function"]
        for key, value in d.items():
            if key != "function":
                resource_metrics_combined[(function, key)] = value

    resource_df = pd.DataFrame([resource_metrics_combined])
    resource_df.to_csv(f"resource_metrics_formatted.csv", index=False)