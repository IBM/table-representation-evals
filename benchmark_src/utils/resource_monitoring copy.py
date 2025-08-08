from functools import wraps
import psutil
import os
import GPUtil  # Install with: pip install gputil
import threading
import time
import math # Import math for isnan check


class ResourceMonitor:
    def __init__(self, sample_interval=0.1):
        self._monitoring = False
        self._sample_interval = sample_interval
        self._cpu_samples = []
        self._memory_samples = []
        self._gpu_util_samples = []
        self._gpu_memory_samples = []
        self._thread = None
        self._gpu_available = self._check_gpu_availability()

    def _check_gpu_availability(self):
        try:
            GPUtil.getGPUs()
            return True
        except:
            return False

    def start(self):
        if not self._monitoring:
            self._monitoring = True
            self._thread = threading.Thread(target=self._monitor)
            self._thread.start()

    def stop(self):
        if self._monitoring:
            self._monitoring = False
            if self._thread:
                self._thread.join()

    def _monitor(self):
        process = psutil.Process(os.getpid())
        while self._monitoring:
            try:
                # CPU usage
                self._cpu_samples.append(psutil.cpu_percent(interval=None))

                # Memory usage (RAM)
                self._memory_samples.append(process.memory_info().rss / (1024 * 1024)) # MB

                # GPU usage (if available)
                if self._gpu_available:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # Assuming you have at least one GPU
                        self._gpu_util_samples.append(gpu.load * 100)
                        self._gpu_memory_samples.append(gpu.memoryUsed)

            except Exception as e:
                # Handle potential errors during sampling (e.g., if a resource becomes unavailable)
                print(f"Error during resource monitoring: {e}")

            time.sleep(self._sample_interval)

    def get_metrics(self):
        """Calculates and returns peak and average metrics."""
        metrics = {}

        # CPU metrics
        if self._cpu_samples:
            metrics["peak_cpu_usage"] = max(self._cpu_samples)
            metrics["average_cpu_usage"] = sum(self._cpu_samples) / len(self._cpu_samples)

        # Memory metrics
        if self._memory_samples:
            metrics["peak_memory_usage_mb"] = max(self._memory_samples)
            metrics["average_memory_usage_mb"] = sum(self._memory_samples) / len(self._memory_samples)

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
                metrics["peak_gpu_memory_usage_mb"] = max(valid_gpu_memory)
                metrics["average_gpu_memory_usage_mb"] = sum(valid_gpu_memory) / len(valid_gpu_memory)

        return metrics


def monitor_resources(func):
    """Decorator to measure peak and average resource usage during function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        monitor = ResourceMonitor(sample_interval=0.01)  # Adjust sample_interval as needed
        monitor.start()

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        
        monitor.stop()

        resource_metrics = monitor.get_metrics()
        resource_metrics["execution_time"] = end_time - start_time

        # You can add logging here if you prefer to log the metrics
        print(f"Function '{func.__name__}' execution time: {resource_metrics['execution_time']:.4f} seconds")
        print("Resource Metrics:")
        for key, value in resource_metrics.items():
            print(f"  {key}: {value:.2f}")

        return result, resource_metrics

    return wrapper