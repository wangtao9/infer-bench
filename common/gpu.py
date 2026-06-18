"""GPU 显存后台监控（nvidia-ml-py / pynvml）。

使用 pynvml 每 N 毫秒采样一次 GPU 已用显存。

对于 vLLM/SGLang 等预分配引擎，推理前后显存不变（启动时已全部预分配），
此时 peak_vram_abs_mb（峰值绝对占用量）比 peak_vram_mb（增量）更有意义。
对于 Transformers 等进程内引擎，两者都有价值。
"""

import logging
import threading

import pynvml  # nvidia-ml-py (the maintained package; the old "pynvml" package is deprecated)

logger = logging.getLogger(__name__)


class GPUMonitor:
    """后台线程采样 GPU 显存占用。

    用法:
        monitor = GPUMonitor(device_index=0, interval_ms=100)
        monitor.start()                  # 开始后台采样（记录基线）
        # ... 运行测试 ...
        monitor.stop()                   # 停止采样
        peak_vram_mb = monitor.peak_vram_mb          # 峰值增量
        peak_vram_abs_mb = monitor.peak_vram_abs_mb  # 峰值绝对占用量
    """

    def __init__(self, device_index: int = 0, interval_ms: int = 100):
        self._device_index = device_index
        self._interval_s = interval_ms / 1000.0
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._baseline_mb: float = 0.0
        self._samples: list[float] = []
        self._running = False

        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        logger.info("GPU monitor initialized: device %d", device_index)

    def _read_vram_mb(self) -> float:
        """读取当前 GPU 已用显存 (MB)。"""
        info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        return info.used / (1024 * 1024)

    def _sampling_loop(self):
        """后台采样循环。"""
        while not self._stop_event.is_set():
            try:
                vram = self._read_vram_mb()
                self._samples.append(vram)
            except Exception as e:
                logger.warning("GPU sample failed: %s", e)
            self._stop_event.wait(self._interval_s)

    def start(self, reset_baseline: bool = True):
        """启动后台显存采样。

        Args:
            reset_baseline: 是否重新记录基线值。设为 False 可保留之前
                的基线（例如在模型加载前 start 并锁住基线，之后
                反复 start/stop 采样时不再覆盖基线）。
        """
        self._samples = []
        if reset_baseline:
            self._baseline_mb = self._read_vram_mb()
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(target=self._sampling_loop, daemon=True)
        self._thread.start()
        logger.info("GPU monitoring started (baseline: %.0f MB)", self._baseline_mb)

    def stop(self):
        """停止后台显存采样。"""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._running = False
        logger.info("GPU monitoring stopped (%d samples)", len(self._samples))

    @property
    def peak_vram_mb(self) -> float:
        """返回峰值显存增量 (MB) = 采样峰值 − 基线。

        对 vLLM/SGLang 等预分配引擎，此值可能为 0（推理前后显存不变）。
        对 Transformers 等进程内引擎，此值反映推理期间的显存增长。
        """
        if not self._samples:
            return 0.0
        return max(self._samples) - self._baseline_mb

    @property
    def peak_vram_abs_mb(self) -> float:
        """返回峰值显存绝对占用量 (MB)。

        无论引擎是否预分配，此值始终有意义。
        对 vLLM/SGLang：反映模型 + KV cache 的总占用。
        对 Transformers：反映推理期间的峰值占用。
        """
        if not self._samples:
            return 0.0
        return max(self._samples)

    @property
    def baseline_mb(self) -> float:
        """返回基线显存 (MB)。"""
        return self._baseline_mb

    @property
    def is_running(self) -> bool:
        return self._running

    def __del__(self):
        if self._running:
            self.stop()
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass