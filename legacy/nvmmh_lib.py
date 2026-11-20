"""nvMatmulHeuristics helpers. Config retrieval via the official Python wrapper; silicon metrics via the documented C API
(nvMatmulHeuristicsEstimateSiliconMetricEx takes an nvmmhBackend_t handle; the shipped wrapper 0.1.0.27 passes the
target enum instead and segfaults)."""
import ctypes
from nvMatmulHeuristics.nvMatmulHeuristics import (NvMatmulHeuristicsInterfaceEx, NvMatmulHeuristicsTarget, NvMatmulHeuristicsNvidiaGpu,
                                                    NvMatmulHeuristicsMatmulLayout as L, NvMatmulHeuristicsFlags as F, NvMatmulHeuristicsSiliconMetric as SM)


class NVMMH:
    def __init__(self, flags=F.PERF_MODEL_BASED_AUTO_TUNING):
        self.h = NvMatmulHeuristicsInterfaceEx(backend=NvMatmulHeuristicsTarget.TRITON, flags=flags, load_discovery_implicitly=True,
                                               gpu=NvMatmulHeuristicsNvidiaGpu.RTX_6000_ADA)
        self.backend = self.h.createBackend(NvMatmulHeuristicsTarget.TRITON)
        f = self.h.lib.nvMatmulHeuristicsEstimateSiliconMetricEx
        f.restype = ctypes.c_double
        f.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        self._metric = f

    def problem(self, M, N, K, layout):
        return self.h.makeNvMatmulHeuristicsProblem(M, N, K, L.NN_ROW_MAJOR if layout == 'AB' else L.NT_ROW_MAJOR)

    def configs(self, M, N, K, layout, prec, count):
        p = self.problem(M, N, K, layout)
        return p, self.h.get(p, count, precision=prec)

    def metric(self, p, kcfg, metric, prec):
        hd = self.h.hardware_descriptor
        return self._metric(self.h.handle, prec.encode('ascii'), ctypes.cast(self.backend, ctypes.c_void_p), ctypes.byref(p),
                            ctypes.byref(kcfg), int(metric), ctypes.cast(hd, ctypes.c_void_p) if hd else None)
