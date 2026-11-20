"""Thin NVML helpers, incl. total energy via raw function pointer (old pynvml lacks the wrapper)."""
import ctypes, pynvml as n
n.nvmlInit()
def handle(i): return n.nvmlDeviceGetHandleByIndex(i)
def energy_mJ(h):
    fn = n._nvmlGetFunctionPointer("nvmlDeviceGetTotalEnergyConsumption")
    v = ctypes.c_ulonglong(); ret = fn(h, ctypes.byref(v)); n._nvmlCheckReturn(ret); return v.value
def throttle(h):
    f = getattr(n, 'nvmlDeviceGetCurrentClocksThrottleReasons', None) or getattr(n, 'nvmlDeviceGetCurrentClocksEventReasons')
    return f(h)
SW_POWER, HW_SLOW = 0x4, 0x8
