# ShadowBoxer/core/memory.py
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import gc
import sys
from typing import Any, List, Tuple, Union

import torch

from .env import _canonicalize_device

_VRAM_TRIM_THRESHOLD_BYTES = int(28.35 * (1024**3))
_MIN_RECLAIMABLE_CACHE_BYTES = 64 * (1024**2)
_MAX_INACTIVE_SPLIT_BYTES = 512 * (1024**2)


class _DXGI_ADAPTER_DESC1(ctypes.Structure):
    _fields_ = [
        ("Description", ctypes.c_wchar * 128),
        ("VendorId", ctypes.c_uint),
        ("DeviceId", ctypes.c_uint),
        ("SubSysId", ctypes.c_uint),
        ("Revision", ctypes.c_uint),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", ctypes.c_longlong),
        ("Flags", ctypes.c_uint),
    ]


class _DXGI_QUERY_VIDEO_MEMORY_INFO(ctypes.Structure):
    _fields_ = [
        ("Budget", ctypes.c_ulonglong),
        ("CurrentUsage", ctypes.c_ulonglong),
        ("AvailableForReservation", ctypes.c_ulonglong),
        ("CurrentReservation", ctypes.c_ulonglong),
    ]


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_ulong),
        ("Data2", ctypes.c_ushort),
        ("Data3", ctypes.c_ushort),
        ("Data4", ctypes.c_ubyte * 8),
    ]


@dataclass
class WDDMMemorySnapshot:
    dedicated_total_gb: float
    dedicated_used_gb: float
    dedicated_avail_gb: float
    shared_budget_gb: float
    shared_used_gb: float
    shared_delta_gb: float
    pt_allocated_gb: float
    pt_reserved_gb: float
    pt_reclaimable_gb: float
    is_spilling: bool
    pt_inactive_split_gb: float = 0.0
    contiguous_avail_gb: float = 0.0
    host_total_gb: float = 0.0
    host_avail_gb: float = 0.0


class WDDMMemoryGuard:
    def __init__(self, device: Union[str, torch.device]):
        self.device = _canonicalize_device(device)
        self.active = False
        self.adapter_name = "Generic GPU"
        self.luid_key = ""
        self.dedicated_total_gb = 0.0
        self.shared_budget_gb = 0.0
        self.shared_baseline_gb = 0.0
        self._adapter3 = ctypes.c_void_p()

        if sys.platform == "win32" and self.device.type == "cuda":
            self._init_win32_telemetry()

    @staticmethod
    def _call_com(
        ptr: ctypes.c_void_p, index: int, argtypes: List[Any], *args: Any
    ) -> int:
        vtable = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_void_p))[0]
        func_addr = ctypes.cast(vtable, ctypes.POINTER(ctypes.c_void_p))[index]
        func = ctypes.WINFUNCTYPE(ctypes.c_long, *argtypes)(func_addr)
        return int(func(ptr, *args))

    @staticmethod
    def _query_host_ram_gb() -> Tuple[float, float]:
        if sys.platform != "win32":
            return 0.0, 0.0
        try:
            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)) != 0:
                return (
                    float(stat.ullTotalPhys) / (1024**3),
                    float(stat.ullAvailPhys) / (1024**3),
                )
        except Exception:
            pass
        return 0.0, 0.0

    def _init_win32_telemetry(self) -> None:
        try:
            iid_factory1 = _GUID(
                0x770AAE78,
                0xF26F,
                0x4DBA,
                (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87),
            )
            iid_adapter3 = _GUID(
                0x645967A4,
                0x1392,
                0x4310,
                (ctypes.c_ubyte * 8)(0xA7, 0x98, 0x80, 0x53, 0xCE, 0x3E, 0x93, 0xFD),
            )
            factory = ctypes.c_void_p()
            if (
                ctypes.windll.dxgi.CreateDXGIFactory1(
                    ctypes.byref(iid_factory1), ctypes.byref(factory)
                )
                != 0
            ):
                return

            target_hw_idx = self.device.index if self.device.index is not None else 0
            hw_seen = 0
            adapter_idx = 0
            while True:
                adapter = ctypes.c_void_p()
                hr = self._call_com(
                    factory,
                    12,
                    [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)],
                    adapter_idx,
                    ctypes.byref(adapter),
                )
                if hr < 0:
                    break

                desc = _DXGI_ADAPTER_DESC1()
                self._call_com(
                    adapter,
                    10,
                    [ctypes.c_void_p, ctypes.POINTER(_DXGI_ADAPTER_DESC1)],
                    ctypes.byref(desc),
                )
                if not (desc.Flags & 2) and desc.DedicatedVideoMemory > 0:
                    if hw_seen == target_hw_idx:
                        high = (desc.AdapterLuid >> 32) & 0xFFFFFFFF
                        low = desc.AdapterLuid & 0xFFFFFFFF
                        self.luid_key = f"0x{high:08x}_0x{low:08x}".lower()
                        self.adapter_name = str(desc.Description)
                        self.dedicated_total_gb = desc.DedicatedVideoMemory / (1024**3)
                        self.shared_budget_gb = desc.SharedSystemMemory / (1024**3)

                        adapter3 = ctypes.c_void_p()
                        hr_qi = self._call_com(
                            adapter,
                            0,
                            [
                                ctypes.c_void_p,
                                ctypes.POINTER(_GUID),
                                ctypes.POINTER(ctypes.c_void_p),
                            ],
                            ctypes.byref(iid_adapter3),
                            ctypes.byref(adapter3),
                        )
                        if hr_qi >= 0 and adapter3.value:
                            self._adapter3 = adapter3
                            self.active = True
                        self._call_com(adapter, 2, [ctypes.c_void_p])
                        break
                    hw_seen += 1
                self._call_com(adapter, 2, [ctypes.c_void_p])
                adapter_idx += 1

            self._call_com(factory, 2, [ctypes.c_void_p])
            if self.active:
                _, shr_used = self._query_video_memory_info()
                self.shared_baseline_gb = shr_used
        except Exception:
            self.active = False

    def _query_video_memory_info(self) -> Tuple[float, float]:
        if not self.active or not self._adapter3.value:
            return 0.0, 0.0
        local_info = _DXGI_QUERY_VIDEO_MEMORY_INFO()
        non_local_info = _DXGI_QUERY_VIDEO_MEMORY_INFO()
        ded_gb = 0.0
        shr_gb = 0.0
        hr_local = self._call_com(
            self._adapter3,
            14,
            [
                ctypes.c_void_p,
                ctypes.c_uint,
                ctypes.c_int,
                ctypes.POINTER(_DXGI_QUERY_VIDEO_MEMORY_INFO),
            ],
            0,
            0,
            ctypes.byref(local_info),
        )
        if hr_local >= 0:
            ded_gb = float(local_info.CurrentUsage) / (1024**3)

        hr_non_local = self._call_com(
            self._adapter3,
            14,
            [
                ctypes.c_void_p,
                ctypes.c_uint,
                ctypes.c_int,
                ctypes.POINTER(_DXGI_QUERY_VIDEO_MEMORY_INFO),
            ],
            0,
            1,
            ctypes.byref(non_local_info),
        )
        if hr_non_local >= 0:
            shr_gb = float(non_local_info.CurrentUsage) / (1024**3)
            if non_local_info.Budget > 0:
                self.shared_budget_gb = float(non_local_info.Budget) / (1024**3)
        return ded_gb, shr_gb

    def recalibrate_baseline(self) -> None:
        if self.active:
            _, shr_used = self._query_video_memory_info()
            self.shared_baseline_gb = shr_used

    def poll(self) -> WDDMMemorySnapshot:
        pt_alloc_bytes = 0
        pt_res_bytes = 0
        pt_split_bytes = 0
        if self.device.type == "cuda" and torch.cuda.is_available():
            pt_alloc_bytes = torch.cuda.memory_allocated(self.device)
            pt_res_bytes = torch.cuda.memory_reserved(self.device)
            try:
                stats = torch.cuda.memory_stats(self.device)
                pt_split_bytes = int(
                    stats.get("inactive_split_bytes.all.current", 0)
                )
            except Exception:
                pt_split_bytes = 0

        pt_alloc_gb = pt_alloc_bytes / (1024**3)
        pt_res_gb = pt_res_bytes / (1024**3)
        pt_reclaim_gb = max(0.0, pt_res_gb - pt_alloc_gb)
        pt_split_gb = pt_split_bytes / (1024**3)
        host_total_gb, host_avail_gb = self._query_host_ram_gb()

        if self.active:
            ded_used, shr_used = self._query_video_memory_info()
            ded_avail = max(0.0, self.dedicated_total_gb - ded_used)
            shr_delta = max(0.0, shr_used - self.shared_baseline_gb)
            contig_avail = ded_avail + max(0.0, pt_reclaim_gb - pt_split_gb)
            is_spilling = shr_delta > 0.25 or ded_avail < 1.00
            return WDDMMemorySnapshot(
                dedicated_total_gb=self.dedicated_total_gb,
                dedicated_used_gb=ded_used,
                dedicated_avail_gb=ded_avail,
                shared_budget_gb=self.shared_budget_gb,
                shared_used_gb=shr_used,
                shared_delta_gb=shr_delta,
                pt_allocated_gb=pt_alloc_gb,
                pt_reserved_gb=pt_res_gb,
                pt_reclaimable_gb=pt_reclaim_gb,
                is_spilling=is_spilling,
                pt_inactive_split_gb=pt_split_gb,
                contiguous_avail_gb=contig_avail,
                host_total_gb=host_total_gb,
                host_avail_gb=host_avail_gb,
            )

        free_b, total_b = (0, 0)
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                free_b, total_b = torch.cuda.mem_get_info(self.device)
            except Exception:
                total_b = int(32 * (1024**3))
                free_b = max(0, total_b - pt_res_bytes)
        ded_total = total_b / (1024**3)
        ded_avail = free_b / (1024**3)
        ded_used = max(0.0, ded_total - ded_avail)
        contig_avail = ded_avail + max(0.0, pt_reclaim_gb - pt_split_gb)
        return WDDMMemorySnapshot(
            dedicated_total_gb=ded_total,
            dedicated_used_gb=ded_used,
            dedicated_avail_gb=ded_avail,
            shared_budget_gb=0.0,
            shared_used_gb=0.0,
            shared_delta_gb=0.0,
            pt_allocated_gb=pt_alloc_gb,
            pt_reserved_gb=pt_res_gb,
            pt_reclaimable_gb=pt_reclaim_gb,
            is_spilling=False,
            pt_inactive_split_gb=pt_split_gb,
            contiguous_avail_gb=contig_avail,
            host_total_gb=host_total_gb,
            host_avail_gb=host_avail_gb,
        )

    def enforce_stage_budget(
        self,
        required_headroom_gb: float = 1.25,
        force: bool = False,
    ) -> WDDMMemorySnapshot:
        snap = self.poll()
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return snap

        reserved_bytes = int(snap.pt_reserved_gb * (1024**3))
        split_bytes = int(snap.pt_inactive_split_gb * (1024**3))
        reclaim_bytes = int(snap.pt_reclaimable_gb * (1024**3))
        adaptive_trim_bytes = (
            min(
                _VRAM_TRIM_THRESHOLD_BYTES,
                int(snap.dedicated_total_gb * 0.90 * (1024**3)),
            )
            if snap.dedicated_total_gb > 0.0
            else _VRAM_TRIM_THRESHOLD_BYTES
        )
        low_dedicated = snap.dedicated_avail_gb < 1.00
        needs_eviction = (
            force
            or snap.is_spilling
            or (snap.contiguous_avail_gb < required_headroom_gb)
            or (
                reserved_bytes > adaptive_trim_bytes
                and reclaim_bytes >= _MIN_RECLAIMABLE_CACHE_BYTES
            )
            or (low_dedicated and split_bytes > _MAX_INACTIVE_SPLIT_BYTES)
        )
        if needs_eviction:
            gc.collect()
            reclaimable_bytes = max(
                0,
                torch.cuda.memory_reserved(self.device)
                - torch.cuda.memory_allocated(self.device),
            )
            if (
                force
                or snap.is_spilling
                or reclaimable_bytes >= _MIN_RECLAIMABLE_CACHE_BYTES
            ):
                torch.cuda.synchronize(self.device)
                torch.cuda.empty_cache()
                snap = self.poll()
        return snap

    def close(self) -> None:
        if self.active and self._adapter3.value:
            try:
                self._call_com(self._adapter3, 2, [ctypes.c_void_p])
            except Exception:
                pass
        self._adapter3 = ctypes.c_void_p()
        self.active = False

    def __del__(self) -> None:
        self.close()


__all__ = [
    "_VRAM_TRIM_THRESHOLD_BYTES",
    "_MIN_RECLAIMABLE_CACHE_BYTES",
    "_MAX_INACTIVE_SPLIT_BYTES",
    "_DXGI_ADAPTER_DESC1",
    "_DXGI_QUERY_VIDEO_MEMORY_INFO",
    "_MEMORYSTATUSEX",
    "_GUID",
    "WDDMMemorySnapshot",
    "WDDMMemoryGuard",
]