"""Single logical CPU, pre-import thread limits, and audited software FFmpeg children."""
import ctypes
import os
import subprocess

THREAD_VARS = ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'OMP_THREAD_LIMIT']


def configure_environment():
    for name in THREAD_VARS:
        os.environ[name] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['OMP_DYNAMIC'] = 'FALSE'
    os.environ['MKL_DYNAMIC'] = 'FALSE'


def affinity(pid=None):
    if os.name == 'nt':
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_bool, ctypes.c_uint32]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.GetCurrentProcess() if pid is None else kernel.OpenProcess(0x1000, False, pid)
        proc, system = ctypes.c_size_t(), ctypes.c_size_t()
        try:
            if not handle or not kernel.GetProcessAffinityMask(handle, ctypes.byref(proc), ctypes.byref(system)):
                raise ctypes.WinError(ctypes.get_last_error())
            return proc.value
        finally:
            if pid is not None and handle:
                kernel.CloseHandle(handle)
    return sum(1 << cpu for cpu in os.sched_getaffinity(pid or 0))


def pin_cpu(cpu=None):
    allowed = affinity()
    if cpu is None:
        cpu = (allowed & -allowed).bit_length() - 1
    mask = 1 << cpu
    if not mask & allowed:
        raise ValueError(f'CPU {cpu} unavailable in affinity mask {allowed}')
    if os.name == 'nt':
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), mask):
            raise ctypes.WinError(ctypes.get_last_error())
    else:
        os.sched_setaffinity(0, {cpu})
    if affinity() != mask:
        raise RuntimeError('Failed to enforce single-CPU affinity')
    return cpu, mask


def audit_ffmpeg(expected_mask, events):
    original = subprocess.Popen

    class GuardedPopen(original):
        def __init__(self, args, *pos, **kwargs):
            if isinstance(args, (str, bytes)):
                raise ValueError('Shell commands are forbidden inside measured codecs')
            args = list(map(str, args))
            if 'ffmpeg' not in os.path.basename(args[0]).lower():
                raise ValueError(f'Unexpected codec child: {args[0]}')
            for flag in ['-threads', '-filter_threads', '-filter_complex_threads']:
                for i, token in enumerate(args):
                    if token == flag and args[i + 1] != '1':
                        raise ValueError(f'Nonserial FFmpeg option: {flag}')
            for i, token in enumerate(args):
                if token == '-row-mt' and args[i + 1] != '0':
                    raise ValueError('AV1 row-mt must be disabled')
                if token in ['-c:v', '-vcodec'] and args[i + 1] not in ['libaom-av1', 'rawvideo']:
                    raise ValueError(f'Video codec is outside AV1-only experiment: {args[i+1]}')
            args[1:1] = ['-filter_complex_threads', '1', '-hwaccel', 'none']
            super().__init__(args, *pos, **kwargs)
            try:
                child_mask = affinity(self.pid)
                if child_mask != expected_mask:
                    raise RuntimeError(f'Child CPU affinity mismatch: {child_mask} != {expected_mask}')
                events.append({'pid': self.pid, 'affinity_mask': child_mask, 'command': args})
            except BaseException:
                self.kill()
                self.wait()
                raise

    subprocess.Popen = GuardedPopen
    return original
