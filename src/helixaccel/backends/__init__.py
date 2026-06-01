from helixaccel.backends.rapids_gpu import RapidsGPUBackend
from helixaccel.backends.scanpy_cpu import ScanpyCPUBackend

BACKENDS = {
    ScanpyCPUBackend.name: ScanpyCPUBackend,
    RapidsGPUBackend.name: RapidsGPUBackend,
}
