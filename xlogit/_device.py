import numpy
_gpu_available = False
try:
    import cupy
    _gpu_available = True
except ImportError:
    pass


class Device():
    def __init__(self):
        self.np = numpy
        self._using_gpu = False
        if _gpu_available:
            self.np = cupy
            self._using_gpu = True

    def enable_gpu_acceleration(self):
        if(_gpu_available):
            self.np = cupy
            self._using_gpu = True
        else:
            print("CuPy not found. Verify it is properly installed")
    
    def disable_gpu_acceleration(self):
        self.np = numpy
        self._using_gpu = False

    @property
    def using_gpu(self):
        return self._using_gpu

    def to_cpu(self, arr):
        return cupy.asnumpy(arr)

    def to_gpu(self, arr):
        return cupy.asarray(arr)


device = Device()
