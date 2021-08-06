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

    def enable_gpu_acceleration(self, device_id=0):
        if(_gpu_available):
            self.np = cupy
            self._using_gpu = True
        else:
            raise Exception("CuPy not found. Verify it is properly installed")
        cupy.cuda.Device(device_id).use()

    def disable_gpu_acceleration(self):
        self.np = numpy
        self._using_gpu = False

    @property
    def using_gpu(self):
        return self._using_gpu
    
    def cust_einsum(self, expr, a, b):
        """Efficient einsum for common expressions"""
        if True:
            if expr == 'npjk,nkr -> npjr':
                n, p, j, k = a.shape
                r = b.shape[-1]
                return self.np.matmul(a.reshape(n, p*j, k), b).reshape(n, p, j, r)
            elif expr == 'npjk,k -> npj':
                return self.np.matmul(a, b)
            elif expr == 'npjr,npjk -> nkr':
                n, p, j, r = a.shape
                k = b.shape[-1]
                return self.np.matmul(
                    b.reshape(n, p*j, k).transpose([0, 2, 1]),
                    a.reshape(n, p*j, r))
            else:
                raise Exception(f"The expression {expr} is not supported by "
                                 "custeinsum")
        return self.np.einsum(expr, a, b)
            

    def to_cpu(self, arr):
        return cupy.asnumpy(arr)

    def to_gpu(self, arr):
        return cupy.asarray(arr)

    def get_device_count(self):
        if _gpu_available:
            return cupy.cuda.runtime.getDeviceCount()
        else:
            return 0


device = Device()
