# -*- coding: utf-8 -*-
from xlogit import device
import numpy as np
import pytest

_gpu_available = False
try:
    import cupy
    _gpu_available = True
except ImportError:
    pass

def test_cust_einsum():
    """
    Ensure that the custom einsum operation matches numpy's einsum
    """
    N, J, K, R = 10, 4, 3, 5
    njk = np.arange(N*J*K).reshape(N, J, K)
    nkr = np.arange(N*K*R).reshape(N, K, R)
    njr = np.arange(N*J*R).reshape(N, J, R)
    nj = np.arange(N*J).reshape(N, J)
    k = np.arange(K)
    
    assert np.array_equal(device.cust_einsum('njk,nkr -> njr', njk, nkr),
                          np.einsum('njk,nkr -> njr', njk, nkr))
    assert np.array_equal(device.cust_einsum('njk,k -> nj', njk, k),
                          np.einsum('njk,k -> nj', njk, k))
    assert np.array_equal(device.cust_einsum('njr,njk -> nkr', njr, njk),
                          np.einsum('njr,njk -> nkr', njr, njk))

def test_disable_gpu_acceleration():
    """
    Ensure xlogit properly disables GPU and uses numpy
    """
    device.disable_gpu_acceleration()
    assert not device.using_gpu
    assert device.np.__name__ == 'numpy'


def test_enable_gpu_acceleration():
    """
    Test that when CuPy is not available, it is not possible to enable
    GPU acceleration. This test assumes that CuPy is not installed
    """
    if not _gpu_available:
        with pytest.raises(Exception):
            device.enable_gpu_acceleration()
    else:
        pass


def test_get_device_count():
    """
    Test that when CuPy is not available, 0 GPU devices are returned.
    This test assumes that CuPy is not installed
    """
    count = device.get_device_count()
    if not _gpu_available:
        assert count == 0
    else:
        assert count > 0
