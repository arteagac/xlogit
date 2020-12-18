# -*- coding: utf-8 -*-
from xlogit import device
import pytest

_gpu_available = False
try:
    import cupy
    _gpu_available = True
except ImportError:
    pass


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
