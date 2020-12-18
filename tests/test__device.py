# -*- coding: utf-8 -*-
from xlogit import device


def test_disable_gpu_acceleration():
    """
    Ensure xlogit properly disables GPU and uses numpy
    """
    device.disable_gpu_acceleration()
    assert not device.using_gpu
    assert device.np.__name__ == 'numpy'
