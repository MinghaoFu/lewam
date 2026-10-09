"""A nonzero CUDA device must not break the real DexMimicGen import."""
import os
import subprocess
import sys

import pytest

@pytest.mark.parametrize('explicit,expected', [(None, '6'), ('7', '7')])
def test_dexmimicgen_egl_device(explicit, expected):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='6,7', MUJOCO_GL='egl')
    env.pop('MUJOCO_EGL_DEVICE_ID', None)
    if explicit is not None:
        env['MUJOCO_EGL_DEVICE_ID'] = explicit
    code = 'import lewam.envs.dexmimicgen_env; import os; assert os.environ["MUJOCO_EGL_DEVICE_ID"] == '+repr(expected)
    result = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
