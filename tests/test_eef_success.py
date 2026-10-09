"""Success boundaries and goal reset must stay unchanged when EEF code moves locally."""
import importlib
from types import SimpleNamespace
import numpy as np
import pytest

@pytest.mark.parametrize('task', ['scene', 'puzzle'])
@pytest.mark.parametrize('object_success,distance,threshold,has_goal,expected', [
 (True,.039,.04,True,True), (True,.04,.04,True,False),
 (True,.041,.04,True,False), (False,0.,.04,True,False),
 (True,10.,float('inf'),True,True), (True,10.,.04,False,True),
])
def test_success_predicate(task,object_success,distance,threshold,has_goal,expected,monkeypatch):
 module=importlib.import_module('lewam.envs.'+task+'_env')
 def original(env):env._success=object_success
 monkeypatch.setattr(module,'_ORIG_POST_STEP',original)
 env=SimpleNamespace(_eef_threshold=threshold,_goal_eef=np.zeros(3) if has_goal else None,
                     _pinch_site_id=0,_data=SimpleNamespace(site_xpos=np.array([[distance,0.,0.]])))
 module._post_step(env)
 assert bool(env._success)==expected

@pytest.mark.parametrize('task', ['scene', 'puzzle'])
def test_goal_copy_and_reset(task,monkeypatch):
 module=importlib.import_module('lewam.envs.'+task+'_env');env=SimpleNamespace()
 module.set_goal_effector(env,[1.,2.,3.,4.]);np.testing.assert_array_equal(env._goal_eef,[1.,2.,3.])
 module.set_goal_effector(env,None);assert env._goal_eef is None
 module.set_goal_effector(env,[1.,2.,3.])
 def reset(env,*args,**kwargs):assert env._goal_eef is None;return 'reset result'
 monkeypatch.setattr(module,'_ORIG_RESET',reset)
 assert module._reset(env,seed=42)=='reset result'
