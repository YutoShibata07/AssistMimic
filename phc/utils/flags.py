__all__ = ['flags', 'summation']

class Flags(object):
    def __init__(self, items):
        for key, val in items.items():
            setattr(self,key,val)

flags = Flags({
    'test': False,
    'debug': False,
    "real_traj": False,
    "im_eval": False,
    "save_rollout": False,  # Flag to save successful rollout data (dof, root pos/rot)
    })
