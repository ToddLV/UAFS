import os
from functools import wraps

DEFAULT_ROOT = './materials'


datasets = {}
def register(name):
    @wraps(name)
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


def make(name, **kwargs):
    if kwargs.get('root_path') is None:
        kwargs['root_path'] = os.path.join(DEFAULT_ROOT, name)
    dataset = datasets[name](**kwargs)
    return dataset

