from enum import Enum
from types import ModuleType
from inspect import isclass


class Config(object):
    def __init__(self, f):
        tmp_d = {}
        exec(open(f, 'r').read(), tmp_d)

        self._params = {
            k: v for k, v in tmp_d.items()
            if not k.startswith('__') and
               not isclass(v) and
               not isinstance(v, ModuleType)
        }

        self.validate()

    def validate(self):
        assert True

    def get(self, key, default=None):
        return self._params.get(key, default)

    def has_key(self, key):
        return key in self._params

    def to_dict(self):
        return recursive_dicter(self._params)


def recursive_dicter(obj, classkey=None, depth=0):
    if depth > 10:
        return 'MAX DEPTH'
    if isinstance(obj, dict):
        data = {}
        for (k, v) in obj.items():
            if isinstance(k, Enum):
                key = k.name
            else:
                key = str(k)
            data[key] = recursive_dicter(v, classkey, depth=depth + 1)
        return data
    elif isinstance(obj, Enum):
        return obj.name
    elif hasattr(obj, "_ast"):
        return recursive_dicter(obj._ast(), depth=depth + 1)
    elif isinstance(obj, bytes):
        return f"b'{obj.decode()}'"
    elif hasattr(obj, "__iter__") and \
            not isinstance(obj, str):
        return [recursive_dicter(v, classkey, depth=depth + 1) for v in obj]
    elif hasattr(obj, "__dict__"):
        data = dict([(key, recursive_dicter(value, classkey, depth=depth + 1))
                     for key, value in obj.__dict__.items()
                     if
                     not callable(value) and not key.startswith('_')])
        if classkey is not None and hasattr(obj, "__class__"):
            data[classkey] = obj.__class__.__name__
        return data
    elif callable(obj):
        return None
    else:
        return obj
