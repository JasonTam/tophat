class Config(object):
    def __init__(self, f):
        tmp_d = {}
        exec(open(f, 'r').read(), tmp_d)

        self._params = {
            k: v for k, v in tmp_d.items()
            if not k.startswith('__')
        }

        self.validate()

    def validate(self):
        assert True

    def get(self, key, default=None):
        return self._params.get(key, default)

    def has_key(self, key):
        return key in self._params
