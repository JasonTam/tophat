class Config(object):
    def __init__(self, f):
        self.__config = {}
        exec(open(f, 'r').read(), self.__config)

        self.validate()

    def validate(self):
        assert True

    def get(self, key, default=None):
        return self.__config.get(key, default)

    def has_key(self, key):
        return self.__config.has_key(key)
