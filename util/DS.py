class InvertedIndex:
    def __init__(self):
        self.vec = {}

    def __getitem__(self, key):
        if key not in self.vec:
            self.vec[key] = []
        return self.vec[key]

    # def push_back(self, key, value):
    #     if key not in self.vec:
    #         self.vec[key] = []
    #     self.vec[key].append(value)


class Array:
    def __init__(self):
        self.a = {}

    def __getitem__(self, key):
        if key not in self.a:
            self.a[key] = Array()
        return self.a[key]

    def __setitem__(self, key, value):
        self.a[key] = value
        return self.a[key]


class TwoDMap:
    def __init__(self):
        self.a = {}

    def __getitem__(self, key):
        self.a.setdefault(key, {})
        return self.a[key]
