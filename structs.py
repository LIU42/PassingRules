from functools import cache


class TrafficSignal:

    def __init__(self, x, y, width, height, color, shape=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.shape = shape

    def __str__(self):
        return f'Box: [{self.x}, {self.y}, {self.width}, {self.height}] Color: {self.color} Shape: {self.shape}'

    def __eq__(self, value):
        if not isinstance(value, TrafficSignal):
            return False
        if self.x != value.x:
            return False
        if self.y != value.y:
            return False
        if self.width != value.width:
            return False
        if self.height != value.height:
            return False
        return True

    def __hash__(self):
        return hash(self.x) + hash(self.y) + hash(self.width) + hash(self.height)

    @property
    @cache
    def center_x(self):
        return int(self.x + self.width * 0.5)

    @property
    @cache
    def center_y(self):
        return int(self.y + self.height * 0.5)

    @property
    @cache
    def x1(self):
        return int(self.x)

    @property
    @cache
    def y1(self):
        return int(self.y)

    @property
    @cache
    def x2(self):
        return int(self.x + self.width)

    @property
    @cache
    def y2(self):
        return int(self.y + self.height)


class PassingRules:

    def __init__(self, strategy='conservative'):
        assert strategy == 'radical' or strategy == 'conservative'

        if strategy == 'radical':
            self.straight = True
            self.left = True
            self.right = True
        else:
            self.straight = False
            self.left = False
            self.right = True

    def __str__(self):
        return f'Straight: {str(self.straight):<8} Left: {str(self.left):<8} Right: {str(self.right):<8}'

    def allow_all(self):
        self.straight = True
        self.left = True
        self.right = True

    def forbid_all(self):
        self.straight = False
        self.left = False
        self.right = True
