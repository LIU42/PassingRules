class TrafficSignal:
    def __init__(self, x, y, w, h, color_index, shape_index):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color_index = color_index
        self.shape_index = shape_index

    def __str__(self):
        return f'box: [{self.x}, {self.y}, {self.w}, {self.h}] color: {self.color} shape: {self.shape}'

    @property
    def color(self):
        if self.color_index == 0:
            return 'red'
        if self.color_index == 1:
            return 'green'
        if self.color_index == 2:
            return 'yellow'

    @property
    def shape(self):
        if self.shape_index == 0:
            return 'full'
        if self.shape_index == 1:
            return 'left'
        if self.shape_index == 2:
            return 'right'
        if self.shape_index == 3:
            return 'straight'

    @property
    def center_x(self):
        return self.x + (self.w >> 1)

    @property
    def center_y(self):
        return self.y + (self.h >> 1)

    @property
    def x1(self):
        return self.x

    @property
    def y1(self):
        return self.y

    @property
    def x2(self):
        return self.x + self.w

    @property
    def y2(self):
        return self.y + self.h

    @staticmethod
    def from_bbox(bbox, color_index):
        x, y, w, h = bbox
        return TrafficSignal(x, y - 80, w, h, color_index, None)


class PassingDirects:
    def __init__(self, left, right, straight):
        self.left = left
        self.right = right
        self.straight = straight

    def __str__(self):
        return f'left: {self.left} straight: {self.straight} right: {self.right}'

    @staticmethod
    def allow():
        return PassingDirects(True, True, True)

    @staticmethod
    def prohibit():
        return PassingDirects(False, True, False)
