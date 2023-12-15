class BoundingBox:
    def __init__(self, min_point, max_point):
        self.min_point = min_point
        self.max_point = max_point

    def get_coordinates(self):
        return (
            self.min_point['r'],
            self.min_point['c'],
            self.max_point['r'],
            self.max_point['c']
        )
