class DataSetLoader:
    def __init__(self, filename):
        self.filename = filename

    def load(self):
        array = []
        with open(self.filename, 'r') as ins:
            for line in ins:
                array.append([line.rstrip()[2:], line[0]])

        return array
