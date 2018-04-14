from collections import deque


class DataSetLoader:
    def __init__(self, filename, size=68):
        self.filename = filename
        self.size = size

    def load(self):
        features = []
        labels = []

        with open(self.filename, 'r') as ins:
            for line in ins:
                feature_row = [1 if i == 0 else 0 for i in range(self.size + 1)]

                pieces = deque(line.split(' '))
                labels.append(int(pieces.popleft()))

                for feature_data in pieces:
                    feature_parts = feature_data.split(':')
                    feature_row[int(feature_parts[0])] = float(feature_parts[1])

                features.append(feature_row)

        return features, labels
