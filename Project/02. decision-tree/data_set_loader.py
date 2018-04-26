from collections import deque


class DataSetLoader:
    def __init__(self, filename, size=16):
        self.filename = filename
        self.size = size

    def load(self):
        features = []
        labels = []

        with open(self.filename, 'r') as ins:
            for line in ins:
                feature_row = [0 for _ in range(self.size)]

                pieces = deque(line.split(' '))

                new_label = int(pieces.popleft())
                labels.append(new_label)

                for feature_data in pieces:
                    feature_parts = feature_data.split(':')
                    feature_row[int(feature_parts[0]) - 1] = float(feature_parts[1])

                features.append(feature_row)

        return features, labels
