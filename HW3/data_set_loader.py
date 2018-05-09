from collections import deque


class DataSetLoader:
    def __init__(self, filename):
        self.filename = filename

    def load(self):
        features = []
        labels = []

        with open(self.filename, 'r') as ins:
            for line in ins:
                line = line.rstrip('\n')
                feature_row = {
                    0: 1
                }

                pieces = deque(line.split(' '))
                labels.append(int(pieces.popleft()))

                for feature_data in pieces:
                    feature_parts = feature_data.split(':')
                    if not feature_parts[0]:
                        break

                    key = int(feature_parts[0])
                    value = int(feature_parts[1])
                    feature_row[key] = value

                features.append(feature_row)

        return features, labels
