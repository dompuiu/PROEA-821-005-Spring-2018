from collections import deque
from scipy.sparse import csr_matrix
import numpy as np


class DataSetLoader:
    def __init__(self, filename, size=67692):
        self.filename = filename
        self.size = size

    def load(self, zeros=False):
        labels = []

        rows = []
        cols = []
        row = 0

        with open(self.filename, 'r') as ins:
            for line in ins:
                line = line.rstrip('\n')

                # Add bias term
                cols.append(0)
                rows.append(row)

                pieces = deque(line.split(' '))
                label = int(pieces.popleft())
                label = 0 if zeros and label == -1 else label

                labels.append(label)

                for feature_data in pieces:
                    feature_parts = feature_data.split(':')
                    if not feature_parts[0] or int(feature_parts[0]) > self.size:
                        break

                    rows.append(row)
                    cols.append(int(feature_parts[0]))

                row += 1

        data = np.ones(len(cols))

        return csr_matrix((data, (rows, cols)), shape=(row, self.size + 1), dtype=np.float128), \
               csr_matrix(labels, dtype=np.float128)
