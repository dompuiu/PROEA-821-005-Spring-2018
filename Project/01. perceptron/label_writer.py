from data_set_loader import DataSetLoader
import numpy as np


class LabelWriter:
    def __init__(self, input_features_file, input_ids_file, output_ids_file, w):
        self.input_features_file = input_features_file
        self.input_ids_file = input_ids_file
        self.output_ids_file = output_ids_file
        self.w = w

    def write(self):
        eval_features, _ = DataSetLoader(self.input_features_file).load()
        ids_list = self.read_ids_file()

        for i, feature in enumerate(eval_features):
            ids_list[i].append(LabelWriter.predict(feature, self.w))

        with open(self.output_ids_file, mode='w', encoding='utf-8') as myfile:
            myfile.write('Id,Prediction\n' + '\n'.join([','.join(x) for x in ids_list]))

    def read_ids_file(self):
        ids_list = []

        with open(self.input_ids_file, 'r') as ins:
            for line in ins:
                ids_list.append([line.rstrip()])

        return ids_list

    @staticmethod
    def predict(x, w):
        x = np.array(x)
        if np.dot(x, w) < 0:
            return '0'
        else:
            return '1'
