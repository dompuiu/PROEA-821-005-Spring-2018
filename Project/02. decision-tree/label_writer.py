from data_set_loader import DataSetLoader


class LabelWriter:
    def __init__(self, input_features_file, input_ids_file, output_ids_file, clf):
        self.input_features_file = input_features_file
        self.input_ids_file = input_ids_file
        self.output_ids_file = output_ids_file
        self.clf = clf

    def write(self):
        eval_features, _ = DataSetLoader(self.input_features_file).load()
        ids_list = self.read_ids_file()
        predicted = self.clf.predict(eval_features)

        for i in range(len(ids_list)):
            ids_list[i].append(str(predicted[i]))

        with open(self.output_ids_file, mode='w', encoding='utf-8') as myfile:
            myfile.write('Id,Prediction\n' + '\n'.join([','.join(x) for x in ids_list]))

    def read_ids_file(self):
        ids_list = []

        with open(self.input_ids_file, 'r') as ins:
            for line in ins:
                ids_list.append([line.rstrip()])

        return ids_list
