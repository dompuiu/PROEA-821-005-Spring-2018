class DataSetClassifier:
    def __init__(self, classifier, data_set_enricher, verbose = False):
        self.classifier = classifier
        self.data_set_enricher = data_set_enricher
        self.verbose = verbose

    def classify_data_set(self, plain_data_set):
        invalid_entry = 0
        for original_entry in plain_data_set:
            enrich_feature = self.data_set_enricher.get_enriched_feature(original_entry)

            classifier_response = self.classifier.classify(enrich_feature)
            if classifier_response != original_entry[1]:
                invalid_entry += 1

            if self.verbose:
                print(original_entry[0],
                      ' has the result ',
                      original_entry[1],
                      '. Classifier detects: ',
                      classifier_response,
                      '.'
                      )

        data_set_length = len(plain_data_set)
        error = (invalid_entry / data_set_length) * 100
        print('Invalid classified entries:', invalid_entry, '\nTotal entries:',
              data_set_length, '\nError:', str(round(error, 2)) + '%')
