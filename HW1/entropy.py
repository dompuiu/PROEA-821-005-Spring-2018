from math import log


class Entropy:
    def __init__(self, data_set):
        self.data_set = data_set

        # Computing the count of data set entries.
        self.data_set_entries_count = len(data_set)

    # We create a registry of each label in the data set.
    # Calculate how many times each label is encountered.
    # For a data set like this:
    #   [
    #       ['S', 'H', 'H', 'W', '-'],
    #       ['S', 'H', 'H', 'S', '-'],
    #       ['O', 'H', 'H', 'W', '+'],
    #       ['R', 'M', 'H', 'W', '+'],
    #   ]
    #
    # we will get:
    #   {'-': 2, '+': 2}

    def __get_label_registry(self):
        label_registry = {}

        for feature_vector in self.data_set:
            current_label = feature_vector[-1]
            label_registry[current_label] = label_registry.get(current_label, 0) + 1

        return label_registry

    # Calculate entropy using Shannon's formula.
    def value(self):
        label_registry = self.__get_label_registry()

        entropy_value = 0.0

        for label in label_registry:
            probability = float(label_registry[label]) / self.data_set_entries_count
            entropy_value -= probability * log(probability, 2)

        return entropy_value

# print(
#     Entropy([
#         ['S', 'H', 'H', 'W', '-'],
#         ['S', 'H', 'H', 'S', '-'],
#         ['O', 'H', 'H', 'W', '+'],
#         ['R', 'M', 'H', 'W', '+'],
#         ['R', 'C', 'N', 'W', '+'],
#         ['R', 'C', 'N', 'S', '-'],
#         ['O', 'C', 'N', 'S', '+'],
#         ['S', 'M', 'H', 'W', '-'],
#         ['S', 'C', 'N', 'W', '+'],
#         ['R', 'M', 'N', 'W', '+'],
#         ['S', 'M', 'N', 'S', '+'],
#         ['O', 'M', 'H', 'S', '+'],
#         ['O', 'H', 'N', 'W', '+'],
#         ['R', 'M', 'H', 'S', '-']
#     ]).value()
# )
