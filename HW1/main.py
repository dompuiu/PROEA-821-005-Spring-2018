from decision_tree import DecisionTree
from tree_plotter import create_plot
from data_set_loader import DataSetLoader
from data_set_features_enricher import DataSetFeaturesEnricher

labels = [
    'first_name_longer_that_last_name',
    'has_middle_name',
    'first_name_starts_and_ends_with_same_letter',
    'first_name_come_alphabetically_before_their_last_name',
    'second_letter_of_their_first_name_a_vowel',
    'is_the_number_of_last_name_letter_even'
]
d = DataSetLoader('dataset/training.data').load()
d = DataSetFeaturesEnricher(d).get_enrich_data_set(labels)

tree = DecisionTree(d, [
    'fn_longer_ls',
    'middle',
    'f&l',
    'fn_before_ln',
    '2vowel',
    'ln_even'
]).make_tree()
create_plot(tree)

# print('Feature to split:', choose_best_feature_to_split([
#     [False, False, False],
#     [False, False, False],
#     [False, False, False],
#     [False, False, False],
#     [False, False, False],
#     [False, True, False],
#     [False, False, False],
#     [False, True, True],
#     [False, True, False],
#     [False, True, True],
#     [False, True, False],
#     [False, True, False],
#     [False, True, False],
#     [False, True, False],
#     [False, True, False]
# ]))

#
# print(
#     BestFeatureFinder([
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
#     ]).best_feature()
# )

# tree = DecisionTree([
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
#     ], ['Outlook', 'Temperature', 'Humidity', 'Wind']).make_tree()
# print(tree)
# create_plot(tree)
