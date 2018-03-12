from decision_tree import DecisionTree
from decision_tree_plotter import create_plot

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

tree = DecisionTree([
        ['S', 'H', 'H', 'W', '-'],
        ['S', 'H', 'H', 'S', '-'],
        ['O', 'H', 'H', 'W', '+'],
        ['R', 'M', 'H', 'W', '+'],
        ['R', 'C', 'N', 'W', '+'],
        ['R', 'C', 'N', 'S', '-'],
        ['O', 'C', 'N', 'S', '+'],
        ['S', 'M', 'H', 'W', '-'],
        ['S', 'C', 'N', 'W', '+'],
        ['R', 'M', 'N', 'W', '+'],
        ['S', 'M', 'N', 'S', '+'],
        ['O', 'M', 'H', 'S', '+'],
        ['O', 'H', 'N', 'W', '+'],
        ['R', 'M', 'H', 'S', '-']
    ], ['Outlook', 'Temperature', 'Humidity', 'Wind']).make_tree()
print(tree)
create_plot(tree)
