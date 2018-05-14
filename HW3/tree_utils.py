def get_number_of_leafs(tree):
    number_of_leafs = 0
    label = list(tree.keys())[0]
    node = tree[label]
    for key in node.keys():
        if type(node[key]).__name__ == 'dict':
            number_of_leafs += get_number_of_leafs(node[key])
        else:
            number_of_leafs += 1

    return number_of_leafs


def get_tree_depth(tree):
    max_depth = 0
    label = list(tree.keys())[0]
    node = tree[label]

    for key in node.keys():
        if type(node[key]).__name__ == 'dict':
            this_depth = 1 + get_tree_depth(node[key])
        else:
            this_depth = 1

        if this_depth > max_depth:
            max_depth = this_depth

    return max_depth
