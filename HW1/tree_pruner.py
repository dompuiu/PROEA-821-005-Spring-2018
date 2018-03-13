class TreePruner:
    def __init__(self, tree):
        self.tree = tree

    def prune(self):
        return self.__prune_tree(self.tree)

    def __prune_tree(self, tree, level=0):
        label = list(tree.keys())[0]
        node = tree[label]

        # Prune all the inner trees
        for key in node.keys():
            if type(node[key]).__name__ == 'dict':
                v = self.__prune_tree(node[key], level+1)
                node[key] = v

        # Check if tress still exist after pruning
        for key in node.keys():
            if type(node[key]).__name__ == 'dict':
                return tree

        node_values = list(node.values())
        if len(set(node_values)) == 1 and level > 0:
            return node_values[0]

        return tree
