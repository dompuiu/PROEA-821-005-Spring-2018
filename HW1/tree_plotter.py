import matplotlib.pyplot as plt
from tree_utils import get_tree_depth, get_number_of_leafs

decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plot_node(node_text, center_point, parent_point, node_type):
    create_plot.ax1.annotate(
        node_text,
        xy=parent_point,
        xycoords='axes fraction',
        xytext=center_point, textcoords='axes fraction',
        va="center",
        ha="center",
        bbox=node_type,
        arrowprops=arrow_args
    )


def plot_mid_text(center_point, parent_point, text):
    x_mid = (parent_point[0] - center_point[0]) / 2.0 + center_point[0]
    y_mid = (parent_point[1] - center_point[1]) / 2.0 + center_point[1]
    create_plot.ax1.text(x_mid, y_mid, text)


def plot_tree(tree, parent_point, text):
    number_of_leafs = get_number_of_leafs(tree)
    get_tree_depth(tree)

    label = list(tree.keys())[0]

    center_point = (plot_tree.xOff + (1.0 + float(number_of_leafs)) / 2.0 / plot_tree.totalW, plot_tree.yOff)
    plot_mid_text(center_point, parent_point, text)
    plot_node(label, center_point, parent_point, decision_node)

    node = tree[label]
    plot_tree.yOff = plot_tree.yOff - 1.0 / plot_tree.totalD

    for key in node.keys():
        if type(node[key]).__name__ == 'dict':
            plot_tree(node[key], center_point, str(key))
        else:
            plot_tree.xOff = plot_tree.xOff + 1.0 / plot_tree.totalW
            plot_node(node[key], (plot_tree.xOff, plot_tree.yOff), center_point, leaf_node)
            plot_mid_text((plot_tree.xOff, plot_tree.yOff), center_point, str(key))

    plot_tree.yOff = plot_tree.yOff + 1.0 / plot_tree.totalD


def create_plot(in_tree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()

    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)

    plot_tree.totalW = float(get_number_of_leafs(in_tree))
    plot_tree.totalD = float(get_tree_depth(in_tree))
    plot_tree.xOff = -0.5 / plot_tree.totalW; plot_tree.yOff = 1.0;
    plot_tree(in_tree, (0.5, 1.0), '')
    plt.show()
