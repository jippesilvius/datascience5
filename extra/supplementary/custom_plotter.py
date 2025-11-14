from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np


#### plot a fancy tree 
# source: https://github.com/Senbon-Sakura/DecisionTree/blob/master/PlotTree.py

def plot_node(node_txt, center_pt, parent_pt, node_type):
    plt.annotate(node_txt, xy=parent_pt, xycoords='axes fraction',
                 xytext=center_pt, textcoords='axes fraction',
                 va="center", ha="center", bbox=node_type, arrowprops=dict(arrowstyle="<-"))

def plot_mid_text(cntr_pt, parent_pt, txt_string):
    x_mid = (parent_pt[0] - cntr_pt[0]) / 2.0 + cntr_pt[0]
    y_mid = (parent_pt[1] - cntr_pt[1]) / 2.0 + cntr_pt[1]
    plt.text(x_mid, y_mid, txt_string, va="center", ha="center", rotation=30)

def plot_tree(my_tree, parent_pt, node_txt):
    num_leafs = get_num_leafs(my_tree)
    depth = get_tree_depth(my_tree)
    first_str = list(my_tree.keys())[0]
    cntr_pt = (plot_tree.x_off + (1.0 + float(num_leafs)) / 2.0 / plot_tree.total_w, plot_tree.y_off)
    plot_mid_text(cntr_pt, parent_pt, node_txt)
    plot_node(first_str, cntr_pt, parent_pt, decision_node)
    second_dict = my_tree[first_str]
    plot_tree.y_off = plot_tree.y_off - 1.0 / plot_tree.total_d
    for key in second_dict.keys():
        if isinstance(second_dict[key], dict):
            plot_tree(second_dict[key], cntr_pt, str(key))
        else:
            plot_tree.x_off = plot_tree.x_off + 1.0 / plot_tree.total_w
            plot_node(second_dict[key], (plot_tree.x_off, plot_tree.y_off), cntr_pt, leaf_node)
            plot_mid_text((plot_tree.x_off, plot_tree.y_off), cntr_pt, str(key))
    plot_tree.y_off = plot_tree.y_off + 1.0 / plot_tree.total_d

def get_num_leafs(my_tree):
    num_leafs = 0
    first_str = list(my_tree.keys())[0]
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if isinstance(second_dict[key], dict):
            num_leafs += get_num_leafs(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs

def get_tree_depth(my_tree):
    max_depth = 0
    first_str = list(my_tree.keys())[0]
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if isinstance(second_dict[key], dict):
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth

decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")


def create_plot(in_tree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plot_tree.total_w = float(get_num_leafs(in_tree))
    plot_tree.total_d = float(get_tree_depth(in_tree))
    plot_tree.x_off = -0.5 / plot_tree.total_w
    plot_tree.y_off = 1.0
    plot_tree(in_tree, (0.5, 1.0), '')
    plt.show()



# print decision tree
def print_tree(tree, depth=0):
    """A simple function to print the decision tree."""
    if not isinstance(tree, dict):
        print("\t" * depth + str(tree))
        return
    for key, value in tree.items():
        print("\t" * depth + str(key))
        if isinstance(value, dict):
            for sub_key, sub_tree in value.items():
                print("\t" * (depth + 1) + str(sub_key) + " -> ", end="")
                print_tree(sub_tree, depth + 2)
        else:
            print("\t" * (depth + 1) + str(value))



### plot a decision boundary

def decision_boundery(classifier, X_train, y_train, X_test, y_test):
    # source: https://medium.com/analytics-vidhya/decision-boundary-for-classifiers-an-introduction-cc67c6d3da0e
    # Create a meshgrid for visualization using two features
    x_min, x_max = X_train[:, 2].min() - 1, X_train[:, 2].max() + 1
    y_min, y_max = X_train[:, 3].min() - 1, X_train[:, 3].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    # Prepare a full feature set for prediction
    # Using the means of the other features to keep them constant
    X_vis = np.zeros((xx.ravel().shape[0], X_train.shape[1]))
    X_vis[:, 2] = xx.ravel()  # Petal Length
    X_vis[:, 3] = yy.ravel()  # Petal Width
    X_vis[:, 0] = X_train[:, 0].mean()  # Sepal Length (constant)
    X_vis[:, 1] = X_train[:, 1].mean()  # Sepal Width (constant)

    # Predict on the meshgrid
    Z = classifier.predict(X_vis)
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary and data points
    plt.contourf(xx, yy, Z, alpha=0.3, levels=[-0.5, 0.5, 1.5, 2.5], cmap=plt.cm.RdYlBu)

    # Plot the training data points
    plt.scatter(X_train[:, 2], X_train[:, 3], c=y_train, edgecolors='black', marker='o', cmap=plt.cm.RdYlBu, label='Train')

    # Plot the test data points
    plt.scatter(X_test[:, 2], X_test[:, 3], c=y_test, edgecolors='black', marker='x', cmap=plt.cm.RdYlBu, label='Test')

    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.title('Decision Boundary of Custom Gaussian Naive Bayes on Iris Dataset')
    plt.legend(title="Dataset")
    plt.show()
