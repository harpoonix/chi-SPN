import numpy as np
from sklearn import tree
from sklearn.tree._export import _tree

from models.decisionTree.DecisionTree import DecisionTree
import argparse
import pickle

import torch

from ciSPN.E3_helpers import get_E3_experiment_name
from ciSPN.datasets.tabularDataset import TabularDataset
from descriptions.description import get_data_description
from environment import environment, get_dataset_paths
from helpers.configuration import Config
from helpers.determinism import make_deterministic
from libs.pawork.log_redirect import PrintLogger

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=606)

parser.add_argument("--model", choices=["DTSciKit"], default='DTSciKit')
parser.add_argument("--dataset", choices=["ASIA", "CANCER", "EARTHQUAKE", "CHAIN"], default="ASIA")
parser.add_argument("--score", choices=["GiniIndex"], default="GiniIndex")
parser.add_argument("--score_alpha", default=None)
parser.add_argument("--provide_interventions", choices=["true", "false"], default="true")  # provide intervention vector during training and use inintervened data
cli_args = parser.parse_args()


conf = Config()
conf.model_name = cli_args.model
conf.batch_size = 1000
conf.seed = cli_args.seed
conf.dataset = cli_args.dataset
conf.score = cli_args.score
conf.score_alpha = None if cli_args.score_alpha is None else float(cli_args.score_alpha)
conf.provide_interventions = (cli_args.provide_interventions.lower() == "true")

# provide the same CLI as E3_DT_train - but score_alpha is only relevant for GICL score
assert conf.score_alpha is None



make_deterministic(conf.seed)

# setup experiments folder
runtime_loss_base_dir = environment["experiments"]["base"] / "E1" / "runtimes"
runtime_base_dir = environment["experiments"]["base"] / "E3" / "runtimes"
log_base_dir = environment["experiments"]["base"] / "E3" / "logs"


experiment_name = get_E3_experiment_name(conf.dataset, conf.model_name, conf.seed, conf.score, loss_params=conf.score_alpha,
                                         provide_interventions=conf.provide_interventions)
save_dir = runtime_base_dir / experiment_name
save_dir.mkdir(exist_ok=True, parents=True)

# redirect logs
log_path = log_base_dir / (experiment_name + ".txt")
log_path.parent.mkdir(exist_ok=True, parents=True)
logger = PrintLogger(log_path)


print("Arguments:", cli_args)

torch.set_grad_enabled(False)

# setup dataset
X_vars, Y_vars, interventionProvider = get_data_description(conf.dataset, no_interventions=not conf.provide_interventions)
dataset_paths = get_dataset_paths(conf.dataset, "train", no_interventions=not conf.provide_interventions)
dataset = TabularDataset(dataset_paths, X_vars, Y_vars, conf.seed, part_transformer=interventionProvider)

num_condition_vars = dataset.X.shape[1]
num_target_vars = dataset.Y.shape[1]


x, y = dataset.get_all_data()
x = x.cpu().numpy()
y = y.cpu().numpy()
clf = tree.DecisionTreeClassifier()
clf.fit(x, y)


# transform scikit tree into out tree class


def tree_from_scikit(sktree):
    tree = DecisionTree(root_node_name="root")
    _node_from_scikit_node(sktree.tree_, 0, tree.get_root())
    return tree


def _node_from_scikit_node(sktree, skid, node):
    left_child = sktree.children_left[skid]
    right_child = sktree.children_right[skid]

    if left_child == _tree.TREE_LEAF:
        node.scores = None  # FIXME
        node.decision_feature = None

        value = sktree.value[skid]
        decision = np.argmax(value, axis=1)
        node._class = decision
    else:
        node.scores = None  # FIXME
        node.decision_feature = sktree.feature[skid]
        node._class = np.array([-1])  # FIXME
        _node_from_scikit_node(sktree, left_child, node.create_child_node(name=str(left_child)))
        _node_from_scikit_node(sktree, right_child, node.create_child_node(name=str(right_child)))
    return node


decision_tree = tree_from_scikit(clf)
decision_tree.to_torch(to_torch=True)
decision_tree.prune()
decision_tree.clean()
decision_tree.to_torch(to_torch=False)
with open(save_dir / "tree.pkl", "wb") as f:
    pickle.dump(decision_tree, f)

print(f'Final parameters saved to "{save_dir}"')
logger.close()
