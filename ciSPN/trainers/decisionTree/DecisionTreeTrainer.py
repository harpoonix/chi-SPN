import torch
import numpy as np


class BinaryDecisionTreeTrainer:

    def __init__(self, scorer):
        # A higher score indicates a better split!
        # scorer.score(node, x0, y0, x1, y1) where node is the parent node
        # and (x0,y0) and (x1,y1) are the proposed splits
        self.scorer = scorer

    def fit(self, dt, dataset, decision_split_stop=0.0):
        root = dt.get_root()

        x, y = dataset.get_all_data()
        vars = list(range(x.shape[1]))

        self._fit_internal(root, vars, x, y, decision_split_stop)

    def _get_constant_dims(self, data, decision_split_stop=0.0):
        stat = torch.sum(data, dim=0, keepdim=True)

        # test if (almost) all values of a variable are zero or one
        mask = ~torch.logical_or((stat >= (1.0-decision_split_stop)*len(data)), (stat <= decision_split_stop*len(data)))
        # print("mask", len(data), stat, mask, mask.shape)
        # print(f"constant dimension reduction ({data.shape[1]} -> {torch.sum(mask)})")

        return mask

    def _fit_internal(self, node, vars_names, x, y, decision_split_stop):
        """
        vars_names: list of variable names in x
        """

        self._classify_node(node, y)

        # Remove inputs that have a constant value. Constant vars never lead to a
        # decision (one child will take 100% of samples) - this improves performance.
        var_mask = self._get_constant_dims(x, decision_split_stop)
        # True=variable is considered, False=variable is not considered (already evaluated or constant)
        node.mask = var_mask
        node.active_var_names = [vars_names[i] for i in range(var_mask.shape[1]) if var_mask[0, i]]

        if len(node.active_var_names) == 0:
            # no active vars -> leaf node
            node.scores = None
            node.decision_feature = None
            return

        scores = []
        idxs = []
        for i in range(len(vars_names)):
            # ignore already considered or constant variables
            if not var_mask[0, i]:
                continue

            # split data by i-th feature
            row_mask = (x[:, i] == 0).unsqueeze(1)
            x0 = torch.masked_select(x, row_mask).view((-1, x.shape[1]))
            y0 = torch.masked_select(y, row_mask).view((-1, y.shape[1]))
            x1 = torch.masked_select(x, ~row_mask).view((-1, x.shape[1]))
            y1 = torch.masked_select(y, ~row_mask).view((-1, y.shape[1]))

            # note that since all variables are binary, all decided vars are constant
            score = self.scorer.score(node, x0, y0, x1, y1)
            scores.append(score.cpu().item())
            idxs.append(i)
        best_idx = idxs[np.argmax(scores)]

        node.scores = scores
        node.decision_feature = best_idx

        # recreate the best split and create nodes
        row_mask = (x[:, best_idx] == 0).unsqueeze(1)
        x0 = torch.masked_select(x, row_mask).view((-1, x.shape[1]))
        y0 = torch.masked_select(y, row_mask).view((-1, y.shape[1]))
        x1 = torch.masked_select(x, ~row_mask).view((-1, x.shape[1]))
        y1 = torch.masked_select(y, ~row_mask).view((-1, y.shape[1]))

        n0 = node.create_child_node("0")
        n1 = node.create_child_node("1")
        self._fit_internal(n0, vars_names, x0, y0, decision_split_stop)
        self._fit_internal(n1, vars_names, x1, y1, decision_split_stop)


    def _classify_node(self, node, y):
        # predict most probable configuration
        classes, counts = torch.unique(y, dim=0, return_counts=True)
        #print("Node ", node.name)
        #print("Cls|#:", classes, counts)

        mp_class = classes[torch.argmax(counts)]
        #print("MPC:", mp_class)
        node._class = mp_class
