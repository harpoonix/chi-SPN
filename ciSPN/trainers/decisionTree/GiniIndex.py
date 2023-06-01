import torch

from ciSPN.trainers.decisionTree.score import Score


class GiniIndex(Score):
    """
    Higher is better
    """

    def compute_gini_index(self, y):
        classes, counts = torch.unique(y, dim=0, return_counts=True)
        N = y.shape[0]

        gini = 0.0
        for count in counts:
            p_c = count / N
            gini += (p_c * (1 - p_c))
        return gini

    def score(self, node, x0, y0, x1, y1):
        # gini = 0.0
        gini0 = self.compute_gini_index(y0)
        gini1 = self.compute_gini_index(y1)

        c0 = len(y0)
        c1 = len(y1)
        total = (c0 + c1)
        c0 /= total
        c1 /= total

        # negative since score is argmaxed ...
        return - (gini0 * c0 + gini1 * c1)
