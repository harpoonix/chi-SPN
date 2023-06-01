import torch

from ciSPN.trainers.decisionTree.score import Score


class CausalLossScore(Score):
    """
    Higher is better
    """

    def __init__(self, spn, batch_size):
        self._spn = spn
        self._batch_size = batch_size

    def eval_split_likelihood(self, x, y):
        # predict the most probable configuration
        classes, counts = torch.unique(y, dim=0, return_counts=True)
        mp_class = classes[torch.argmax(counts)]
        pred0 = torch.tile(mp_class, (self._batch_size, 1))

        i = 0
        evals = torch.empty(len(x), 1).cuda()
        while i < len(x):
            x_batch = x[i:i + self._batch_size, :]
            num_samples = x_batch.shape[0]

            # check if the last batch has less samples
            if num_samples != pred0.shape[0]:
                pred0 = torch.tile(mp_class, (num_samples, 1))

            # compute the data probability give the most probable class. The output of the spn is in log domain.
            p_batch = self._spn.forward(x_batch, pred0)
            evals[i:i + num_samples] = torch.exp(p_batch)

            i += num_samples

        return torch.mean(evals)

    def score(self, node, x0, y0, x1, y1):
        p0 = self.eval_split_likelihood(x0, y0)
        p1 = self.eval_split_likelihood(x1, y1)

        c0 = len(x0)
        c1 = len(x1)
        total = (c0 + c1)
        c0 /= total
        c1 /= total
        return p0 * c0 + p1 * c1
