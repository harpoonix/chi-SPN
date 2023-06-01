from ciSPN.trainers.decisionTree.score import Score


class CombinedScore(Score):

    def __init__(self, alpha, score_a: Score, score_b:Score):
        self.alpha = alpha
        self.scoreA = score_a
        self.scoreB = score_b

    def score(self, node, x0, y0, x1, y1):
        return self.scoreA.score(node, x0, y0, x1, y1) + (self.alpha * self.scoreA.score(node, x0, y0, x1, y1))
