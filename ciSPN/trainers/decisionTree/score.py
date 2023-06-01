from abc import ABCMeta, abstractmethod


class Score:
    """
    Represents a score. All scores are expected to be argmaxed!
    We expect a two-split of the data
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def score(self, node, x0, y0, x1, y1):
        raise NotImplementedError
