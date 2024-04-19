import math

import numpy as np


class SumTree(object):
    """
    A sum tree data structure for storing replay priorities.

    A sum tree is a complete binary tree whose leaves contain values called
    priorities. Internal nodes maintain the sum of the priorities of all leaf
    nodes in their subtree.

    For capacity = 4, the tree may look like this:

                +---+
                |2.5|
                +-+-+
                    |
            +-------+--------+
            |                |
        +-+-+            +-+-+
        |1.5|            |1.0|
        +-+-+            +-+-+
            |                |
        +----+----+      +----+----+
        |         |      |         |
    +-+-+     +-+-+  +-+-+     +-+-+
    |0.5|     |1.0|  |0.5|     |0.5|
    +---+     +---+  +---+     +---+

    This is stored in a list of numpy arrays:
    self.nodes = [ [2.5], [1.5, 1], [0.5, 1, 0.5, 0.5] ]

    For conciseness, we allocate arrays as powers of two, and pad the excess
    elements with zero values.

    This is similar to the usual array-based representation of a complete binary
    tree, but is a little more user-friendly.
    """

    def __init__(self, max_size: int):
        self.levels = [np.zeros(1)]
        # Tree construction
        # Double the number of nodes at each level
        level_size = 1
        while level_size < max_size:
            level_size *= 2
            self.levels.append(np.zeros(level_size))

    def sample(self, batch_size: int) -> list[int]:
        """
        Samples indices from the sum tree based on a given batch size.

        Batch binary search through sum tree.

        Sample a priority between 0 and the max priority and then search the tree for the corresponding index

        Args:
            batch_size (int): The number of indices to sample.

        Returns:
            numpy.ndarray: An array of sampled indices.
        """
        value = np.random.uniform(0, self.levels[0][0], size=batch_size)
        ind = np.zeros(batch_size, dtype=int)

        for nodes in self.levels[1:]:
            ind *= 2
            left_sum = nodes[ind]

            is_greater = np.greater(value, left_sum)
            # If value > left_sum -> go right (+1), else go left (+0)
            ind += is_greater
            # If we go right, we only need to consider the values in the right tree
            # so we subtract the sum of values in the left tree
            value -= left_sum * is_greater

        return ind

    def set(self, ind: int, new_priority: float) -> None:
        """
        Set the priority of a node at a given index.

        Args:
            ind (int): The index of the node.
            new_priority (float): The new priority value.

        Returns:
            None
        """
        priority_diff = new_priority - self.levels[-1][ind]

        for nodes in self.levels[::-1]:
            np.add.at(nodes, ind, priority_diff)
            ind //= 2

    def batch_set(self, ind: list[int], new_priority: list[float]) -> None:
        """
        Batch update the priorities of multiple nodes in the sum tree.

        Args:
            ind (list[int]): The indices of the nodes to update.
            new_priority (list[float]): The new priorities to assign to the nodes.

        Returns:
            None
        """

        # Confirm we don't increment a node twice
        ind, unique_ind = np.unique(ind, return_index=True)
        priority_diff = new_priority[unique_ind] - self.levels[-1][ind]

        for nodes in self.levels[::-1]:
            np.add.at(nodes, ind, priority_diff)
            ind //= 2
