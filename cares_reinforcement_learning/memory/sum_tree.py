import random

import numpy as np


class SumTree:
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

    def sample_value(self, query_value: float | None = None) -> int:
        """Samples an element from the sum tree.

        Each element has probability p_i / sum_j p_j of being picked, where p_i is
        the (positive) value associated with node i (possibly unnormalized).

        Args:
            query_value: float in [0, 1], used as the random value to select a sample.
            If None, will select one randomly in [0, 1).

        Returns:
            int, a random element from the sum tree.
        """

        # Sample a value in range [0, R), where R is the value stored at the root.
        query_value = random.random() if query_value is None else query_value
        query_value *= self.levels[0][0]
        return self._retrieve(np.array([query_value]))[0]

    def sample_simple(self, batch_size: int) -> np.ndarray:
        """
        Samples indices from the sum tree based on a given batch size.

        Batch binary search through sum tree.

        Sample a priority between 0 and the max priority and then search the tree for the corresponding index

        Args:
            batch_size (int): The number of indices to sample.

        Returns:
            numpy.ndarray: An array of sampled indices.
        """
        values = np.random.uniform(0, self.levels[0][0], size=batch_size)
        return self._retrieve(values)

    def sample_stratified(self, batch_size: int) -> np.ndarray:
        """Performs stratified sampling using the sum tree.

        Let R be the value at the root (total value of sum tree). This method will
        divide [0, R) into batch_size segments, pick a random number from each of
        those segments, and use that random number to sample from the sum_tree. This
        is as specified in Schaul et al. (2015).

        PER Paper: https://arxiv.org/pdf/1511.05952.pdf

        Args:
            batch_size: int, the number of strata to use.

        Returns:
            np.ndarray: list of batch_size elements sampled from the sum tree.
        """

        bounds = np.linspace(0.0, 1.0, batch_size + 1)

        segments = [(bounds[i], bounds[i + 1]) for i in range(batch_size)]

        query_values = [
            random.uniform(segment[0], segment[1]) * self.levels[0][0]
            for segment in segments
        ]
        return self._retrieve(np.array(query_values))

    def _retrieve(self, values: np.ndarray) -> np.ndarray:
        """
        Retrieves the indices of the values in the sum tree that correspond to the given array of values.

        Args:
            values (np.ndarray): The array of values for which to retrieve the indices.

        Returns:
            np.ndarray: The indices of the values in the sum tree.

        """
        ind = np.zeros(len(values), dtype=int)
        for nodes in self.levels[1:]:
            ind *= 2
            left_sum = nodes[ind]
            # right_sum = nodes[ind + 1]

            is_greater = np.greater(values, left_sum)

            # If value > left_sum -> go right (+1), else go left (+0)
            ind += is_greater

            # If we go right, we only need to consider the values in the right tree
            # so we subtract the sum of values in the left tree
            values -= left_sum * is_greater

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

    def batch_set(self, ind: np.ndarray, new_priority: np.ndarray) -> None:
        """
        Batch update the priorities of multiple nodes in the sum tree.

        Args:
            ind (np.ndarray): The indices of the nodes to update.
            new_priority (np.ndarray): The new priorities to assign to the nodes.

        Returns:
            None
        """

        # Confirm we don't increment a node twice
        ind, unique_ind = np.unique(ind, return_index=True)
        priority_diff = new_priority[unique_ind] - self.levels[-1][ind]

        for nodes in self.levels[::-1]:
            # Best with numpy >= 1.26.4 for optimised add.at
            np.add.at(nodes, ind, priority_diff)
            ind //= 2
