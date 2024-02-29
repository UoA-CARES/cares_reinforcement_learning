import numpy as np

class SumTree(object):
    def __init__(self, max_size):
        self.levels = [np.zeros(1)]
        # Tree construction
        # Double the number of nodes at each level
        level_size = 1
        while level_size < max_size:
            level_size *= 2
            self.levels.append(np.zeros(level_size))

    # Batch binary search through sum tree
    # Sample a priority between 0 and the max priority
    # and then search the tree for the corresponding index
    def sample(self, batch_size):
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

    def set(self, ind, new_priority):
        priority_diff = new_priority - self.levels[-1][ind]

        for nodes in self.levels[::-1]:
            np.add.at(nodes, ind, priority_diff)
            ind //= 2

    def batch_set(self, ind, new_priority):
        # Confirm we don't increment a node twice
        ind, unique_ind = np.unique(ind, return_index=True)
        priority_diff = new_priority[unique_ind] - self.levels[-1][ind]

        for nodes in self.levels[::-1]:
            np.add.at(nodes, ind, priority_diff)
            ind //= 2