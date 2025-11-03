import pytest

import cares_reinforcement_learning.util.helpers as hlp


def test_denormalize():
    action = 0.5
    max_action_value = 5
    min_action_value = -5
    result = hlp.denormalize(action, max_action_value, min_action_value)
    assert result == 2.5, "Result does not match expected denormalized value"


def test_normalize():
    action = 2.5
    max_action_value = 5
    min_action_value = -5
    result = hlp.normalize(action, max_action_value, min_action_value)
    assert result == 0.5, "Result does not match expected normalized value"
