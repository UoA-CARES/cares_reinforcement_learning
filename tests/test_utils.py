import pytest

import torch
import cares_reinforcement_learning.util.helpers as hlp
import cares_reinforcement_learning.util.activation_functions as afs

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

def test_golu_activation():
    golu = afs.GoLU()
    input_tensor = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    output_tensor = golu(input_tensor)
    # GoLU: expected_output = input_tensor * torch.exp(-torch.exp(-input_tensor))
    expected_output = torch.tensor([-0.0659880358453, 0.0, 0.692200627555, 1.74684603699])
    assert torch.allclose(output_tensor, expected_output, atol=1e-6), "GoLU activation output is incorrect"
