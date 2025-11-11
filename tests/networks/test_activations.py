import torch
import cares_reinforcement_learning.networks.fractional_activations as afs

def test_golu_activation():
    golu = afs.GoLU()
    input_tensor = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    output_tensor = golu(input_tensor)
    # GoLU: expected_output = input_tensor * torch.exp(-torch.exp(-input_tensor))
    expected_output = torch.tensor([-0.0659880358453, 0.0, 0.692200627555, 1.74684603699])
    assert torch.allclose(output_tensor, expected_output, atol=1e-6), "GoLU activation output is incorrect"
