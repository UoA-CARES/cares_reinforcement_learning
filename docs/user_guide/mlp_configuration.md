--8<-- "include/glossary.md"
# MLP Configuration
Network components are configured through the `MLPConfig` data class which defines the internal architectures of each component. Rather than hardcoding network structures, each component (e.g. actor, critic, or DQN) receives an `MLPConfig` that specifies how its internal modules should be constructed. This instantiates the generic and configurable [MLP class][mlp-code].  

An `MLPConfig` defines a sequence of layers with their default configurations (e.g. `linear` in_features=64 and out_features=64). This allows complex architectures to be expressed as simple configuration objects, which are then interpreted by the [MLP class][mlp-code] to build the corresponding PyTorch modules.

The supported layer abstractions that are used to construct an MLP are:

- **TrainableLayer**:  
  Represents a learnable layer with parameters, such as `Linear` or custom layers like `NoisyLinear`. These correspond to standard `nn.Module` layers in PyTorch that contain weights and are updated during training.

- **NormLayer**:  
  Represents normalisation layers (e.g. `LayerNorm`, `BatchNorm1d`, or `BatchRenorm1d`) that stabilise training by normalising activations. These are non-trainable in structure but may contain internal parameters (e.g. scale and shift).

- **FunctionLayer**:  
  Represents parameter-free operations such as activation functions (`ReLU`, `Tanh`, `Sigmoid`) or other functional transformations (e.g. `Dropout`). These modify the data but do not define learnable weights.

- **ResidualLayer**:  
  Represents a residual block, where an input is passed through a sequence of layers (the “main path”) and then combined with a shortcut connection. This follows the standard residual connection pattern (`output = f(x) + x`), enabling deeper and more stable architectures. Internally, this is implemented as a nested MLP with an optional learnable shortcut.

This design enables fully configurable network components: the overall structure of a component can be modified through configuration files. As a result, changing architectures (e.g. depth, width, activations, or adding residual connections) requires no code changes — only updates to the launch configuration files.

--8<-- "include/links.md"