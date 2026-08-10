--8<-- "include/glossary.md"
# MLP Configuration

All network architectures in CARES Reinforcement Learning are defined through `MLPConfig` objects, which specify the structure of each component (e.g. actor, critic, or value network). While each algorithm provides a sensible default configuration, these architectures are fully configurable through the `alg_config.json` file without modifying any code. This design enables flexible experimentation, reproducibility, and consistency across implementations by separating network structure from implementation logic.

## MLPConfig Structure
An algorithm's network components are configured through the `MLPConfig` data class which defines the internal architectures of each component. Rather than hardcoding network structures, each component (e.g. actor, critic, or DQN) receives an `MLPConfig` that specifies how its internal modules should be constructed. This instantiates the generic and configurable [MLP class][mlp-code].  

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

**Example** SAC in its `SACConfig` defines the actor as:

```python
actor_config: MLPConfig = MLPConfig(
    layers=[
        TrainableLayer(layer_type="Linear", out_features=256),
        FunctionLayer(layer_type="ReLU"),
        TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
        FunctionLayer(layer_type="ReLU"),
    ]
)
```

!!! warning "Do not change network architectures in code"
    Default network structures are in place based on papers or modern expectations of algorithms.

    Only change network structures through `alg_config.json` files.

This actor configuration creates an `MLP` with the equivalent pytorch Sequential module:
```python
actor_network = nn.Sequential(
  nn.Linear(observation_size, 256),
  nn.ReLU(),
  nn.Linear(256, 256),
  nn.ReLU(),
  nn.Linear(256, num_actions),)
```


## Configuring MLPConfigs via alg_config.json
Each algorithm defines default network architectures using `MLPConfig` within its configuration class. These defaults provide a reference implementation, but can be overridden directly through the `alg_config.json` file. By modifying the layer definitions in this file, you can customise the structure of each component (e.g. actor, critic) without changing any code.

**Example** Modify the SAC actor network
```json
{
  "algorithm": "SAC",

  "actor_config": {
    "layers": [
      { "layer_category": "trainable", "layer_type": "Linear", "out_features": 256, "params": {} },
      { "layer_category": "function", "layer_type": "ReLU", "params": {} },
      { "layer_category": "norm", "layer_type": "LayerNorm", "in_features": 256, "params": {} },
      { "layer_category": "function", "layer_type": "Dropout", "params": { "p": 0.1 } },
      { "layer_category": "trainable", "layer_type": "Linear", "in_features": 256, "out_features": 256, "params": {} },
      { "layer_category": "function", "layer_type": "ReLU", "params": {} }
    ]
  }
}
```

!!! note "How This Is Applied In Experiments"
    These `MLPConfig` definitions are loaded from `alg_config.json` when you launch experiments with the `cares-rl train config` workflow.

    For the full experiment launch flow and config loading details, see the [Experiment Usage guide](./experiment.md).

The modification to the  actor configuration creates an `MLP` with the equivalent pytorch Sequential module:
```python
actor_network = nn.Sequential(
  nn.Linear(observation_size, 256),
  nn.ReLU(),
  nn.LayerNorm(256),
  nn.Dropout(p=0.1),
  nn.Linear(256, 256),
  nn.ReLU(),
  nn.Linear(256, num_actions),)
```

!!! tip "layer_category is required"
    Each layer entry must include `"layer_category"` to identify the layer type:
    
    - `"trainable"` — learnable layers (e.g. `Linear`, `NoisyLinear`)
    - `"norm"` — normalisation layers (e.g. `LayerNorm`, `BatchNorm1d`, `BatchRenorm1d`)
    - `"function"` — activation functions and parameter-free transforms (e.g. `ReLU`, `Tanh`, `Dropout`)
    - `"residual"` — residual blocks (with nested `main_layers` and optional `shortcut_layer`)

!!! tip "in_features / out_features"
    The first `Linear` layer's `in_features` is automatically inferred from the observation/action space — you only need to specify `out_features` for the first layer. For all subsequent `Linear` layers, both `in_features` and `out_features` should be specified to match your intended architecture.

--8<-- "include/links.md"