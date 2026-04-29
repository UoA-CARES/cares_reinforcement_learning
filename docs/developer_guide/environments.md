--8<-- "include/glossary.md"

# Envrionment Wrapper Guide { #environment-guide }

To integrate a new environment into the CARES Reinforcement Learning framework, create an environment wrapper that adapts the source environment to the framework’s standard interface. The wrapper should follow either the SARL or MARL environment interface, both of which build on the shared `BaseEnvironment` abstraction. This ensures that all environments expose a consistent set of methods and metadata to the training loops, regardless of whether they originate from Gymnasium, PettingZoo, or another simulator.

The purpose of the wrapper is not to mirror the external API exactly, but to translate it into the observation, action, and experience types expected by the CARES RL algorithms - the base interface for all wrappers is through the [BaseEnvironment][base-env-code]. In practice, this means converting the raw outputs of the environment into the framework’s `Observation` and `Experience` [types][types-code], while also exposing properties such as `observation_space`, `action_num`, action bounds, and action sampling through a common interface. The base environment defines the required methods, including `reset`, `step`, `sample_action`, `set_seed`, `grab_frame`, and rendering-related helpers, which all wrappers must implement or override as needed.

![Architecture Overview](../images/envrionment_wrapper.png)

## Wrapper Responsibilities

The environment wrappers are meant to be a lightweight conversion from the third-party environment data types and interfaces to the CARES Reinforcement Learning interface and data types. The additional types introduced in this package are to enable additional clarity about data handling and passing through the Algorithms in our library - improving type hinting and readability. Full details on the overall abstractions can be found in the [abstractions](./abstractions.md) documentation. 

A wrapper is responsible for three main tasks:

1. Adapting the source environment API to the framework [interface][base-env-code].
2. Converting environment outputs into the framework’s standard observation and experience [types][types-code].
3. Providing the metadata required by the algorithms and training loops in a consistent format.

For single-agent environments, this means wrapping the task as a [SARL][sarl-env-code] environment with a single observation and action interface. For multi-agent environments, the wrapper must instead follow the [MARL][marl-env-code] structure, preserving agent ordering and returning observations, actions, and transition data in a way that is consistent across all agents. This is especially important because the framework uses the environment wrappers to enforce the expected typing and structure used by the algorithms.

## Implementation Steps
To implement a new envrionment wrapper in the CARES Reinforcement Learning package you need to follow four steps:

1. Implement the `<Environment>` class in the [envrionment folder][envs-code].
2. Define the run time parameters for the environment in [configurations file][envs-config].
3. Register the constructor in the [EnvironmentFactory][envs-fac].

This modular design enables reproducibility, flexible experimentation, and minimal code changes for new envrionments.

!!! tip "Design Guidance"

    Keep the envrionment wrapper as clean and stable as possible. The algorithms should not need to know whether the underlying environment comes from Gymnasium, PettingZoo, or a custom envrionment. This separation keeps the training code independent of the envrionment backend and allows new environments to be added without modifying the algorithms or training loops. 

## 1. Create Environment Interface

The SARL and MARL interfaces ultimately inherit from [BaseEnvironment][base-env-code], which defines the shared contract across environment types. At minimum, every wrapper is expected to implement a public:

- `reset` to initialise the environment and return the initial observation.
- `step` to apply an action and return an `Experience` object containing the transition information.
- `sample_action` to sample a valid action from the environment.
- `set_seed` to ensure reproducibility.
- `observation_space`, `action_num`, `min_action_value`, and `max_action_value` to expose the action and observation definitions expected by the framework.
- `grab_frame` and optionally `get_overlay_info` for rendering and visualisation support. 

The shared `BaseEnvironment` provides the rendering interface through `render()` and `grab_frame()`, where `render()` calls `grab_frame()` and displays the result using OpenCV. This allows wrappers to standardise how image observations and visual debugging are handled without exposing simulator-specific rendering logic to the algorithms. 

### SARL Environment
Single agent environments (e.g. [OpenAI Gymnasium][gymnasium], and [Deep Mind Control Suite][dm-control]) are designed for single agent algorithms (e.g. [DQN][dqn-code], [SAC][sac-code]).

```python
from cares_reinforcement_learning.envs.sarl.sarl_environment import SARLEnvironment

class ExampleSARLEnvironment(SARLEnvironment):
    def __init__(self, config, seed: int):
        super().__init__(config=config, seed=seed)

        # Create the environment as per normal here
        self.env = ...

    # returns vector_state, reward, done, truncated, info 
    def _step(self, action: np.ndarray) -> tuple:
        ...

    # returns vector_state
    def _reset(self, training: bool = True) -> np.ndarray:
        ...

    # returns a valid action - discrete (int) or continuous (np.ndarray)
    def sample_action(self) -> int  | np.ndarray:
        ...

    # the size of the vector_state
    def _vector_space(self) -> int:
        ...

    # Defines how image states are generated by the environment
    def grab_frame(self, height: int = 240, width: int = 300) -> np.ndarray:
        ...
```

!!! tip "SARLEnvironment Base Support"

    The base SARLEnrionment handles wrapping data into the SARLExperience and SARLObservation for you. The base SARL wrappers just need to return data in a vector state format. 

### MARL Environment
The multi-agent environments (e.g. [MPE2][mpe2]) are designed for multi-agent algorithms (e.g. [MADDPG][maddpg-code]).

```python
from cares_reinforcement_learning.envs.sarl.marl_environment import MARLEnvironment

class ExampleMARLEnvironment(MARLEnvironment):
    def __init__(self, config, seed: int):
        super().__init__(config=config, seed=seed)

        # Create the environment as per normal here
        self.env = ...

    def step(self, action: list[int] | list[np.ndarray]) -> MultiAgentExperience:
        ...

    def reset(self, training: bool = True) -> MARLObservation:
        ...

    def sample_action(self) -> list[int] | list[np.ndarray]:
        ...

    def observation_space(self) -> dict[str, Any]:
        ...
```

## 2. Create Environment Configurations

To support flexible and reproducible environment setup, each environment wrapper in CARES RL uses a configuration class. These configuration classes define all parameters needed to instantiate and control the environment, such as observation shape, action space, noise, and environment-specific options.

All environment configuration classes should inherit from [`GymEnvironmentConfig`][config-file], which provides common fields (e.g., `frames_to_stack`, `frame_width`, `frame_height`, `grey_scale`, `state_std`, `action_std`, etc.).

### How to Add a New Environment Configuration

1. **Create a new config class** in [`envs/configurations.py`][envs-config] by subclassing `GymEnvironmentConfig`.
2. **Set the `gym` class variable** to a unique string identifier for your environment - this name is used by the command line tool parameter reader [rl_parser.py][rl-parser]. 
3. **Add any environment-specific attributes** as class variables with default values.
4. **(Optional) Override defaults** for inherited attributes if your environment requires different defaults.

**Example:**

```python
class MyCustomEnvConfig(GymEnvironmentConfig):
    gym: ClassVar[str] = "my_custom_env"

    my_param: int = 42
    another_setting: str = "default"
```

You can now use `MyCustomEnvConfig` as the `config` argument when creating your environment wrapper. All configuration values will be available as attributes on the config object.

!!! Tip "OpenAI Example"

    See the existing config classes (e.g., `OpenAIConfig`, `DMCSConfig`, `PyBoyConfig`, etc.) in [`envs/configurations.py`][envs-config] for more examples.

## 3. Define the Constructor
After defining your environment class and its configuration, you must register it in the [EnvironmentFactory][envs-fac] so it can be instantiated by the framework. This step is required for your environment to be discoverable and usable in experiments.

1. In the `create_environment` method, add a new `case` for your configuration class. This should match the pattern of the existing cases.
2. Import your environment class inside the case (dynamic import pattern to reduce dependency requirements).
3. Instantiate your environment for both `env` and `eval_env` as needed, passing the correct arguments (see other cases for SARL or MARL patterns).

### Example MyCustomEnvConfig

Suppose you created `MyCustomEnvConfig` and `MyCustomEnvironment`:

```python
case cfg.MyCustomEnvConfig():
    from cares_reinforcement_learning.envs.sarl.my_custom.my_custom_environment import MyCustomEnvironment
    env = MyCustomEnvironment(config, train_seed, image_observation=image_observation)
    eval_env = MyCustomEnvironment(config, eval_seed, image_observation=image_observation)
```

!!! warning "Naming Convention Matters"
    The naming convention between:

    - `<Environment>`
    - `<Environment>Config`
    - `gym: ClassVar[str] = "<Environment>"`

    must remain consistent.

    These names are used by the automated configuration loader. 

!!! warning "Dynamic Imports"
    
    Always use dynamic imports (inside the case) to reduce dependency creep for users who aren't using this gym.

--8<-- "include/links.md"