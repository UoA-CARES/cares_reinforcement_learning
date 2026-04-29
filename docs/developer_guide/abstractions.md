--8<-- "include/glossary.md"

# Core Abstractions

This page explains the core abstractions used throughout CARES Reinforcement Learning. It is intended for contributors who want to understand how the package is structured, where new components should fit, and which interfaces need to be preserved when adding or modifying functionality.

The main training loop is built around a small number of shared abstractions. These contracts define how components interact and should remain stable as new algorithms, environments, and replay buffers are added. Contributors should aim to fit new implementations into these existing interfaces before modifying the abstractions themselves - although we are open to changes as things evolve!

![Abstraction Overview](../images/abstraction-loop.png)

!!! tip "Evolving Abstraction"
    Contributors should treat these abstractions as stable contracts that allow the package to scale without repeatedly rewriting training infrastructure.

    If you look at the commit history this abstraction has evolved over time. We are open to adjusting things when required, but it must be from deep thought and not quick short-cuts. 

    If a new feature (e.g. Algorithm) requires `if algorithm == ...` checks inside the runner or other components, the abstraction probably needs another evolution.

## Why These Abstractions Exist

The system is intentionally structured around a small number of shared abstractions so that algorithms, environments, and replay buffers can be developed independently while remaining compatible with the same training loop. 

Most components should not need to know the concrete implementation of the other components they interact with. For example, a training loop should not need custom logic for every algorithm, and an algorithm should not need to know whether experience came from a Gymnasium environment, a PettingZoo environment, or a custom simulator.

Instead, contributors should aim to preserve a small number of stable contracts:

- environments return standard observation and experience objects
- algorithms consume those objects through a common interface
- memory buffers store and sample experience through a common interface
- runners coordinate these components without depending on implementation details

!!! tip "Contributor mindset"
    When adding a new algorithm, environment, or replay buffer, try to make it fit the
    existing contracts before changing the runner or training loop.

    If a new feature requires changing one of these abstractions, consider whether the
    change is broadly useful across the package or specific to one implementation.

## Single-Agent and Multi-Agent Abstractions

The interfaces in the codebase support both single-agent reinforcement learning (SARL) and multi-agent reinforcement learning (MARL). Rather than treating MARL as a completely separate system, the package uses the same core abstractions with agent-specific extensions where needed. This allows contributors to implement both paradigms while preserving a consistent training loop.

![SARL vs MARL](../images/sarl-vs-marl.png)

!!! note "SARL vs MARL"
    Single-Agent Reinforcement Learning (SARL) involves one learning agent interacting
    with the environment.

    Multi-Agent Reinforcement Learning (MARL) extends this to multiple agents acting in
    the same environment, often requiring shared state, coordination, or centralized
    training with decentralized execution.

### Environment

The environment is responsible for interacting with the external simulator or task. This may be a Gymnasium environment, a PettingZoo environment, Gazebo simulation, or a custom environment. This interface is managed by the envrionments wrappers - full details on how to implement these are given in the [environments guide](./environments.md).

Its role is to:

- reset the task
- apply actions
- return transition data via experiences
- works with observation and action spaces

The environment should expose a consistent interface regardless of the backend being used.

The goal is that algorithms should not need environment-specific logic or knowledge.

### Experience

An [Experience][exp-code] represents the transition generated from an environment step. It contains the data required for both action selection and training, such as:

- current [observation][obs-code]
- action taken
- reward
- next [observation][obs-code]
- done flag
- truncated flag
- additional environment information
- additional algorithmic information related to the experience

This is the primary object passed between the environment, episode context, and memory buffer.

### Observations

[Observations][obs-code] represent the state information exposed to the algorithm. They are stored inside an [Experience][exp-code] as the current observation and next observation. Observations should describe what the agent can use to make decisions. Observations are designed to support multi-modal inputs rather than assuming a single state representation.

This allows an observation to contain:

- vector-based state information
- image-based state information
- both vector and image inputs together when required
- future multi-modal data types

This keeps the environment flexible while allowing algorithms to define their own network structure for processing different input types.

!!! tip "Observation design"
    Observations should describe what the agent can know, not what the algorithm needs
    for training.

    Keep environment state separate from training logic. If information only exists to
    support optimisation, it usually belongs in the algorithm or replay system rather
    than the observation itself.

!!! tip "Multi-modal observations"
    Observations define the available data, not how that data must be processed.

    An algorithm may use an MLP for vector inputs, a CNN for image inputs, or separate
    network paths for both before combining them.

    The observation abstraction keeps the input format consistent while allowing model
    architecture decisions to remain algorithm-specific.

### Episode Context

[EpisodeContext][ecx-code] is the abstraction used for algorithm training and contains information about the current episode and training loop states. 

This includes:

- Current training Episode
- The number of steps in the current Episode
- Current episode reward total
- Whether the Episode is Done/Truncated

This abstraction adds training loop level metrics to the algorithms for logic on how training is conducted throughout training.

### Algorithm

The algorithm implements the learning logic. All algorithms should follow the same public interface so that runners do not require algorithm-specific logic. This interface is managed by the algorithm interface - full details on how to implement these are given in the [algorithm guide](./algorithm.md).

This typically includes:

- `act(...)`
- `train(...)`
- `save_models(...)`
- `load_models(...)`

Algorithms use [Observations][obs-code] in order to produce actions as [ActionSamples][act-code]. 

Algorithms train using batches of [Experiences][exp-code] sampled from memory along with the current [EpisodeContext][ecx-code]. Each algorithm handles all training logic internally. 

!!! tip "Contributor mindset"
    When adding a new algorithm, try to extend the shared abstractions rather than
    creating special-case training loops.

    Most new functionality should be handled through richer experience and context
    objects, not runner-specific logic.

### Action Sample

Algorithms do not return raw actions directly. Instead, they return [ActionSamples][act-code], which contains:

- the selected action/s
- the action source (e.g. policy or random exploration)
- optional action metadata stored in `extras` (e.g. action values)

This helps the training loop generic across different algorithms.

### Memory Buffer

The Memory Buffer stores the experience generated during environment interaction.

Its role is to:

- store new experiences throughout training
- manage replay structure
- sample training batches for learning

This abstraction allows the same algorithm to work with:

- uniform replay
- prioritized replay
- inverse prioritized replay
- sequence replay
- ...

The algorithm should depend on sampled batches, not on how the buffer stores them.

!!! tip "Replay flexibility"
    The memory abstraction defines how experiences are stored and sampled, but it does not force a single replay strategy.

    The same algorithm can work with uniform replay, prioritized replay, sequence replay, or other sampling strategies without changing the training loop.

!!! warning Keep Data Structures Explicit

    Objects such as `Experience`, `EpisodeContext`, and `ActionSample` should make data flow clear and predictable.

    Avoid hidden assumptions or implicit states where possible.

    Explicit transition objects make debugging easier, simplify replay buffer design, and reduce coupling between components.

## Configuration-Driven Construction

Most components in the package are created through configuration classes rather than being manually constructed inside training scripts. Contributors should avoid hardcoding implementation details directly into runners or training scripts.

This keeps experiments reproducible, reduces duplicated setup logic, and allows runners to remain generic across many algorithms and environments.

The general pattern is:

1. A configuration defines parameters and architecture choices
2. A factory uses that configuration to construct the correct implementation
3. The runner interacts only with the shared abstraction, not the concrete class

This applies to:

- algorithms
- environments
- memory buffers

![Configuration Development](../images/configuration-dev.png)

!!! tip "Contributor Mindset"

    If adding a new feature requires editing multiple runners, the design likely needs to be reconsidered.
    
    In most cases, the correct solution is to extend configuration and factory logic rather than introducing algorithm-specific handling inside the training loop.

## Keep Runners Generic

Runners should coordinate training, not implement algorithm logic.

They should work with abstractions such as:

- `Algorithm`
- `Environment`
- `MemoryBuffer`
- `EpisodeContext`

rather than concrete implementations such as DQN, SAC, or QMIX.

This separation allows contributors to add new methods without rewriting the training loop.

!!! warning "Design Guide"
    If runners have logic being added that is only utilised by one algorithm - rather than by types of algorithms - then something in the design is wrong. 

## When to Change an Abstraction

Sometimes new research requires extending an existing abstraction.

Before changing a shared interface, ask:

- is this broadly useful across multiple algorithms?
- does this improve the package architecture overall?
- can this be solved inside the implementation instead?

Shared abstractions should change slowly.

Changing them should be a design decision, not a shortcut.

Contributors should optimise for consistency first and convenience second.

--8<-- "include/links.md"