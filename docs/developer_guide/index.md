# Developer Guide
Thank you for your interest in contributing to CARES Reinforcement Learning! This guide explains how to add a new algorithm or environment wrapper to the library.

Before contributing, please review this guide to understand the structure and requirements for adding new features to the CARES Reinforcement Learning library. Whether you are implementing a new algorithm or environment wrapper, following these steps will help ensure consistency and maintainability across the project.

## Types of Contributions
Contributions should integrate cleanly with the existing framework and follow the current project structure. To assist with any implementations you can read the code [abstraction guide](./abstractions.md).

We are happy to consider contributions such as:

- [Implementations of reinforcement learning algorithms][algorithm-guide]
- [New Gym Environment Support][environment-guide]
- New General Methods or Mechanisms useful for Reinforcement Learning

## Package Overview

The `cares_reinforcement_learning` package is organised around the core abstractions of the framework: `runners`, `algorithm`, `networks`, `envs`, `memory`, `types`, and supporting utilities. Each folder is responsible for a specific part of the reinforcement learning pipeline, making it easier to identify where new features, algorithms, or environment integrations should be added.

## Core Framework Structure

### `runners/`
The [runners][runners-code] folder contains the primary training and evaluation loop logic, including experiment execution, evaluation scheduling, and checkpoint management. This is the main entry point for how algorithms interact with environments during training.

### `algorithm/`
The [algorithm][algorithm-code] folder contains the learning logic for each method, grouped by broad method families such as `value-based`, `policy-based`, and `unsupervised skill discovery`.

This folder also contains:
- Shared `SARL` and `MARL` algorithm interfaces
- Default algorithm configuration classes
- The `AlgorithmFactory` used to construct fully configured agents

This is where new learning methods are implemented.

### `networks/`
The [networks][networks-code] folder contains the neural network components used by the algorithms. Each algorithm typically has its own subfolder containing components such as actors, critics, encoders, mixers, or value networks.

The `algorithm/` and `networks/` folders work closely together:

- `algorithm/` defines how learning happens
- `networks/` defines the trainable structures used for that learning

These are the two primary folders used when adding a new algorithm to the library.

### `envs/`
The [envs][envs-code] folder contains the environment wrappers that adapt external environment APIs (such as Gymnasium or PettingZoo) to the framework’s `SARL` and `MARL` environment interfaces.

These wrappers standardise observation, action, and transition handling so that all algorithms can interact with environments through a consistent abstraction defined in `types`. This is where new environment integrations should be added.

### `memory/`
The [memory][memory-code] folder contains replay buffers, rollout storage, and memory abstractions used by the training loops and algorithms.

The memory supports both single-agent and multi-agent learning workflows.

### `types/`
The [types][types-code] folder contains the shared data abstractions used across the framework, including the `Observation`, `Experience`, and related training data structures. These types ensure consistency between environments, algorithms, memory buffers, and runners across both SARL and MARL implementations.

## General Development Guide

As a general rule:

- Add new algorithms in [`algorithm/`][algorithm-code]
- Add new network components in [`networks/`][networks-code]
- Add new environment integrations in [`envs/`][envs-code]
- Modify replay and rollout handling in [`memory/`][memory-code]
- Update shared data structures in [`types/`][types-code]
- Adjust training loop behaviour in [`runners/`][runners-code]

This structure helps keep implementation details isolated while maintaining consistent abstractions across the full framework.

--8<-- "include/links.md"