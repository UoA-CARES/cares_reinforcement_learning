# Contributing to CARES Reinforcement Learning

Thank you for your interest in contributing to CARES Reinforcement Learning.

We welcome contributions that expand the library, particularly implementations of established reinforcement learning algorithms and methodologies.

---

## Types of Contributions

We are happy to consider contributions such as:

- Implementations of established reinforcement learning algorithms
- Implementations of known RL training methodologies
- Bug fixes
- Tests
- Documentation improvements

Contributions should integrate cleanly with the existing framework and follow the current project structure.

---

## Before Starting Work

For non-trivial contributions, please **open an issue or discussion first**.

This helps ensure the proposed work aligns with the goals of the project and avoids duplicated effort.

This is particularly important for:

- Structural or architectural changes
- Large refactors
- Introducing new dependencies
- Changes that affect existing workflows

The repository architecture is intentionally designed around the workflows of the local research team.  
We are open to improvement ideas, but **structural changes should be discussed before implementation**.

---

## Development Setup

Please follow the installation instructions in the repository README.

Typical setup:

```bash
git clone https://github.com/UoA-CARES/cares_reinforcement_learning.git
cd cares_reinforcement_learning
pip install -e ".[gym]"
```

Python 3.10+ is required.

## Algorithm Contributions

When contributing a new algorithm, please:

- Follow the existing code abstractions and structure
- Include a citation to the original paper
- Document any deviations from the reference implementation
- Integrate cleanly with the current training framework

Exact reproduction of published results is not required, but implementations should produce reasonable learning behaviour on standard environments.
Performance may differ due to hyperparameters, environment differences, and implementation details but the algorithm should learn. 

## Testing
Please run the following checks before opening a pull request.

Format Code
```bash
black .
```

Run tests
```bash
pytest tests
```

Run linting
```bash
pylint $(git ls-files '*.py') --rcfile .pylintrc --fail-under=9 --fail-on=error
```

## Pull Request Guidelines

When submitting a pull request:
- Clearly describe the changes and motivation
- Link any relevant issues or discussions
- Keep the PR focused and well-scoped
- Include tests where possible
- Update documentation if needed

## Review Process

All contributions are reviewed on a best-effort basis. While we appreciate all contributions, not all pull requests may be merged.

## License

By contributing to this repository, you agree that your contributions will be licensed under the same license as the project.
