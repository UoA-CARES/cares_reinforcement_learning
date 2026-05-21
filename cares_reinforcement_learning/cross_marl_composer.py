import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

from cares_reinforcement_learning.algorithm import (
    configurations as algo_configs,
)

logger = logging.getLogger(__name__)


def load_json(path: Path) -> dict[str, Any]:
    logger.debug(f"Loading JSON from {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Saving JSON to {path}")
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def parse_team_log(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Team logs must be TEAM_NAME=BASE_LOG_DIR")

    team_name, base_log_dir = value.split("=", maxsplit=1)
    return team_name, Path(base_log_dir).expanduser()


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing source folder: {src}")

    if dst.exists():
        logger.debug(f"Removing existing destination: {dst}")
        shutil.rmtree(dst)

    logger.info(f"Copying model folder: {src} → {dst}")
    shutil.copytree(src, dst)


def load_team_alg_config(
    base_log_dir: Path, seed: str, output_dir: Path, team_name: str
) -> dict[str, Any]:
    logger.info(f"Loading team config for '{team_name}' from {base_log_dir}")

    alg_config_path = base_log_dir / "alg_config.json"
    src_model_path = base_log_dir / seed / "models" / "final"

    if not alg_config_path.exists():
        raise FileNotFoundError(f"Missing alg_config.json: {alg_config_path}")

    if not src_model_path.exists():
        raise FileNotFoundError(f"Missing model folder: {src_model_path}")

    dst_model_path = output_dir / seed / "models" / "final" / team_name
    copy_tree(src_model_path, dst_model_path)

    alg_config = load_json(alg_config_path)
    alg_config["model_path"] = str(dst_model_path)

    logger.debug(f"Loaded algorithm config for '{team_name}'")

    return alg_config


def validate_shared_env_config(team_log_dirs: dict[str, Path]) -> dict[str, Any]:
    logger.info(f"Validating environment config across {len(team_log_dirs)} teams...")

    env_configs = {
        team_name: load_json(base_log_dir / "env_config.json")
        for team_name, base_log_dir in team_log_dirs.items()
    }

    first_team = next(iter(env_configs))
    reference = env_configs[first_team]

    for team_name, env_config in env_configs.items():
        if env_config != reference:
            raise ValueError(
                f"env_config.json mismatch between '{first_team}' and '{team_name}'"
            )

    logger.info("✓ Environment configs match across all teams")
    return reference


def default_train_config() -> dict[str, Any]:
    from cares_reinforcement_learning.algorithm.configurations import TrainingConfig

    return TrainingConfig().model_dump()


def create_cross_marl_folder(
    team_log_dirs: dict[str, Path],
    seed: str,
    output_dir: Path,
    learning_team_name: str | None = None,
    learning_algorithm_name: str | None = None,
) -> None:
    logger.info("=" * 60)
    logger.info("Creating CrossMARL folder structure...")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Frozen teams: {list(team_log_dirs.keys())}")
    logger.info(f"Learning team: {learning_team_name}")
    logger.info(f"Learning algorithm: {learning_algorithm_name}")
    logger.info("=" * 60)

    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    agents_config: dict[str, Any] = {}

    for team_name, base_log_dir in team_log_dirs.items():
        if team_name == learning_team_name:
            continue
        agents_config[team_name] = load_team_alg_config(
            base_log_dir=base_log_dir,
            seed=seed,
            output_dir=output_dir,
            team_name=team_name,
        )

    if learning_team_name is not None or learning_algorithm_name is not None:
        if learning_algorithm_name is None:
            raise ValueError(
                "--learning-algorithm_name is required when --learning-team-name is set"
            )
        if learning_team_name is None:
            raise ValueError(
                "--learning-team_name is required when --learning-algorithm_name is set"
            )

        # Dynamically instantiate the config class by algorithm name
        config_class_name = f"{learning_algorithm_name.upper()}Config"
        config_class = getattr(algo_configs, config_class_name, None)

        if config_class is None:
            raise ValueError(f"Unknown algorithm name: {learning_algorithm_name}")
        logger.info(
            f"Instantiating default config for '{learning_team_name}' using algorithm '{learning_algorithm_name}'"
        )
        agents_config[learning_team_name] = config_class().model_dump()

    cross_marl_config = {
        "algorithm": "CrossMARL",
        "marl_observation": 1,
        "learning_team_name": learning_team_name,
        "agents_config": agents_config,
    }

    env_config = validate_shared_env_config(team_log_dirs)

    logger.info(f"Writing config files to {output_dir}")
    save_json(cross_marl_config, output_dir / "alg_config.json")
    save_json(env_config, output_dir / "env_config.json")
    save_json(default_train_config(), output_dir / "train_config.json")

    logger.info("=" * 60)
    logger.info("✓ CrossMARL folder created successfully")
    logger.info("=" * 60)


def main() -> None:
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Create a self-contained CrossMARL log folder."
    )

    parser.add_argument(
        "--team-log",
        action="append",
        type=parse_team_log,
        required=True,
        help="Frozen team log folder as TEAM_NAME=BASE_LOG_DIR",
    )

    parser.add_argument(
        "--seed",
        required=True,
        help="Seed folder to copy, e.g. 10, 20, 30, 40, 50",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output CrossMARL log folder",
    )

    parser.add_argument(
        "--learning-team-name",
        default=None,
        help="Optional new learning team name",
    )
    parser.add_argument(
        "--learning-algorithm",
        default=None,
        help="Algorithm name for the new learning agent (e.g. MADDPG, MATD3, MASAC)",
    )

    args = parser.parse_args()

    try:
        create_cross_marl_folder(
            team_log_dirs=dict(args.team_log),
            seed=args.seed,
            output_dir=args.output_dir,
            learning_team_name=args.learning_team_name,
            learning_algorithm_name=args.learning_algorithm,
        )
    except Exception as e:
        logger.error(f"Failed to create CrossMARL folder: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
