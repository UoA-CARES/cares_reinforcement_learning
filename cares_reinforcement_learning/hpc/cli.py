"""
CARES RL HPC CLI.

Submit CARES RL training jobs to the HPC scheduler using the hpc-client.

Example:
    cares-rl-hpc --image cares-rl:v3.1.0 --runtime 12 \
        train cli --gym openai --task HalfCheetah-v4 TD3
"""

import argparse
import json
import os
import shlex
import sys
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit CARES RL training jobs to the HPC cluster.",
    )

    parser.add_argument(
        "--image",
        type=str,
        default=os.environ.get("CARES_RL_HPC_IMAGE"),
        help="Docker image to execute (or set CARES_RL_HPC_IMAGE).",
    )

    parser.add_argument(
        "--runtime",
        type=float,
        default=1.0,
        help="Maximum job runtime in hours (default: 1).",
    )

    parser.add_argument(
        "--job-name",
        type=str,
        default="cares-rl-training",
        help="Name of the HPC job.",
    )

    parser.add_argument(
        "--resumable",
        action="store_true",
        help="Mark the HPC job as resumable.",
    )

    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Required NAS dataset. Can be specified multiple times.",
    )

    parser.add_argument(
        "--worker",
        action="append",
        default=[],
        help="Required worker ID. Can be specified multiple times.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the job specification without submitting it.",
    )

    parser.add_argument(
        "command",
        choices=["train"],
        help="CARES RL command to execute.",
    )

    parser.add_argument(
        "rl_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the CARES RL command.",
    )

    return parser


def build_job(args: argparse.Namespace) -> Any:
    try:
        from hpc_client.models import JobSpec
    except ImportError as error:
        raise RuntimeError(
            "HPC support is not installed. " "Install CARES RL with the 'hpc' extra."
        ) from error

    if not args.image:
        raise ValueError(
            "No Docker image specified. " "Use --image or set CARES_RL_HPC_IMAGE."
        )

    if args.runtime <= 0:
        raise ValueError("Runtime must be greater than zero.")

    rl_args = list(args.rl_args)

    # Allow an optional separator before the forwarded arguments.
    if rl_args and rl_args[0] == "--":
        rl_args.pop(0)

    if not rl_args or rl_args[0] != "cli":
        raise ValueError(
            "Only 'train cli' is currently supported. "
            "Example: cares-rl-hpc --image cares-rl:v3.1.0 "
            "train cli --gym openai --task CartPole-v1 DQN"
        )

    if len(rl_args) == 1:
        raise ValueError("No CARES RL training arguments provided.")

    command = shlex.join(["cares-rl", args.command, *rl_args])

    return JobSpec(
        job_name=args.job_name,
        image=args.image,
        command=command,
        max_runtime_hours=args.runtime,
        resumable=args.resumable,
        required_datasets=args.dataset,
        required_worker_ids=args.worker,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        job = build_job(args)

        if args.dry_run:
            print(json.dumps(job.model_dump(), indent=2))
            return

        try:
            from hpc_client.client import HPCClient
        except ImportError as error:
            raise RuntimeError(
                "HPC support is not installed. "
                "Install CARES RL with the 'hpc' extra."
            ) from error

        client = HPCClient.from_config()

        try:
            result = client.submit(job)
        finally:
            client.close()

        print(json.dumps(result, indent=2))

    except (ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
