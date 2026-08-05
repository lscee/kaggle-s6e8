from __future__ import print_function

import argparse
import json
import os

from .audit import run_audit
from .config import ensure_output_dir, load_config, project_root
from .data import load_competition_data
from .demo import make_demo_data
from .ensemble import run_ensemble
from .hardware import run_gpu_check
from .io_utils import write_json
from .public_stack import run_public_stack
from .training import train_models


def _parser():
    default_config = os.path.join(project_root(), "configs", "base.yaml")
    parser = argparse.ArgumentParser(
        description="Kaggle Playground S6E8 competition pipeline"
    )
    parser.add_argument("--config", default=default_config, help="YAML config path")
    parser.add_argument("--output-dir", help="Override experiment output directory")
    parser.add_argument(
        "--drop-columns",
        default=None,
        help="Comma-separated extra feature columns to drop",
    )
    parser.add_argument("--use-id", action="store_true", help="Include ID as a feature")
    parser.add_argument(
        "--no-engineered", action="store_true", help="Disable engineered features"
    )
    subparsers = parser.add_subparsers(dest="command")

    audit_parser = subparsers.add_parser("audit", help="Run leakage and data audit")
    audit_parser.add_argument(
        "--adversarial", action="store_true", help="Run train-vs-test classifier"
    )

    train_parser = subparsers.add_parser("train", help="Train OOF models")
    train_parser.add_argument(
        "--models",
        nargs="+",
        help="Override model or configured experiment names",
    )

    subparsers.add_parser("blend", help="Build greedy rank ensemble and submission")
    subparsers.add_parser(
        "stack-public",
        help="Audit and stack the aligned public seed-42 OOF library",
    )

    all_parser = subparsers.add_parser("all", help="Audit, train, and blend")
    all_parser.add_argument(
        "--models",
        nargs="+",
        help="Override model or configured experiment names",
    )
    all_parser.add_argument(
        "--adversarial", action="store_true", help="Run train-vs-test classifier"
    )

    demo_parser = subparsers.add_parser(
        "demo-data", help="Create synthetic CSV files for a smoke test"
    )
    demo_parser.add_argument("--rows", type=int, default=1400, help="Training rows")
    demo_parser.add_argument("--test-rows", type=int, default=500, help="Test rows")
    demo_parser.add_argument(
        "--destination",
        default=os.path.join(project_root(), "data", "raw"),
        help="Destination directory",
    )
    subparsers.add_parser(
        "gpu-check", help="Fit tiny CatBoost and LightGBM models on the GPU"
    )
    return parser


def _resolved_config(args):
    config = load_config(args.config)
    if args.output_dir:
        path = args.output_dir
        if not os.path.isabs(path):
            path = os.path.join(project_root(), path)
        config["output"]["directory"] = os.path.abspath(path)
    if args.drop_columns is not None:
        config["features"]["drop_columns"] = [
            column.strip()
            for column in args.drop_columns.split(",")
            if column.strip()
        ]
    if args.use_id:
        config["features"]["use_id"] = True
    if args.no_engineered:
        config["features"]["engineered"] = False
    return config


def _save_resolved_config(config, output_dir):
    payload = {
        key: value
        for key, value in config.items()
        if not str(key).startswith("_")
    }
    write_json(os.path.join(output_dir, "config_resolved.json"), payload)


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return
    if args.command == "demo-data":
        make_demo_data(args.destination, args.rows, args.test_rows)
        print("Demo data written to {}".format(os.path.abspath(args.destination)))
        return

    config = _resolved_config(args)
    if args.command == "gpu-check":
        print(json.dumps(run_gpu_check(config), indent=2, sort_keys=True))
        return

    output_dir = ensure_output_dir(config)
    _save_resolved_config(config, output_dir)
    train, test, sample = load_competition_data(config, require_sample=True)

    if args.command == "audit":
        summary, univariate = run_audit(
            train, test, config, output_dir, adversarial=args.adversarial
        )
        print("Target mean: {:.6f}".format(summary["target_mean"]))
        print(univariate.head(10).to_string(index=False))
        return

    if args.command == "train":
        summary = train_models(
            train, test, config, output_dir, requested_models=args.models
        )
        print(summary.to_string(index=False))
        return

    if args.command == "blend":
        payload = run_ensemble(train, test, sample, config, output_dir)
        print("Ensemble OOF AUC: {:.6f}".format(payload["ensemble_oof_auc"]))
        print("Weights: {}".format(payload["weights"]))
        print("Submission: {}".format(payload["submission_path"]))
        return

    if args.command == "stack-public":
        payload = run_public_stack(train, test, sample, config, output_dir)
        print("Public stack OOF AUC: {:.6f}".format(payload["selected_oof_auc"]))
        print("Selected meta-model: {}".format(payload["selected"]))
        print("Submission: {}".format(payload["submission_path"]))
        return

    if args.command == "all":
        summary, univariate = run_audit(
            train, test, config, output_dir, adversarial=args.adversarial
        )
        print("Top univariate features:")
        print(univariate.head(10).to_string(index=False))
        train_summary = train_models(
            train, test, config, output_dir, requested_models=args.models
        )
        print(train_summary.to_string(index=False))
        payload = run_ensemble(train, test, sample, config, output_dir)
        print("Ensemble OOF AUC: {:.6f}".format(payload["ensemble_oof_auc"]))
        print("Weights: {}".format(payload["weights"]))
        print("Submission: {}".format(payload["submission_path"]))
        return

    raise ValueError("Unsupported command: {}".format(args.command))
