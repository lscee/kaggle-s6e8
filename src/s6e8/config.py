from __future__ import print_function

import copy
import os

import yaml


def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _resolve_path(root, value):
    if not value:
        return value
    if os.path.isabs(value):
        return value
    return os.path.abspath(os.path.join(root, value))


def _deep_merge(base, override):
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_mapping(path, stack=None):
    path = os.path.abspath(path)
    stack = tuple(stack or ())
    if path in stack:
        chain = " -> ".join(stack + (path,))
        raise ValueError("Circular config inheritance: {}".format(chain))

    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config must contain a YAML mapping")

    config = copy.deepcopy(config)
    parent = config.pop("extends", None)
    if not parent:
        return config
    if not isinstance(parent, str):
        raise ValueError("Config 'extends' must be a YAML string")
    if not os.path.isabs(parent):
        parent = os.path.join(os.path.dirname(path), parent)
    base = _load_mapping(parent, stack + (path,))
    return _deep_merge(base, config)


def load_config(path):
    path = os.path.abspath(path)
    config = _load_mapping(path)

    root = project_root()
    config = copy.deepcopy(config)
    for key in (
        "train_path",
        "test_path",
        "sample_submission_path",
        "parent_data_path",
    ):
        config["data"][key] = _resolve_path(root, config["data"].get(key))
    config["output"]["directory"] = _resolve_path(
        root, config["output"]["directory"]
    )
    config["_config_path"] = path
    config["_project_root"] = root
    return config


def ensure_output_dir(config):
    output_dir = config["output"]["directory"]
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    return output_dir
