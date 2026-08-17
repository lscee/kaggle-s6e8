from __future__ import print_function

import argparse
import hashlib
import json
import os
import urllib.request


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OWNER = "ern711"
SLUG = "contextualized-deep-univariate-spline-transformer"
API_URL = (
    "https://www.kaggle.com/api/v1/kernels/output"
    "?userName={}&kernelSlug={}&pageSize=20".format(OWNER, SLUG)
)
SOURCE_API_URL = (
    "https://www.kaggle.com/api/v1/kernels/pull"
    "?userName={}&kernelSlug={}".format(OWNER, SLUG)
)
SOURCE_FILENAME = "contextualized_deep_univariate_spline_transformer_v3.py"
SOURCE_SHA256 = "a02a968711b0a721eb9ed046b81fafeef3775995e1a6c1548680cb319beed218"
FILES = (
    "nonlinear_context_5fold_oof.csv",
    "nonlinear_context_5fold_test_predictions.csv",
    "nonlinear_context_5fold_metrics.csv",
)


def parser():
    result = argparse.ArgumentParser(
        description="Download public spline OOF/test predictions from Kaggle"
    )
    result.add_argument(
        "--destination",
        default=os.path.join(
            ROOT, "data", "external", "public_oof", "spline_transformer_v3"
        ),
    )
    result.add_argument("--force", action="store_true")
    result.add_argument(
        "--include-source",
        action="store_true",
        help="Also download and hash-verify the Apache-2.0 notebook source",
    )
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    request = urllib.request.Request(
        API_URL,
        headers={"Accept": "application/json", "User-Agent": "kaggle-api/v1.7.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    available = {item["fileName"]: item["url"] for item in payload["files"]}
    missing = sorted(set(FILES) - set(available))
    if missing:
        raise RuntimeError("Kaggle output is missing files: {}".format(missing))

    os.makedirs(args.destination, exist_ok=True)
    for filename in FILES:
        destination = os.path.join(args.destination, filename)
        if os.path.isfile(destination) and not args.force:
            print("Exists: {}".format(destination))
            continue
        urllib.request.urlretrieve(available[filename], destination)
        print("Downloaded: {}".format(destination))

    if args.include_source:
        source_request = urllib.request.Request(
            SOURCE_API_URL,
            headers={
                "Accept": "application/json",
                "User-Agent": "kaggle-api/v1.7.0",
            },
        )
        with urllib.request.urlopen(source_request, timeout=60) as response:
            source_payload = json.loads(response.read().decode("utf-8"))
        notebook = json.loads(source_payload["blob"]["source"])
        code_cells = [
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        ]
        if len(code_cells) != 1:
            raise RuntimeError(
                "Expected exactly one code cell, found {}".format(len(code_cells))
            )
        source = code_cells[0]
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if digest != SOURCE_SHA256:
            raise RuntimeError(
                "Notebook source hash changed: expected {}, got {}".format(
                    SOURCE_SHA256, digest
                )
            )
        source_path = os.path.join(args.destination, SOURCE_FILENAME)
        if not os.path.isfile(source_path) or args.force:
            with open(source_path, "w", encoding="utf-8", newline="") as handle:
                handle.write(source)
        print("Verified source: {} ({})".format(source_path, digest))

    provenance = {
        "owner": OWNER,
        "slug": SLUG,
        "script_version_id": 342607757,
        "outer_fold_seed": 21,
        "reported_oof_auc": 0.9665204981918207,
        "license": "Apache-2.0",
        "api_url": API_URL,
        "files": list(FILES),
        "source_api_url": SOURCE_API_URL,
        "source_filename": SOURCE_FILENAME,
        "source_sha256": SOURCE_SHA256,
    }
    with open(
        os.path.join(args.destination, "provenance.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(provenance, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
