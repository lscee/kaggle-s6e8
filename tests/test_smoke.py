from __future__ import print_function

import copy
import os
import shutil
import sys
import tempfile
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from s6e8.audit import run_audit  # noqa: E402
from s6e8.config import load_config  # noqa: E402
from s6e8.data import load_competition_data  # noqa: E402
from s6e8.demo import make_demo_data  # noqa: E402
from s6e8.ensemble import run_ensemble  # noqa: E402
from s6e8.features import (  # noqa: E402
    add_generator_artifact_features,
    encode_for_sklearn,
    prepare_feature_frames,
)
from s6e8.models import model_available  # noqa: E402
from s6e8.public_stack import (  # noqa: E402
    load_additional_predictions,
    load_logit_parquet_sources,
    _reorder_inside_subset,
    regime_design,
)
from s6e8.training import train_models  # noqa: E402


class PipelineSmokeTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="s6e8_test_")
        self.data_dir = os.path.join(self.temp_dir, "data")
        self.output_dir = os.path.join(self.temp_dir, "output")
        make_demo_data(self.data_dir, train_rows=240, test_rows=80, seed=17)
        self.config = load_config(os.path.join(ROOT, "configs", "base.yaml"))
        self.config = copy.deepcopy(self.config)
        self.config["data"]["train_path"] = os.path.join(self.data_dir, "train.csv")
        self.config["data"]["test_path"] = os.path.join(self.data_dir, "test.csv")
        self.config["data"]["sample_submission_path"] = os.path.join(
            self.data_dir, "sample_submission.csv"
        )
        self.config["output"]["directory"] = self.output_dir
        self.config["cv"]["n_splits"] = 3
        self.config["training"]["seeds"] = [17]
        self.config["models"]["extra_trees"].update(
            {"n_estimators": 30, "max_depth": 8, "min_samples_leaf": 2}
        )
        self.config["models"]["logistic"]["max_iter"] = 300
        os.makedirs(self.output_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_end_to_end(self):
        train, test, sample = load_competition_data(self.config)
        train_x, test_x = prepare_feature_frames(train, test, self.config)
        self.assertEqual(list(train_x.columns), list(test_x.columns))
        self.assertIn("weekend_delta", train_x.columns)
        self.assertIn("missing_count", train_x.columns)
        filled_train, _ = encode_for_sklearn(train_x, test_x, fill_missing=True)
        native_train, _ = encode_for_sklearn(train_x, test_x, fill_missing=False)
        self.assertEqual(str(filled_train.dtypes.iloc[0]), "float32")
        self.assertEqual(int(filled_train.isnull().sum().sum()), 0)
        self.assertGreater(int(native_train.isnull().sum().sum()), 0)
        self.assertTrue(
            any(column.endswith("__missing") for column in filled_train.columns)
        )

        summary, univariate = run_audit(
            train, test, self.config, self.output_dir, adversarial=False
        )
        self.assertEqual(summary["train_shape"][0], 240)
        self.assertGreater(float(univariate.iloc[0]["univariate_auc"]), 0.70)

        model_summary = train_models(
            train,
            test,
            self.config,
            self.output_dir,
            requested_models=["extra_trees", "logistic"],
        )
        self.assertEqual(set(model_summary["model"]), {"extra_trees", "logistic"})

        payload = run_ensemble(
            train, test, sample, self.config, self.output_dir
        )
        self.assertGreater(payload["ensemble_oof_auc"], 0.70)
        self.assertTrue(os.path.isfile(os.path.join(self.output_dir, "submission.csv")))

    def test_optional_boosters(self):
        if not model_available("catboost") or not model_available("lightgbm"):
            self.skipTest("CatBoost and LightGBM are optional in the active environment")
        train, test, sample = load_competition_data(self.config)
        self.config["models"]["catboost"].update(
            {
                "iterations": 80,
                "depth": 5,
                "learning_rate": 0.08,
                "early_stopping_rounds": 15,
            }
        )
        self.config["models"]["lightgbm"].update(
            {
                "n_estimators": 120,
                "num_leaves": 15,
                "early_stopping_rounds": 15,
            }
        )
        model_summary = train_models(
            train,
            test,
            self.config,
            self.output_dir,
            requested_models=["catboost", "lightgbm"],
        )
        self.assertEqual(set(model_summary["model"]), {"catboost", "lightgbm"})
        payload = run_ensemble(
            train, test, sample, self.config, self.output_dir
        )
        self.assertGreater(payload["ensemble_oof_auc"], 0.70)

    def test_gpu_config_inherits_base(self):
        config = load_config(os.path.join(ROOT, "configs", "gpu_wsl.yaml"))
        self.assertEqual(config["project"]["target"], "addicted_label")
        self.assertEqual(config["models"]["catboost"]["task_type"], "GPU")
        self.assertEqual(config["models"]["lightgbm"]["device_type"], "cuda")
        self.assertEqual(config["models"]["extra_trees"]["n_estimators"], 300)
        self.assertTrue(
            config["output"]["directory"].endswith(
                os.path.join("outputs", "gpu_wsl")
            )
        )

    def test_generator_artifact_features(self):
        import numpy as np
        import pandas as pd

        frame = pd.DataFrame(
            {
                "daily_screen_time_hours": [11.2, 8.42, np.nan],
                "social_media_hours": [1.87, 2.57, 1.0],
                "gaming_hours": [2.81, 3.41, 1.0],
                "work_study_hours": [1.95, 1.9, 1.0],
                "sleep_hours": [5.25, 6.64, 7.0],
                "weekend_screen_time": [13.39, 8.35, 8.0],
                "age": [26.0, 21.0, 30.0],
                "notifications_per_day": [100.0, 80.0, 60.0],
                "app_opens_per_day": [20.0, 15.0, 10.0],
            }
        )
        result = add_generator_artifact_features(
            frame, include_exact_categories=True
        )
        self.assertEqual(
            result["daily_screen_time_hours__first_decimal_digit"].tolist()[:2],
            ["2", "4"],
        )
        self.assertEqual(
            result["daily_screen_time_hours__decimal_length"].tolist()[:2],
            ["1", "2"],
        )
        self.assertAlmostEqual(
            float(result.loc[0, "screen_component_residual"]), 4.57, places=6
        )
        self.assertEqual(
            result.loc[0, "daily_screen_time_hours__exact_level"], "11.2"
        )
        self.assertTrue(
            pd.isna(result.loc[2, "daily_screen_time_hours__first_decimal_digit"])
        )

    def test_artifact_feature_views_are_aligned(self):
        train, test, _ = load_competition_data(self.config)
        for view in ("artifact", "engineered_artifact", "artifact_cat"):
            train_x, test_x = prepare_feature_frames(
                train, test, self.config, view=view
            )
            self.assertEqual(list(train_x.columns), list(test_x.columns))
            self.assertIn(
                "daily_screen_time_hours__first_decimal_digit", train_x.columns
            )
        artifact_train, _ = prepare_feature_frames(
            train, test, self.config, view="artifact_cat"
        )
        self.assertIn("daily_screen_time_hours__exact_level", artifact_train.columns)

    def test_fm_rank_config_preserves_the_complete_stack(self):
        config = load_config(
            os.path.join(ROOT, "configs", "leaderboard_v4_fm_rank.yaml")
        )
        prediction_sources = config["public_stack"][
            "additional_prediction_sources"
        ]
        self.assertEqual(len(prediction_sources), 2)
        self.assertEqual(
            prediction_sources[1]["include_models"], ["fm_rank_cross"]
        )
        self.assertEqual(
            len(config["public_stack"]["additional_logit_parquet_sources"]), 1
        )
        self.assertEqual(config["public_stack"]["regime_c_grid"], [0.03])

    def test_budget_geometry_values_and_validity_align(self):
        import importlib.util
        import numpy as np

        if importlib.util.find_spec("torch") is None:
            self.skipTest("PyTorch is optional and tested in the WSL GPU environment")
        script_path = os.path.join(ROOT, "scripts", "train_budget_cross.py")
        spec = importlib.util.spec_from_file_location("train_budget_cross", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        values = np.array(
            [
                [24, 8, 2, 1, 3, 7, 100, 20, 9],
                [30, 0, 2, 0, 1, 8, 0, 15, 0],
            ],
            dtype=np.float32,
        )
        observed = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 1, 0, 1, 1, 0, 1, 0],
            ],
            dtype=bool,
        )
        geometry, valid = module.budget_geometry_numpy(values, observed, 15.0)
        self.assertEqual(geometry.shape, valid.shape)
        self.assertEqual(geometry.shape[1], 31)
        self.assertTrue(np.isfinite(geometry).all())
        self.assertGreater(int(valid[0].sum()), int(valid[1].sum()))

    def test_public_regime_design(self):
        import numpy as np

        logits = np.arange(24, dtype=np.float64).reshape(6, 4) / 10.0
        missing = np.array([0, 1, 2, 3, 4, 5])
        design, mean, std = regime_design(logits, missing)
        test_design, test_mean, test_std = regime_design(
            logits, missing, disagreement_mean=mean, disagreement_std=std
        )
        self.assertEqual(design.shape, (6, 4 * 4 + 5))
        self.assertTrue(np.array_equal(design, test_design))
        self.assertEqual(mean, test_mean)
        self.assertEqual(std, test_std)

    def test_additional_predictions_reject_wrong_folds(self):
        import numpy as np
        import pandas as pd

        train, test, _ = load_competition_data(self.config)
        wrong_dir = os.path.join(self.temp_dir, "wrong_folds")
        os.makedirs(wrong_dir)
        pd.DataFrame(
            {
                "id": train["id"],
                "addicted_label": train["addicted_label"],
                "fold": np.zeros(len(train), dtype=int),
            }
        ).to_csv(os.path.join(wrong_dir, "folds.csv"), index=False)
        folds = [(np.arange(80, len(train)), np.arange(80))]
        with self.assertRaises(ValueError):
            load_additional_predictions(
                train,
                test,
                [{"directory": wrong_dir}],
                "addicted_label",
                "id",
                folds,
            )

    def test_logit_parquet_source_alignment(self):
        import numpy as np
        import pandas as pd

        train, test, _ = load_competition_data(self.config)
        source_dir = os.path.join(self.temp_dir, "logit_source")
        os.makedirs(source_dir)
        pd.DataFrame({"id": train["id"], "member": np.arange(len(train))}).to_parquet(
            os.path.join(source_dir, "oof_members.parquet"), index=False
        )
        pd.DataFrame({"id": test["id"], "member": np.arange(len(test))}).to_parquet(
            os.path.join(source_dir, "test_members.parquet"), index=False
        )
        names, oof, test_pred, scores = load_logit_parquet_sources(
            train,
            test,
            [{"directory": source_dir, "prefix": "fm"}],
            "addicted_label",
            "id",
        )
        self.assertEqual(names, ["fm__member"])
        self.assertEqual(oof.shape, (len(train), 1))
        self.assertEqual(test_pred.shape, (len(test), 1))
        self.assertEqual(scores.shape[0], 1)

    def test_subset_reorder_preserves_values_and_outside_rows(self):
        import numpy as np

        base = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        indices = np.array([1, 3, 4])
        local = np.array([3.0, 2.0, 1.0])
        result = _reorder_inside_subset(base, indices, local, 1.0)
        self.assertTrue(np.array_equal(result[[0, 2]], base[[0, 2]]))
        self.assertTrue(np.array_equal(np.sort(result[indices]), np.sort(base[indices])))
        self.assertTrue(np.array_equal(result[indices], np.array([0.5, 0.4, 0.2])))


if __name__ == "__main__":
    unittest.main()
