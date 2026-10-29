
"""Unit tests for the PipelineModelEvaluator.

Tests initialization, model fitting, prediction caching, evaluation
metrics, and result compilation using training and validation partitions.
"""

from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score

from califiornia_housing_ml_project.data.data_pipeline import UniversalDataPipeline
from califiornia_housing_ml_project.models.evaluator import PipelineModelEvaluator


class DummyModelInterface(RegressorMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.array([11.0, 19.0])

    def get_params(self, deep=True):
        return {}


@pytest.fixture
def mock_pipeline():
    """Create a mock data pipeline with distinct data partitions."""
    pipeline = MagicMock(spec=UniversalDataPipeline)

    # Training partition
    pipeline.X_train_split = np.array([[1.0, 2.0], [3.0, 4.0]])
    pipeline.y_train_split = np.array([10.0, 20.0])

    # Validation partition
    pipeline.X_val_split = np.array([[5.0, 6.0], [7.0, 8.0]])
    pipeline.y_val_split = np.array([30.0, 40.0])

    # Full training pool used for cross-validation
    pipeline.X_train = np.array(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    )
    pipeline.y_train = np.array([10.0, 20.0, 30.0, 40.0])

    # Combined transformed training pool
    pipeline.X_train_combined = pipeline.X_train.copy()

    # Held-out test partition
    pipeline.X_test_combined = np.array([[9.0, 10.0], [11.0, 12.0]])
    pipeline.y_test = np.array([50.0, 60.0])

    pipeline.transformer = MagicMock()

    return pipeline


@pytest.fixture
def mock_model_instance():
    """Create a mock estimator implementing the required model hooks."""
    model = MagicMock(spec=DummyModelInterface)
    model.fit = MagicMock(return_value=model)
    model.predict = MagicMock(return_value=np.array([11.0, 19.0]))
    return model


def test_initialization_state_with_valid_interfaces(
    mock_model_instance, mock_pipeline
):
    """Verify that initialization sets the expected internal state."""
    evaluator = PipelineModelEvaluator(mock_model_instance, mock_pipeline)

    assert evaluator._model_instance is mock_model_instance
    assert evaluator.pipeline is mock_pipeline
    assert evaluator._is_fitted is False
    assert evaluator._model_train_prediction is None
    assert evaluator._model_val_prediction is None


def test_initialization_raises_type_error_for_invalid_pipeline(
    mock_model_instance,
):
    """Reject an object that does not satisfy the pipeline type requirement."""
    invalid_pipeline = MagicMock()

    with pytest.raises(TypeError, match="pipeline must be an instance of"):
        PipelineModelEvaluator(mock_model_instance, invalid_pipeline)


def test_initialization_raises_type_error_for_missing_model_hooks(
    mock_pipeline,
):
    """Reject a model without a callable fit method."""
    broken_model = MagicMock(spec=DummyModelInterface)
    del broken_model.fit

    with pytest.raises(TypeError, match="must provide a callable fit"):
        PipelineModelEvaluator(broken_model, mock_pipeline)


@patch("califiornia_housing_ml_project.models.evaluator.clone")
def test_fit_lifecycle_clones_and_isolates_estimator(
    mock_clone, mock_model_instance, mock_pipeline
):
    """Verify cloning, training inputs, and fluent chaining."""
    mock_clone.return_value = mock_model_instance

    evaluator = PipelineModelEvaluator(mock_model_instance, mock_pipeline)
    result = evaluator.fit()

    mock_clone.assert_called_once_with(mock_model_instance)
    mock_model_instance.fit.assert_called_once_with(
        mock_pipeline.X_train_split,
        mock_pipeline.y_train_split,
    )

    assert evaluator._is_fitted is True
    assert result is evaluator


def test_predict_raises_runtime_error_if_unfitted(
    mock_model_instance, mock_pipeline
):
    """Prevent prediction before the estimator has been fitted."""
    evaluator = PipelineModelEvaluator(mock_model_instance, mock_pipeline)

    with pytest.raises(RuntimeError, match="The model has not been fitted"):
        evaluator.predict()


@patch("califiornia_housing_ml_project.models.evaluator.clone")
def test_predict_caches_arrays_and_invalidates_stale_metrics(
    mock_clone, mock_model_instance, mock_pipeline
):
    """Verify prediction inputs, cached arrays, and metric invalidation."""
    mock_clone.return_value = mock_model_instance

    train_pred = np.array([11.0, 19.0])
    val_pred = np.array([31.0, 39.0])

    mock_model_instance.predict.side_effect = [train_pred, val_pred]

    evaluator = PipelineModelEvaluator(mock_model_instance, mock_pipeline)
    evaluator.fit()

    # Simulate previously cached metrics.
    evaluator._model_train_r2_score = 0.95
    evaluator._model_val_rmse = 0.02

    result = evaluator.predict()

    # Verify the correct feature partitions are used in order.
    mock_model_instance.predict.assert_has_calls(
        [
            call(mock_pipeline.X_train_split),
            call(mock_pipeline.X_val_split),
        ]
    )
    assert mock_model_instance.predict.call_count == 2

    # Verify the predictions are cached correctly.
    np.testing.assert_array_equal(
        evaluator._model_train_prediction, train_pred
    )
    np.testing.assert_array_equal(
        evaluator._model_val_prediction, val_pred
    )

    # Verify stale metrics are cleared.
    assert evaluator._model_train_r2_score is None
    assert evaluator._model_val_rmse is None
    assert result is evaluator


@patch(
    "califiornia_housing_ml_project.models.evaluator.r2_score",
    return_value=0.98,
)
@patch(
    "califiornia_housing_ml_project.models.evaluator.root_mean_squared_error",
    return_value=0.03,
)
@patch(
    "califiornia_housing_ml_project.models.evaluator.mean_absolute_error",
    return_value=0.01,
)
@patch("califiornia_housing_ml_project.models.evaluator.clone")
def test_complete_evaluation_and_results_compilation(
    mock_clone,
    mock_mae,
    mock_rmse,
    mock_r2,
    mock_model_instance,
    mock_pipeline,
):
    """Verify metric inputs and the training/validation results payload."""
    mock_clone.return_value = mock_model_instance

    train_pred = np.array([11.0, 19.0])
    val_pred = np.array([31.0, 39.0])

    mock_model_instance.predict.side_effect = [train_pred, val_pred]

    mock_pipeline.inverse_transform.side_effect = lambda values: np.asarray(values)
    evaluator = PipelineModelEvaluator(mock_model_instance, mock_pipeline)
    evaluator.fit()

    result = evaluator.evaluate()
    payload = evaluator.get_results()

    # Each metric must use the matching targets and predictions.
    expected_metric_calls = [
        call(mock_pipeline.y_train_split, train_pred),
        call(mock_pipeline.y_val_split, val_pred),
    ]

    mock_r2.assert_has_calls(expected_metric_calls)
    assert mock_r2.call_count == 2

    mock_rmse.assert_has_calls(expected_metric_calls)
    assert mock_rmse.call_count == 2

    mock_mae.assert_has_calls(expected_metric_calls)
    assert mock_mae.call_count == 2

    assert result is evaluator

    assert payload["train"]["r2"] == 0.98
    assert payload["validation"]["rmse"] == 0.03
    assert payload["validation"]["mae"] == 0.01


def test_model_name_property_extracts_class_name(
    mock_model_instance, mock_pipeline
):
    """Verify that model_name exposes the estimator class name."""
    mock_model_instance.__class__.__name__ = "CustomSequentialNetwork"

    evaluator = PipelineModelEvaluator(mock_model_instance, mock_pipeline)

    assert evaluator.model_name == "CustomSequentialNetwork"

@patch("califiornia_housing_ml_project.models.evaluator.cross_validate")
@patch("califiornia_housing_ml_project.models.evaluator.clone")
@patch("califiornia_housing_ml_project.models.evaluator.CheckModelType.run_check")
def test_cross_validation_uses_full_training_pool(
    mock_check, mock_clone, mock_cross_validate,
    mock_model_instance, mock_pipeline,
):
    """Verify cross-validation uses the full training pool."""

    mock_clone.side_effect = [
        mock_pipeline.transformer,
        mock_model_instance,
    ]

    expected_results = {"test_score": np.array([0.8, 0.9])}
    mock_cross_validate.return_value = expected_results

    evaluator = PipelineModelEvaluator(mock_model_instance, mock_pipeline)

    result = evaluator.cross_validation(n_split=2, n_jobs=1)

    args, kwargs = mock_cross_validate.call_args

    np.testing.assert_array_equal(args[1], mock_pipeline.X_train)
    np.testing.assert_array_equal(args[2], mock_pipeline.y_train)

    assert args[0].named_steps["preprocessor"] is mock_pipeline.transformer
    assert args[0].named_steps["model"] is mock_model_instance

    assert kwargs["cv"].n_splits == 2
    assert kwargs["return_train_score"] is True
    assert kwargs["n_jobs"] == 1

    assert evaluator._model_cv is expected_results
    assert result is evaluator


@pytest.mark.parametrize(
    "kwargs, expected_exception",
    [
        ({"n_split": 1}, ValueError),
        ({"n_split": 2.5}, TypeError),
        ({"n_split": True}, TypeError),
        ({"n_jobs": 1.5}, TypeError),
        ({"n_jobs": False}, TypeError),
    ],
)
def test_cross_validation_rejects_invalid_arguments(
    mock_model_instance,
    mock_pipeline,
    kwargs,
    expected_exception,
):
    """Reject invalid cross-validation configuration values."""
    evaluator = PipelineModelEvaluator(mock_model_instance, mock_pipeline)

    with pytest.raises(expected_exception):
        evaluator.cross_validation(**kwargs)



@patch("califiornia_housing_ml_project.models.evaluator.clone")
def test_fit_final_model_uses_full_training_pool(
    mock_clone, mock_model_instance, mock_pipeline
):
    """Verify final retraining uses the full raw training pool."""
    mock_transformer = MagicMock()
    mock_final_estimator = MagicMock()

    mock_clone.side_effect = [
        mock_transformer,
        mock_final_estimator,
    ]

    evaluator = PipelineModelEvaluator(mock_model_instance, mock_pipeline)

    result = evaluator.fit_final_model()

    # Verify both components were cloned.
    assert mock_clone.call_count == 2
    mock_clone.assert_has_calls(
        [
            call(mock_pipeline.transformer),
            call(mock_model_instance),
        ]
    )

    # Verify preprocessing and estimator are in the final pipeline.
    assert evaluator._final_model.named_steps["preprocessor"] is mock_transformer
    assert evaluator._final_model.named_steps["model"] is mock_final_estimator

    # Verify fitting uses the full raw training pool.
    mock_transformer.fit_transform.assert_called_once()
    np.testing.assert_array_equal(
        mock_transformer.fit_transform.call_args.args[0],
        mock_pipeline.X_train,
    )

    mock_final_estimator.fit.assert_called_once()
    np.testing.assert_array_equal(
        mock_final_estimator.fit.call_args.args[1],
        mock_pipeline.y_train,
    )

    assert result is evaluator


def test_predict_final_test_uses_raw_test_features(
    mock_model_instance, mock_pipeline
):
    """Verify final prediction uses raw held-out test features."""
    evaluator = PipelineModelEvaluator(mock_model_instance, mock_pipeline)

    final_model = MagicMock()
    expected_predictions = np.array([51.0, 59.0])
    final_model.predict.return_value = expected_predictions

    # Simulate a fitted final pipeline.
    evaluator._final_model = final_model

    # Simulate stale metrics from a previous prediction.
    evaluator._final_test_r2_score = 0.9
    evaluator._final_test_rmse = 0.1
    evaluator._final_test_mae = 0.05

    result = evaluator.predict_final_test()

    final_model.predict.assert_called_once_with(mock_pipeline.X_test)

    np.testing.assert_array_equal(
        evaluator._final_test_prediction,
        expected_predictions,
    )

    assert evaluator._final_test_r2_score is None
    assert evaluator._final_test_rmse is None
    assert evaluator._final_test_mae is None
    assert result is evaluator


def test_predict_final_test_requires_fitted_final_model(
    mock_model_instance, mock_pipeline
    ):
    """Reject final test prediction before final model fitting."""
    evaluator = PipelineModelEvaluator(mock_model_instance, mock_pipeline)

    with pytest.raises(
        RuntimeError,
        match="Call fit_final_model",
    ):
        evaluator.predict_final_test()


def evaluate_final_test(self) -> Self:
    """Evaluate the final model on the held-out test set."""
    if self._final_test_prediction is None:
        raise RuntimeError(
            "Final test predictions are unavailable. "
            "Call predict_final_test() first."
        )

    self._final_test_r2_score = r2_score(
        self.pipeline.y_test,
        self._final_test_prediction,
    )

    self._final_test_rmse = root_mean_squared_error(
        self.pipeline.y_test,
        self._final_test_prediction,
    )

    self._final_test_mae = mean_absolute_error(
        self.pipeline.y_test,
        self._final_test_prediction,
    )

    return self


def test_evaluate_final_test_uses_held_out_targets(
    mock_model_instance, mock_pipeline
):
    """Verify final metrics use held-out test targets and predictions."""
    evaluator = PipelineModelEvaluator(mock_model_instance, mock_pipeline)

    # Known held-out targets and predictions.
    mock_pipeline.y_test = np.array([100.0, 200.0, 300.0])
    evaluator._final_test_prediction = np.array([110.0, 190.0, 310.0])

    result = evaluator.evaluate_final_test()

    expected_r2 = r2_score(
        mock_pipeline.y_test,
        evaluator._final_test_prediction,
    )
    expected_rmse = np.sqrt(
        np.mean(
            (
                mock_pipeline.y_test
                - evaluator._final_test_prediction
            ) ** 2
        )
    )
    expected_mae = np.mean(
        np.abs(
            mock_pipeline.y_test
            - evaluator._final_test_prediction
        )
    )

    assert evaluator._final_test_r2_score == pytest.approx(expected_r2)
    assert evaluator._final_test_rmse == pytest.approx(expected_rmse)
    assert evaluator._final_test_mae == pytest.approx(expected_mae)

    assert result is evaluator


def test_evaluate_final_test_requires_predictions(
    mock_model_instance, mock_pipeline
):
    """Reject final evaluation when test predictions are unavailable."""
    evaluator = PipelineModelEvaluator(mock_model_instance, mock_pipeline)

    with pytest.raises(
        RuntimeError,
        match="Call predict_final_test",
    ):
        evaluator.evaluate_final_test()



def test_get_final_test_results_returns_metrics(
    mock_model_instance, mock_pipeline
):
    """Verify final test results contain the expected metrics."""
    evaluator = PipelineModelEvaluator(mock_model_instance, mock_pipeline)

    evaluator._final_test_r2_score = 0.85
    evaluator._final_test_rmse = 0.42
    evaluator._final_test_mae = 0.31

    results = evaluator.get_final_test_results()

    assert results == {
        "r2_score": 0.85,
        "rmse": 0.42,
        "mae": 0.31,
    }


def test_get_final_test_results_requires_evaluation(
    mock_model_instance, mock_pipeline
):
    """Reject retrieving final results before evaluation."""
    evaluator = PipelineModelEvaluator(mock_model_instance, mock_pipeline)

    with pytest.raises(
        RuntimeError,
        match="Call evaluate_final_test",
    ):
        evaluator.get_final_test_results()

@patch("califiornia_housing_ml_project.models.evaluator.clone")
def test_error_metrics_use_original_target_scale(
    mock_clone,
    mock_model_instance,
    mock_pipeline,
):
    """Verify RMSE and MAE are calculated in original target units."""
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error

    mock_clone.return_value = mock_model_instance

    train_predictions = np.array([1.5, 2.5])
    val_predictions = np.array([3.5, 4.5])

    mock_model_instance.predict.side_effect = [
        train_predictions,
        val_predictions,
    ]

    # Simulate converting scaled values back to original units.
    mock_pipeline.inverse_transform.side_effect = (
        lambda values: np.asarray(values) * 100_000
    )

    evaluator = PipelineModelEvaluator(mock_model_instance, mock_pipeline)

    evaluator.fit()
    evaluator.predict()

    evaluator.train_rmse()
    evaluator.train_mae()
    evaluator.val_rmse()
    evaluator.val_mae()

    results = evaluator.get_results()

    expected_train_rmse = root_mean_squared_error(
        mock_pipeline.y_train_split * 100_000,
        train_predictions * 100_000,
    )
    expected_val_rmse = root_mean_squared_error(
        mock_pipeline.y_val_split * 100_000,
        val_predictions * 100_000,
    )

    expected_train_mae = mean_absolute_error(
        mock_pipeline.y_train_split * 100_000,
        train_predictions * 100_000,
    )
    expected_val_mae = mean_absolute_error(
        mock_pipeline.y_val_split * 100_000,
        val_predictions * 100_000,
    )

    assert evaluator._model_train_rmse == pytest.approx(expected_train_rmse)
    assert evaluator._model_val_rmse == pytest.approx(expected_val_rmse)
    assert evaluator._model_train_mae == pytest.approx(expected_train_mae)
    assert evaluator._model_val_mae == pytest.approx(expected_val_mae)