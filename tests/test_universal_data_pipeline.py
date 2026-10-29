"""Provides comprehensive unit test structures for the UniversalDataPipeline.

This test module utilizes pytest execution suites to validate boundary conditions, type rules, 
and behavioural expectations across the unified data engineering pipeline. It guarantees 
strict orchestration sequence enforcement and prevents data leakage across splits.

Validation Scenarios Covered:
    - Sequence Execution Guards: Asserting that operations fail appropriately when 
      splits or transformations are attempted out of sequential order.
    - State Reset Integrity: Verifying that re-running splits or replacing the 
      internal transformer correctly flushes dependent intermediate arrays and models.
    - Error Isolation: Testing that structural validation dropouts (such as missing 
      columns) completely strip old matrices to prevent stale data exposure.
    - Production Inference Cleanup: Confirming that clearing raw training frames successfully 
      wipes memory footprints while keeping the fitted transformer and model usable for inference.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from califiornia_housing_ml_project.data.data_pipeline import UniversalDataPipeline
from califiornia_housing_ml_project.data.preprocessing import (
    AgnosticTabularPreprocessor,
)
from califiornia_housing_ml_project.errors.custom_errors import (
    AgnosticTabularPreprocessorError,
)


@pytest.fixture
def sample_data():
    """Generates a predictable baseline feature matrix and continuous target series."""
    X = pd.DataFrame({
        "MedInc": [3.5, 5.0, 2.0, 4.0, 6.0, 3.0, 7.0, 2.5],
        "HouseAge": 5,
    })
    y = pd.Series([100, 200, 80, 180, 250, 150, 300, 90])
    return X, y


@pytest.fixture
def configured_pipeline(sample_data):
    """Initialises a data pipeline pre-loaded with an AgnosticTabularPreprocessor engine."""
    X, y = sample_data

    pipeline = UniversalDataPipeline(
        X=X,
        y=y,
        test_size=0.25,
        random_state=42,
    )

    transformer = AgnosticTabularPreprocessor(
        numerical_columns=["MedInc", "HouseAge"],
    )

    pipeline.set_transformer(transformer)
    return pipeline


def test_split_requires_transformer(configured_pipeline):
    """Ensures test partitioning drops out if no valid preprocessor has been linked."""
    configured_pipeline.transformer = None

    with pytest.raises(AgnosticTabularPreprocessorError):
        configured_pipeline.run_test_split()


def test_transform_requires_split(configured_pipeline):
    """Ensures fit-transform routines drop out if baseline dataset splits do not exist yet."""
    with pytest.raises(RuntimeError):
        configured_pipeline.fit_and_transform()


def test_transform_new_data_requires_fitted_transformer(
    configured_pipeline,
    sample_data,
):
    """Ensures forward inference transformations reject input before the engine has fitted."""
    X, _ = sample_data

    with pytest.raises(RuntimeError):
        configured_pipeline.transform_new_data(X)


def test_fit_and_transform_produces_expected_arrays(
    configured_pipeline,
):
    """Validates that execution pathways yield arrays matching extraction targets."""
    configured_pipeline.run_test_split()
    configured_pipeline.fit_and_transform()

    assert isinstance(configured_pipeline.X_train_combined, np.ndarray)
    assert isinstance(configured_pipeline.X_test_combined, np.ndarray)

    assert (
        configured_pipeline.X_train_combined.shape[0]
        == len(configured_pipeline.X_train)
    )
    assert (
        configured_pipeline.X_test_combined.shape[0]
        == len(configured_pipeline.X_test)
    )


def test_transform_new_data_after_fit(configured_pipeline, sample_data):
    """Confirms that forward processing maps test sample matrices to correct feature lengths."""
    X, _ = sample_data

    configured_pipeline.run_test_split()
    configured_pipeline.fit_and_transform()

    result = configured_pipeline.transform_new_data(X.iloc[:2])

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)


def test_replacing_transformer_resets_fitted_state(configured_pipeline):
    """Guarantees that injecting a replacement engine flushes all active fitted indicators."""
    configured_pipeline.run_test_split()
    configured_pipeline.fit_and_transform()

    assert configured_pipeline._is_transformer_fitted is True

    replacement = AgnosticTabularPreprocessor(
        numerical_columns=["MedInc", "HouseAge"],
    )
    configured_pipeline.set_transformer(replacement)

    assert configured_pipeline._is_transformer_fitted is False
    assert configured_pipeline._has_transformed is False

    with pytest.raises(RuntimeError, match="fitted"):
        configured_pipeline.transform_new_data(
            configured_pipeline.X_test
        )


def test_failed_refit_does_not_expose_old_arrays(configured_pipeline):
    """Ensures a failed transformation scrub-wipes pre-existing matrices to prevent leakage."""
    configured_pipeline.run_test_split()
    configured_pipeline.fit_and_transform()
    configured_pipeline.run_validation_split()
    configured_pipeline.fit_model(DummyRegressor())

    assert configured_pipeline._has_transformed is True

    # Replace the transformer with one that cannot fit the data.
    replacement = AgnosticTabularPreprocessor(
        numerical_columns=["missing_column"],
    )
    configured_pipeline.set_transformer(replacement)

    with pytest.raises(ValueError, match="missing required columns"):
        configured_pipeline.fit_and_transform()

    assert configured_pipeline._is_transformer_fitted is False
    assert configured_pipeline._has_transformed is False
    assert configured_pipeline._X_train_combined is None
    assert configured_pipeline._X_test_combined is None


def test_rerunning_split_resets_dependent_state(configured_pipeline):
    """Confirms that re-triggering splits fully invalidates all downstream configurations and states."""
    
    # First split, fit the transformer, and train a model.
    configured_pipeline.run_test_split()
    configured_pipeline.fit_and_transform()
    configured_pipeline.run_validation_split()
    configured_pipeline.fit_model(DummyRegressor())

    assert configured_pipeline._has_split is True
    assert configured_pipeline._is_transformer_fitted is True
    assert configured_pipeline._has_transformed is True
    assert configured_pipeline._model is not None

    # Create a new split.
    configured_pipeline.run_test_split()

    # The new split must invalidate all dependent state.
    assert configured_pipeline._has_split is True
    assert configured_pipeline._is_transformer_fitted is False
    assert configured_pipeline._has_transformed is False

    assert configured_pipeline._X_train_combined is None
    assert configured_pipeline._X_test_combined is None
    assert configured_pipeline._model is None

    # Raw data partitions should be available for the new split.
    assert configured_pipeline._X_train is not None
    assert configured_pipeline._X_test is not None

    # Fitting a new transformer is required before accessing the model.
    with pytest.raises(RuntimeError, match="fit_model"):
        _ = configured_pipeline.trained_model


def test_transform_new_data_rejects_unfitted_pipeline_transformer(
    configured_pipeline,
):
    """Verifies that forward processing checks actively catch un-fitted engine modifications."""
    configured_pipeline.run_test_split()
    configured_pipeline.fit_and_transform()

    # Replacing the transformer invalidates the fitted state.
    replacement = AgnosticTabularPreprocessor(
        numerical_columns=["MedInc", "HouseAge"],
    )
    configured_pipeline.set_transformer(replacement)

    with pytest.raises(RuntimeError, match="fitted"):
        configured_pipeline.transform_new_data(
            configured_pipeline.X_test
        )


def test_clear_training_data_preserves_inference_capability(
    configured_pipeline,
):
    """Guarantees memory-purging routines preserve production pipeline inference engines."""
    # Prepare the pipeline and train a model.
    configured_pipeline.run_test_split()
    configured_pipeline.fit_and_transform()
    configured_pipeline.run_validation_split()
    model = configured_pipeline.fit_model(DummyRegressor())

    # Clean up training data and intermediate arrays.
    configured_pipeline.clear_training_data()

    # Raw data and split state must be cleared.
    assert configured_pipeline.X is None
    assert configured_pipeline.y is None
    assert configured_pipeline._X_train is None
    assert configured_pipeline._X_test is None
    assert configured_pipeline._y_train is None
    assert configured_pipeline._y_test is None

    assert configured_pipeline._has_split is False
    assert configured_pipeline._has_transformed is False
    assert configured_pipeline._X_train_combined is None
    assert configured_pipeline._X_test_combined is None

    # The fitted transformer and trained model remain usable.
    assert configured_pipeline._is_transformer_fitted is True
    assert configured_pipeline.trained_model is model

    # Production inference should still work on fresh data.
    X_new = pd.DataFrame({
        "MedInc": [4.5],
        "HouseAge": 10,
    })

    transformed = configured_pipeline.transform_new_data(X_new)

    assert transformed.shape == (1, 2)
