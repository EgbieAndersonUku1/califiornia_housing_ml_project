"""Provides comprehensive unit test structures for the AgnosticTabularPreprocessor.

This test module utilises pytest execution suites to validate boundary conditions, type rules, 
and behavourial expectations across decoupled feature engineering engines. It guarantees 
seamless compatibility with the scikit-learn ecosystem pipelines.

Validation Scenarios Covered:
    - Sequence Execution Guards: Asserting that operations fail appropriately when 
      transforms are attempted out of order.
    - Component Isolation Matrix: Verifying preprocessing behavourial when given purely 
      numerical or purely categorical selections.
    - Error Handling Boundaries: Testing rejection conditions when incoming payloads 
      are missing expected configuration columns.
    - Ecosystem Compliance: Ensuring seamless support for scikit-learn operations like 
      `clone()` and functional integration within a formal `Pipeline`.
"""


import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline, clone
from sklearn.preprocessing import StandardScaler

from califiornia_housing_ml_project.data.preprocessing import AgnosticTabularPreprocessor



@pytest.fixture
def sample_data():
    """Generates a standard reference dataset containing missing and string values."""
    
    return pd.DataFrame({
        "MedInc": [3.5, 5.0, np.nan, 2.0, 6.0],
        "HouseAge": [20, 35, 15, 40, 25],
        "OceanProximity": [
            "INLAND", "NEAR BAY", "INLAND", None, "NEAR OCEAN"
        ],
    })


def test_transform_before_fit_raises_error(sample_data):
    """Ensures that a transform call drops out with an error if fit has not run."""

    transformer = AgnosticTabularPreprocessor(
        numerical_columns=["MedInc", "HouseAge"],
        categorical_columns=["OceanProximity"],
    )

    with pytest.raises(RuntimeError, match="fitted"):
        transformer.transform(sample_data)


def test_fit_returns_self(sample_data):
    """Verifies that the fit routine updates internal flags and returns self."""

    transformer = AgnosticTabularPreprocessor(
        numerical_columns=["MedInc", "HouseAge"],
        categorical_columns=["OceanProximity"],
    )

    result = transformer.fit(sample_data)

    assert result is transformer
    assert transformer._is_fitted is True


def test_transform_returns_expected_shape(sample_data):
    """Validates that transformed outputs match input row lengths and mapped columns."""

    transformer = AgnosticTabularPreprocessor(
        numerical_columns=["MedInc", "HouseAge"],
        categorical_columns=["OceanProximity"],
    )

    transformer.fit(sample_data)
    result = transformer.transform(sample_data)

    assert isinstance(result, np.ndarray)
    assert result.shape[0] == len(sample_data)
    assert result.shape[1] == len(
        transformer.get_feature_names_out()
    )


def test_unknown_categories_are_supported(sample_data):
    """Guarantees that unseen categories are handled smoothly without crashing."""

    transformer = AgnosticTabularPreprocessor(
        numerical_columns=["MedInc", "HouseAge"],
        categorical_columns=["OceanProximity"],
    )

    transformer.fit(sample_data)

    new_data = pd.DataFrame({
        "MedInc": [4.2],
        "HouseAge": [30],
        "OceanProximity": ["UNKNOWN_CATEGORY"],
    })

    result = transformer.transform(new_data)

    assert result.shape[0] == 1
    assert result.shape[1] == len(
        transformer.get_feature_names_out()
    )


def test_numerical_only_preprocessing(sample_data):
    """Checks that preprocessing isolates numerical targets when no categories exist."""

    transformer = AgnosticTabularPreprocessor(
        numerical_columns=["MedInc", "HouseAge"],
    )

    transformer.fit(sample_data)
    result = transformer.transform(sample_data)

    assert isinstance(result, np.ndarray)
    assert result.shape == (len(sample_data), 2)


def test_categorical_only_preprocessing(sample_data):
    """Checks that preprocessing isolates categorical columns when no numerical features exist."""

    transformer = AgnosticTabularPreprocessor(
        categorical_columns=["OceanProximity"],
    )

    transformer.fit(sample_data)
    result = transformer.transform(sample_data)

    assert isinstance(result, np.ndarray)
    assert result.shape[0] == len(sample_data)



def test_fit_rejects_missing_required_columns(sample_data):
    """Ensures fit routine drops out if configuration vectors name a missing key."""

    transformer = AgnosticTabularPreprocessor(
        numerical_columns=["MedInc", "MissingFeature"],
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        transformer.fit(sample_data)


def test_transform_rejects_missing_required_columns(sample_data):
    """Ensures transform drops out if evaluated dataset features are dropped downstream."""

    transformer = AgnosticTabularPreprocessor(
        numerical_columns=["MedInc", "HouseAge"],
    )
    transformer.fit(sample_data)

    incomplete_data = sample_data.drop(columns=["HouseAge"])

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        transformer.transform(incomplete_data)



def test_preprocessor_supports_sklearn_clone():
    """Confirms deep cloning behaves correctly while scrubbing fitted flags."""

    transformer = AgnosticTabularPreprocessor(
        numerical_columns=["MedInc", "HouseAge"],
        categorical_columns=["OceanProximity"],
    )

    cloned_transformer = clone(transformer)

    assert isinstance(cloned_transformer, AgnosticTabularPreprocessor)
    assert cloned_transformer is not transformer
    assert cloned_transformer.numerical_columns == ["MedInc", "HouseAge"]
    assert cloned_transformer.categorical_columns == ["OceanProximity"]
    assert cloned_transformer._is_fitted is False


def test_preprocessor_works_in_sklearn_pipeline(sample_data):
    """Validates that the preprocessor runs successfully within scikit-learn Pipelines."""

    transformer = AgnosticTabularPreprocessor(
        numerical_columns=["MedInc", "HouseAge"],
        categorical_columns=["OceanProximity"],
    )

    pipeline = Pipeline([
        ("preprocessor", transformer),
        ("scaler", StandardScaler()),
    ])

    transformed = pipeline.fit_transform(sample_data)

    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == len(sample_data)
    assert np.isfinite(transformed).all()


def test_feature_names_match_transformed_output(sample_data):
    """Verifies that generated feature tracking name arrays map to the matrix width."""

    transformer = AgnosticTabularPreprocessor(
        numerical_columns=["MedInc", "HouseAge"],
        categorical_columns=["OceanProximity"],
    )

    transformed = transformer.fit_transform(sample_data)
    feature_names = transformer.get_feature_names_out()

    assert transformed.shape[1] == len(feature_names)
    assert len(feature_names) == len(set(feature_names))
