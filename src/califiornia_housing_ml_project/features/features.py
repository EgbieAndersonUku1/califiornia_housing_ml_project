"""
Feature Engineering Layer for Universal ML Pipelines.

This module acts as the creative repository for domain-specific feature engineering functions.
By isolating dataset-specific math and transformations here, the core pipeline infrastructure
remains entirely decoupled, reusable, and asset-agnostic.

Available Boosters:
-------------------
1. housing_math_booster        - Standard structural ratios for residential property datasets.
2. geographic_math_booster     - Advanced spatial layout feature engineering utilising 
                                 coordinate-based cluster hub vectors.

Add as many features as needed here on this page, pass them into the AgnosticTabularPreprocessor instance,
and inject it directly into the UniversalDataPipeline.

Example usage:
--------------
>>> from california_housing_ml_project.features.features import housing_math_booster
>>> from california_housing_ml_project.utils.helpers import get_numerical_and_categorical_columns
>>> from california_housing_ml_project.data.preprocessing import AgnosticTabularAgnosticTabularPreprocessor
>>> from california_housing_ml_project.pipelines.orchestration import UniversalDataPipeline
>>>
>>> # Assume df_cleaned has been initialised and exploratory data analysis is complete.
>>> # AgnosticTabularAgnosticTabularPreprocessor takes explicit lists of numerical and categorical columns.
>>> # You can extract these manually, or utilise 'get_numerical_and_categorical_columns' 
>>> # which returns the sorted numerical and categorical column pandas indexes.
...
...
>>> numerical_columns_index, categorical_columns_index = get_numerical_and_categorical_columns(
...     df_cleaned, numerical_exclude_cols=["median_house_value"]
... )
>>>
>>> numerical_columns = numerical_columns_index.tolist()
>>> categorical_columns = categorical_columns_index.tolist()
>>>
>>> # Initialize the AgnosticTabularPreprocessor with the extracted column layout and custom booster
>>> tabular_AgnosticTabularPreprocessor = AgnosticTabularAgnosticTabularPreprocessor(
...     numerical_columns=numerical_columns, 
...     categorical_columns=categorical_columns,
...     engineering_func=housing_math_booster
... )
>>>                                                       
>>> # Isolate features and target bounds before running orchestration splits
>>> X = df_cleaned.drop(columns=["median_house_value"])
>>> y = df_cleaned["median_house_value"]
>>>
>>> # Orchestrate the split and transform routines seamlessly
>>> pipeline = UniversalDataPipeline(X=X, y=y, test_size=0.2, random_state=42)
>>> pipeline.set_transformer(tabular_AgnosticTabularPreprocessor)
>>> pipeline.run_test_split()
>>> pipeline.fit_and_transform()
>>>
>>> # Verify row counts match expectations
>>> x_train = pipeline.X_train_combined.shape[0]
>>> x_test = pipeline.X_test_combined.shape[0]
>>> split_total = sum([x_train, x_test])
"""


import pandas as pd
from sklearn.cluster import KMeans

from califiornia_housing_ml_project.settings import RANDOM_SEED
from califiornia_housing_ml_project.utils.helpers import validate_dataframe

# Global placeholder for the fitted cluster finder (hydrated dynamically via train.py or main.py)
cluster_finder = None

def init_global_cluster_finder(X_coords: pd.DataFrame):
    """
    Fits and initialises the global coordinate spatial cluster vector model.
    
 Args::
    -----------
    X_coords : pd.DataFrame
        DataFrame containing exactly two columns: ['latitude', 'longitude'].
    """
    global cluster_finder
    cluster_finder = KMeans(n_clusters=10, random_state=RANDOM_SEED, n_init=10)
    cluster_finder.fit(X_coords)


def housing_math_booster(df: pd.DataFrame) -> pd.DataFrame:
    """
    Augment imputed numerical housing features with structural ratios.

    Returns a copy of the input DataFrame with additional features.
    Undefined and non-finite engineered values are represented as NaN,
    leaving imputation to the preprocessing pipeline.
    """
    import numpy as np
    validate_dataframe(df)

    required_columns = [
        "total_rooms",
        "households",
        "total_bedrooms",
        "population",
        "housing_median_age",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Missing columns required for housing engineering: "
            f"{missing_columns}"
        )

    result = df.copy()

    # Ratios with explicit handling of zero denominators.
    result["rooms_per_household"] = (
        result["total_rooms"]
        / result["households"].replace(0, np.nan)
    )

    result["bedrooms_per_room"] = (
        result["total_bedrooms"]
        / result["total_rooms"].replace(0, np.nan)
    )

    result["population_per_household"] = (
        result["population"]
        / result["households"].replace(0, np.nan)
    )

    result["housing_median_age_per_households"] = (
        result["housing_median_age"]
        / result["households"].replace(0, np.nan)
    )

    # Convert any remaining infinite results into missing values.
    result.replace([np.inf, -np.inf], np.nan, inplace=True)

    return result


def geographic_math_booster(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies an advanced combination of base structural ratios and spatial cluster 
    hub distance vectors to capture geographical pricing variance (Achieves R² 0.8245).
    
    Args:
    -----------
    df : pd.DataFrame
        Imputed numeric features including ['latitude', 'longitude'].
    """
    validate_dataframe(df)

    # First run the standard housing ratio booster logic
    df_copy = housing_math_booster(df)
    
    # Inject spatial coordinate tracking arrays dynamically if cluster_finder is active
    if cluster_finder is not None:
        coords = df_copy[['latitude', 'longitude']]
        cluster_dist = cluster_finder.transform(coords)
        
        for i in range(10):
            df_copy[f"dist_to_geo_hub_{i}"] = cluster_dist[:, i]
            
    return df_copy

