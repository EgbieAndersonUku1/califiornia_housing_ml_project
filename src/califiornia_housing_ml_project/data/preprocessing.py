"""Provides an agnostic tabular data preprocessor for machine learning pipelines.

The AgnosticTabularPreprocessor class acts as the centralised preprocessing engine 
responsible for structuring, encoding, scaling, and engineering tabular datasets. By remaining 
entirely agnostic to the underlying dataset domain, it shifts specialised mathematical and 
feature engineering tasks to external functional injections (Boosters).

Functional Relationship:
    This class works directly with custom domain feature boosters (such as `housing_math_booster` 
    or `geographic_math_booster`) and passes its processed states straight into the 
    `UniversalDataPipeline` orchestration layer.

The preprocessor handles:
    - Separating and tracking distinct numerical and categorical column arrays.
    - Consuming an optional `engineering_func` booster to inject domain-specific math 
      without hardcoding transformations into the class structure.
    - Implementing strict fit and transform pipelines for feature scaling and encoding.
    - Managing decoupled target variables through independent target scaling and transformation hooks.

Example:
    >>> # Extract column classifications cleanly using helpers
    >>> num_cols, cat_cols = get_numerical_and_categorical_columns(df, exclude_cols=["target"])
    >>>
    >>> # Instantiate the preprocessor with an injected domain-specific mathematical booster
    >>> preprocessor = AgnosticTabularPreprocessor(
    ...     numerical_columns=num_cols.tolist(),
    ...     categorical_columns=cat_cols.tolist(),
    ...     engineering_func=housing_math_booster
    ... )
    >>>
    >>> # Pass directly to the orchestration container
    >>> pipeline = UniversalDataPipeline(X=X, y=y)
    >>> pipeline.set_transformer(preprocessor)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from numpy import inf, nan, array, hstack, ndarray
from pandas import DataFrame, to_numeric
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from califiornia_housing_ml_project.utils.validator import validate_dataframe, is_dataframe_columns_unique, validate_missing_features_in_dataframe
from califiornia_housing_ml_project.errors.custom_errors import ColumnOrderMismatchError, NonUniqueColumnsError



class AgnosticTabularPreprocessor(BaseEstimator, TransformerMixin):
    """Preprocess numerical and categorical features in tabular data.

    Impute missing numerical values using the median, standardize numerical
    features, and encode categorical features using one-hot encoding.
    Optionally apply a custom feature engineering function before
    preprocessing.

    Validate input DataFrames and feature columns to ensure that required
    columns are present and engineered features conform to the expected
    schema. Track the fitted state and engineered feature names to support
    consistent transformations of subsequent datasets.

    Unseen categorical values encountered during transformation are ignored
    by the one-hot encoder.

    Attributes:
        numerical_columns (list[str] | None): Names of the numerical
            columns to preprocess.
        categorical_columns (list[str] | None): Names of the categorical
            columns to preprocess.
        engineering_func (Callable[[DataFrame], DataFrame] | None):
            Optional function for engineering features before
            preprocessing.
    """

    def __init__(
        self,
        numerical_columns: list[str] | None = None,
        categorical_columns: list[str] | None = None,
        engineering_func: Callable[[DataFrame], DataFrame] | None = None,
    ):
        """Initialize the preprocessor with the specified feature columns.

        Args:
            numerical_columns (list[str] | None): Names of the numerical
                columns to preprocess. Defaults to None.
            categorical_columns (list[str] | None): Names of the categorical
                columns to preprocess. Defaults to None.
            engineering_func (Callable[[DataFrame], DataFrame] | None):
                Optional function for engineering features before
                preprocessing.

        Raises:
            TypeError: If the numerical or categorical column arguments
                have invalid types.
        """
        self._validate_columns(numerical_columns=numerical_columns, 
                               categorical_columns=categorical_columns
                               )

        self.numerical_columns         = numerical_columns
        self.categorical_columns       = categorical_columns
        self.engineering_func         = engineering_func
        self._engineered_feature_names = []
        self._numerical_feature_names  = []

        self._imputer   = SimpleImputer(strategy="median")
        self._scaler    = StandardScaler()
        self._encoder   = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self._is_fitted = False

    def _validate_input_columns(self, X: DataFrame) -> None:
        """Validate the input DataFrame and its required feature columns.

        Check that all configured numerical and categorical columns are
        present in the input DataFrame.

        Args:
            X (DataFrame): The input feature data.

        Raises:
            ValueError: If any required feature columns are missing.
        """

        # validate_dataframe(X)
        required_columns = (
            (self.numerical_columns or [])
            + (self.categorical_columns or [])
        )

        missing_columns = [
            column for column in required_columns
            if column not in X.columns
        ]

        if missing_columns:
            raise ValueError(
                "Input DataFrame is missing required columns: "
                f"{missing_columns}"
            )
    
    def _validate_columns(
        self,
        numerical_columns: list[str],
        categorical_columns: list[str],
    ) -> None:
        """Validate the configured numerical and categorical column names.

        Check that both arguments are lists of strings or None, and that
        at least one numerical or categorical column is provided.

        Args:
            numerical_columns (list[str]): Names of the numerical columns.
            categorical_columns (list[str]): Names of the categorical
                columns.

        Raises:
            TypeError: If either argument is not a list or None, or if
                any column name is not a string.
            ValueError: If no numerical or categorical columns are provided.
        """
        if numerical_columns is not None and not isinstance(
            numerical_columns, list
        ):
            raise TypeError(
                "numerical_columns must be a list of strings or None. "
                f"Got {type(numerical_columns).__name__}."
            )

        if categorical_columns is not None and not isinstance(
            categorical_columns, list
        ):
            raise TypeError(
                "categorical_columns must be a list of strings or None. "
                f"Got {type(categorical_columns).__name__}."
            )

        numerical_columns = numerical_columns or []
        categorical_columns = categorical_columns or []

        if not numerical_columns and not categorical_columns:
            raise ValueError(
                "At least one numerical or categorical column must be provided."
            )

        if any(not isinstance(col, str) for col in numerical_columns):
            raise TypeError(
                "Every numerical column name must be a string."
            )

        if any(not isinstance(col, str) for col in categorical_columns):
            raise TypeError(
                "Every categorical column name must be a string."
            )
        
    def _validate_column_order(
        self,
        expected_columns: list[str],
        actual_columns: list[str],
    ) -> None:
        """Validate that column names and their order match the expected schema.

        Args:
            expected_columns (list[str]): The expected ordered column names.
            actual_columns (list[str]): The actual ordered column names
                to validate, such as those obtained from DataFrame.columns.

        Raises:
            ColumnOrderMismatchError: If the actual column names or their
                order differ from the expected schema.
        """
        if actual_columns != expected_columns:
            raise ColumnOrderMismatchError(
                expected=expected_columns,
                actual=actual_columns,
            )
        
    def _validate_engineering_output(
        self,
        X: DataFrame,
        original_X: DataFrame,
        fitting: bool = False,
    ) -> DataFrame:
        """
        Validate the output produced by the feature engineering function.

        Ensures that the engineered DataFrame is valid, contains unique
        column names, preserves the original row index and order, and
        contains only numerical features.

        During fitting, records the engineered feature names and their
        order for subsequent validation. During transformation, verifies
        that the engineered output matches the schema established during
        fitting.

        Args:
            X (DataFrame):
                DataFrame returned by the feature engineering function.

            original_X (DataFrame):
                Original numerical DataFrame supplied to the feature
                engineering function. Used to verify that the row index
                and order have been preserved.

            fitting (bool, optional):
                If True, records the engineered output schema for later
                use during transformation. If False, validates the output
                against the schema recorded during fitting.
                Defaults to False.

        Returns:
            DataFrame:
                The validated engineered DataFrame, unchanged.

        Raises:
            TypeError:
                If the engineering function does not return a pandas
                DataFrame or produces non-numerical columns.

            ValueError:
                If the engineering function changes the original row
                index or row order.

            NonUniqueColumnsError:
                If the engineered DataFrame contains duplicate column
                names.

            ColumnOrderMismatchError:
                If the engineered column names or their order differ
                from the schema recorded during fitting.
        """
       
        validate_dataframe(X)

        if not X.index.equals(original_X.index):
            raise ValueError(
                "The engineering function must preserve the original "
                "row index and row order."
            )

        if not is_dataframe_columns_unique(X):
            raise NonUniqueColumnsError(
                "The engineered DataFrame columns must be unique."
            )

        non_numeric_columns = (
            X.select_dtypes(exclude=["number"]).columns.tolist()
        )

        if non_numeric_columns:
            raise TypeError(
                "The engineering function produced non-numeric columns: "
                f"{non_numeric_columns}"
            )

        output_columns = X.columns.tolist()

        if fitting:
            self._numerical_feature_names = output_columns.copy()

        elif output_columns != self._numerical_feature_names:
            raise ColumnOrderMismatchError(
                expected=self._numerical_feature_names,
                actual=output_columns,
            )

        return X

    def _apply_engineering(
        self,
        X: DataFrame,
        fitting: bool = False,
    ) -> DataFrame:
        """Apply feature engineering to the numerical input columns.

        Validate that the required numerical columns are present, match
        the expected order, and have unique names. If no feature
        engineering function is configured, return a copy of the input
        DataFrame. Otherwise, apply the configured function to a copy
        of the input and validate the resulting DataFrame.

        Args:
            X (DataFrame): The input DataFrame containing the numerical
                features to process.
            fitting (bool): Whether the feature engineering output is
                being validated during fitting. Passed to the output
                validation method to determine whether to record or
                verify the engineered feature schema. Defaults to False.

        Returns:
            DataFrame: A copy of the input DataFrame if no engineering
                function is configured, or the validated engineered
                DataFrame otherwise.

        Raises:
            TypeError: If the input or engineered output is not a valid
                DataFrame.
            ValueError: If required numerical columns are missing or
                their order does not match the expected schema, or if
                the engineering function changes the row index or order.
            NonUniqueColumnsError: If the input or engineered DataFrame
                contains duplicate column names.
            ColumnOrderMismatchError: If the input columns do not match
                the expected numerical column order, or if the engineered
                output differs from the schema established during fitting.
        """
        validate_missing_features_in_dataframe(X, self.numerical_columns)
        self._validate_column_order(
            expected_columns=self.numerical_columns,
            actual_columns=X.columns.to_list()
        )
        if not is_dataframe_columns_unique(X):
                    raise NonUniqueColumnsError(
                        "The engineered DataFrame columns must be unique."
                    )

        if self.engineering_func is None:
            return X.copy()

        original_X = X.copy()
        engineered_X = self.engineering_func(X.copy())

        return self._validate_engineering_output(
            X=engineered_X,
            original_X=original_X,
            fitting=fitting,
        )
    
    def fit(self, X: DataFrame, y: Any | None = None) -> AgnosticTabularPreprocessor:
        """Fit the preprocessing components to the input feature data.

        Validate the input columns, then fit the numerical and categorical
        preprocessing components independently.

        For numerical features, convert values to numeric form, impute
        missing values, apply optional feature engineering, replace
        infinite values with missing values, impute again, and fit the
        standard scaler on the resulting features.

        For categorical features, replace missing values with the string
        "missing", convert values to strings, and fit the one-hot encoder.

        Args:
            X (DataFrame): The training feature data used to fit the
                preprocessing components.
            y (Any | None): Ignored. Included for compatibility with the
                scikit-learn estimator API. Defaults to None.

        Returns:
            AgnosticTabularPreprocessor: The fitted transformer instance.

        Raises:
            TypeError: If the input is not a valid DataFrame or contains
                invalid column configuration or data types.
            ValueError: If required columns are missing or the input
                columns do not match the expected schema.
        """

        self._validate_input_columns(X)

        numerical_columns = self.numerical_columns or []
        categorical_columns = self.categorical_columns or []

        self._is_fitted = False
        self._engineered_feature_names = []
        self._numerical_feature_names = []

        # Numerical preprocessing
        if numerical_columns:
            X_num = X[numerical_columns].apply(
                to_numeric,
                errors="coerce",
            )

            # First imputation: original numerical features
            self._imputer.fit(X_num)

            X_num_imputed = DataFrame(
                self._imputer.transform(X_num),
                columns=numerical_columns,
                index=X.index,
            )

            # Apply and validate feature engineering
            X_num_engineered = self._apply_engineering(
                X_num_imputed,
                fitting=True,
            )

            self._engineered_feature_names = [
                col for col in X_num_engineered.columns
                if col not in numerical_columns
            ]

            self._numerical_feature_names = (
                X_num_engineered.columns.tolist()
            )

            # Normalize infinite values created by engineering
            X_num_engineered = X_num_engineered.replace([inf, -inf], nan)

            # Second imputation: values created by feature engineering
            self._engineered_imputer = SimpleImputer(strategy="median")

            X_num_engineered_imputed = DataFrame(
                self._engineered_imputer.fit_transform(
                    X_num_engineered
                ),
                columns=self._numerical_feature_names,
                index=X.index,
            )

            # Fit scaler on the final numerical features
            self._scaler.fit(X_num_engineered_imputed)


        # Categorical preprocessing
        if categorical_columns:
            X_cat = X[categorical_columns].fillna("missing").astype(str)
            self._encoder.fit(X_cat)

        self._is_fitted = True
        return self
        
    def transform(self, X: DataFrame) -> ndarray:
        """Transform input data using the fitted preprocessing components.

        Validate the input columns and verify that the transformer has
        been fitted. Apply the fitted numerical imputer, optional feature
        engineering function, post-engineering imputer, and scaler to
        numerical features. Impute missing categorical values with the
        string "missing" and encode categorical features using the fitted
        one-hot encoder.

        Combine the transformed numerical and categorical features into
        a single feature matrix.

        Args:
            X (DataFrame): The input feature data to transform.

        Returns:
            ndarray: The combined numerical and categorical feature matrix.

        Raises:
            RuntimeError: If the transformer has not been fitted.
            TypeError: If the input is not a valid DataFrame or contains
                invalid data types.
            ValueError: If required columns are missing, the input schema
                is invalid, or no numerical or categorical features are
                configured.
            NonUniqueColumnsError: If the input or engineered DataFrame
                contains duplicate column names.
            ColumnOrderMismatchError: If the input or engineered columns
                do not match the expected schema.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Transformer instance must be fitted before running transformations."
            )

        self._validate_input_columns(X)

        numerical_columns = self.numerical_columns or []
        categorical_columns = self.categorical_columns or []

        transformed_blocks = []

        # Numerical preprocessing
        if numerical_columns:
            X_num = X[numerical_columns].apply(
                to_numeric,
                errors="coerce",
            )

            # Apply the original fitted imputer
            X_num_imputed = DataFrame(
                self._imputer.transform(X_num),
                columns=numerical_columns,
                index=X.index,
            )

            # Apply and validate feature engineering
            X_num_engineered = self._apply_engineering(
                X_num_imputed,
                fitting=False,
            )

            # Normalize infinities generated by feature engineering
            X_num_engineered = X_num_engineered.replace([inf, -inf], nan)

            # Apply the fitted post-engineering imputer
            X_num_engineered_imputed = DataFrame(
                self._engineered_imputer.transform(X_num_engineered),
                columns=self._numerical_feature_names,
                index=X.index,
            )

            # Apply the fitted scaler
            X_scaled = self._scaler.transform(X_num_engineered_imputed)

            transformed_blocks.append(X_scaled)

        # Categorical preprocessing
        if categorical_columns:
            X_cat = X[categorical_columns].fillna("missing").astype(str)

            X_encoded = self._encoder.transform(X_cat)
            transformed_blocks.append(X_encoded)

        # Combine numerical and categorical features
        if not transformed_blocks:
            raise ValueError(
                "No numerical or categorical features are configured."
            )

        if len(transformed_blocks) == 1:
            return transformed_blocks[0]

        return hstack(transformed_blocks)
    
    def get_feature_names_out(self, input_features=None) -> ndarray:
        """Return the output feature names generated by the fitted transformer.

        Combine the names of the numerical features, including engineered
        features, with the names generated by the one-hot encoder for
        categorical features.

        Args:
            input_features (array-like of str, optional): Ignored.
                Included for compatibility with the scikit-learn
                transformer API.

        Returns:
            ndarray: The names of the transformed numerical and
                categorical features.

        Raises:
            RuntimeError: If the transformer has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Transformer must be fitted before calling get_feature_names_out(...).")

        categorical_columns = self.categorical_columns or []

        all_numeric = self._numerical_feature_names

        if categorical_columns:
            all_categorical = list(self._encoder.get_feature_names_out(categorical_columns))
            return array(all_numeric + all_categorical)

        return array(all_numeric)

    def fit_transform_target(self, y: Any) -> ndarray:
        """Fit the target scaler and transform the target values.

        Reshape the target values into a two-dimensional array, fit a
        StandardScaler, and return the scaled values as a one-dimensional
        array.

        Args:
            y (Any): The target values used to fit the scaler and
                generate the transformed target array.

        Returns:
            ndarray: The scaled target values as a one-dimensional array.
        """

        import numpy as np
        from sklearn.preprocessing import StandardScaler

        if hasattr(y, "values"):
            y_arr = y.values.reshape(-1, 1)
        else:
            y_arr = np.asarray(y).reshape(-1, 1)

        # Create the scaler if needed, then fit it.
        self._target_scaler = StandardScaler()

        return self._target_scaler.fit_transform(y_arr).ravel()


    def transform_target(self, y: Any) -> ndarray:
        """Transform target values using the fitted target scaler.

        Reshape the target values into a two-dimensional array, apply the
        previously fitted StandardScaler, and return the scaled values
        as a one-dimensional array.

        Args:
            y (Any): The target values to transform.

        Returns:
            ndarray: The scaled target values as a one-dimensional array.

        Raises:
            NotFittedError: If the target scaler has not been fitted.
        """
        import numpy as np
        from sklearn.preprocessing import StandardScaler

        # Safeguard: Automatically build it if it doesn't exist yet
        if not hasattr(self, '_target_scaler'):
            self._target_scaler = StandardScaler()

        if hasattr(y, "values"):
            y_arr = y.values.reshape(-1, 1)
        else:
            y_arr = np.array(y).reshape(-1, 1)

        return self._target_scaler.transform(y_arr).flatten()

    def inverse_transform_target(self, y_scaled: Any) -> ndarray:
        """Convert scaled target values back to their original scale.

        Reshape the scaled target values into a two-dimensional array,
        apply the inverse transformation using the fitted target scaler,
        and return the resulting values as a one-dimensional array.

        Args:
            y_scaled (Any): The scaled target values to inverse transform.

        Returns:
            ndarray: The target values restored to their original scale.

        Raises:
            RuntimeError: If the preprocessor has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before target variables can be inverse transformed.")

        if hasattr(y_scaled, "values"):
            y_arr = y_scaled.values.reshape(-1, 1)
        else:
            y_arr = array(y_scaled).reshape(-1, 1)

        return self._target_scaler.inverse_transform(y_arr).flatten()