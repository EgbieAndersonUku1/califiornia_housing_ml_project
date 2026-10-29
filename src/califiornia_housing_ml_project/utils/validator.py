from typing import Any

from pandas import DataFrame
from sklearn.base import BaseEstimator, is_classifier, is_regressor

from califiornia_housing_ml_project.errors.custom_errors import DataFrameNotLoadedError, MissingFeaturesError
from califiornia_housing_ml_project.models.static_metrics import (
    CLASSIFICATION_METRICS,
    REGRESSION_METRICS,
)

# from califiornia_housing_ml_project.utils.helpers import verbose_message

class CheckModelType:
    """Validator to ensure scikit-learn metrics match the type of estimator.
    
    Proactively checks built-in metric strings and custom make_scorer objects
    to ensure classification metrics are not sent to regression models or vice-versa.
    """

    @classmethod
    def _normalise_scoring_metrics(cls, scoring_metric: Any, estimator: BaseEstimator) -> list[str]:
        """Normalise scoring metric inputs into a list of metric names.

        Accepts a single scoring metric or a collection of metrics,
        including strings, scikit-learn scorer objects, and callable
        scoring functions. Scorer objects are validated for compatibility
        with the provided estimator before their underlying scoring
        function names are extracted.

        Args:
            scoring_metric (Any): A scoring metric or collection of scoring
                metrics. Dictionaries are normalised using their values.
            estimator (BaseEstimator): A scikit-learn model instance,
                such as LinearRegression or RandomForestRegressor, used
                to validate scoring metric compatibility.

        Returns:
            list[str]: The normalised names of the requested scoring metrics.
        """

        if isinstance(scoring_metric, dict):
            raw_payload = list(scoring_metric.values())
        elif isinstance(scoring_metric, (list, set, tuple)):
            raw_payload = list(scoring_metric)
        else:
            raw_payload = [scoring_metric]
        
        requested_metrics = []

        for metric in raw_payload:
            if isinstance(metric, str):
                requested_metrics.append(metric)
            elif hasattr(metric, "_score_func"):  # Unpacks custom make_scorer objects safely
                cls._check_scorer_capabilities(estimator, metric)
                requested_metrics.append(metric._score_func.__name__)
            elif callable(metric):                # Fallback for plain functions
                requested_metrics.append(metric.__name__)
            else:
                requested_metrics.append(str(metric))

        return requested_metrics
    
    @classmethod
    def run_check(cls, estimator: BaseEstimator, scoring_metric: Any) -> None:
        """Validate scoring metrics against the estimator type.

        Determines whether the estimator is a classifier or regressor and
        validates that the requested scoring metrics are appropriate for
        the corresponding model type.

        Args:
            estimator (BaseEstimator): A scikit-learn model instance used
                to determine whether classification or regression metrics
                are valid.
            scoring_metric (Any): A scoring metric or collection of metrics
                to validate, including strings and custom scorer objects.

        Raises:
            ValueError: If a scoring metric is incompatible with the estimator type.
        """
        requested_metrics = []

        if is_classifier(estimator):

            cls._run_check(
                requested_metrics,
                estimator,
                invalid_set=REGRESSION_METRICS,
                model_type="Classification",
            )
        elif is_regressor(estimator):
            cls._run_check(
                requested_metrics,
                estimator,
                invalid_set=CLASSIFICATION_METRICS,
                model_type="Regression",
            )

        requested_metrics = []

        if is_classifier(estimator):
            cls._run_check(requested_metrics, estimator, invalid_set=REGRESSION_METRICS, model_type="Classification")
        elif is_regressor(estimator):
            cls._run_check(requested_metrics, estimator, invalid_set=CLASSIFICATION_METRICS, model_type="Regression")

    @classmethod
    def _run_check(cls, requested_metrics: list, estimator: BaseEstimator, invalid_set: set, model_type: str) -> None:
        """Validate requested metrics against a set of banned metrics.

        Iterates over the requested metrics and raises a ValueError if any metric
        matches an entry in the invalid set. Metrics not present in the invalid set
        are ignored, and this is because the caller could be using a custom scoring
        metric

        Args:
            requested_metrics (Iterable[str]): The metrics requested by the user.
            invalid_set (Collection[str]): The set of metrics that are not permitted.
            estimator (BaseEstimator): The model being evaluated.
            model_type (str): The type of model, used to construct the error message.

        Raises:
            ValueError: If a requested metric is found in the invalid set.

        Returns:
            None
        """
        for metric in requested_metrics:
            if metric in invalid_set:
                error_msg = cls._construct_error_msg(estimator, metric, model_type)
                raise ValueError(error_msg)

    @classmethod
    def _check_scorer_capabilities(cls, estimator: BaseEstimator, scorer: any) -> None:
        """Validate that the estimator supports the prediction methods required by the scorer.

        Checks whether the provided scorer requires probability predictions and raises
        an AttributeError if the estimator does not implement ``predict_proba``.

        Args:
            estimator (BaseEstimator): The model to validate.
            scorer (Any): The scorer whose prediction method requirements are checked.

        Raises:
            AttributeError: If the scorer requires ``predict_proba`` but the estimator
                does not support it.

        Returns:
            None
        """
        response_method = getattr(scorer, "_response_method", None)
        needs_proba     = getattr(scorer, "needs_proba", False)
        
        requires_probabilities = (
            response_method == "predict_proba" or 
            (isinstance(response_method, (list, tuple)) and "predict_proba" in response_method) or
            needs_proba
        )
        
        if requires_probabilities and not hasattr(estimator, "predict_proba"):
            raise AttributeError(
                f"The provided custom scorer '{scorer._score_func.__name__}' requires probability updates "
                f" via 'predict_proba', but the model {estimator.__class__.__name__} does not support it."
            )

    @classmethod
    def _construct_error_msg(cls, estimator: BaseEstimator, metric: str, model_type: str) -> str:
        """Construct an error message describing a metric's incompatibility with a model.

        Determines the problem type for which the metric is intended based on the
        provided model type and returns a descriptive error message identifying
        the incompatible metric and estimator.

        Args:
            estimator (BaseEstimator): The model with which the metric is incompatible.
            metric (str): The name of the incompatible metric.
            model_type (str): The type of model being evaluated, either
                "Classification" or "Regression".

        Returns:
            str: A descriptive error message explaining the metric's incompatibility.
        """

        incompatible_with = "Regression" if model_type == "Classification" else "Classification"
        return (
            f"Metric element '{metric}' is intended for {incompatible_with.lower()} problems "
            f"and cannot be used with the {model_type} model {estimator.__class__.__name__}."
        )




def validate_dataframe(df: DataFrame) -> None:
    if not isinstance(df, DataFrame):
        raise TypeError(
                f"df is not an instance of DataFrame. Got type {type(df).__name__}"
            )


def validate_initial_load(df: DataFrame) -> bool:
    """
    Validates the raw structural integrity of the DataFrame immediately after import.

    Args:
        df: pd.DataFrame: The raw DataFrame to check immediately after loading.
    
    Returns:
        bool: True if the DataFrame structure is valid and populated.
    
    Raises:
        TypeError: If the input is not a pandas DataFrame object.
        DataFrameNotLoadedError: 
            - If the DataFrame is completely empty (zero rows or columns).
            - If the file parsed into accidental index or blank columns (e.g., 'Unnamed: 0').
    """
  
    
    validate_dataframe(df)    

    if df.empty or len(df.columns) == 0:
        raise DataFrameNotLoadedError("The DataFrame is empty or has no columns parsed.")
        
    if any("Unnamed" in str(col) for col in df.columns):
        raise DataFrameNotLoadedError("DataFrame contains unnamed or improperly parsed columns (e.g., 'Unnamed: 0').")
        
    return True




def is_dataframe_columns_unique(df: DataFrame) -> bool:
    """Check if all column names in the DataFrame are unique.

    This function validates the provided DataFrame structure before verifying
    the uniqueness of its column index.

    Parameters
    ----------
    df : DataFrame
        The input pandas DataFrame to evaluate.

    Returns
    -------
    bool
        True if all column names are distinct, False otherwise.

    Raises
    ------
    ValueError
        If the input DataFrame fails the `validate_dataframe` checks.
    """
    validate_dataframe(df)

    return True if df.columns.is_unique else False


def validate_missing_features_in_dataframe(df: DataFrame,  expected_features: set | list,  verbose: bool = False) -> None:
    """Verify that all expected features exist as columns in the DataFrame.

    This function validates the structural integrity of the input DataFrame 
    and checks its column names against a collection of required features.

    Parameters
    ----------
    df : DataFrame
        The input pandas DataFrame to evaluate.
    expected_features : set or list
        The collection of feature (column) names required to be present.
    verbose : bool, default False
        If True, prints a success message when validation passes.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If `expected_features` is neither a set nor a list.
    MissingFeaturesError
        If any of the `expected_features` are missing from the DataFrame columns.
    """
    validate_dataframe(df)

    if not isinstance(expected_features, (set, list)):
        raise TypeError(
            f"Expected features is not a set or list. Got type {type(expected_features).__name__}"
        )

    expected_features_set = set(expected_features)
    current_features = set(df.columns)
    missing_features = expected_features_set - current_features

    if missing_features:
        raise MissingFeaturesError(missing_features)
    

