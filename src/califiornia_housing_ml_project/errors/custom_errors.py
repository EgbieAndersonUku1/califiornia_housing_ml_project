"""Domain-specific exception classes for the machine learning architecture pipeline.

This module acts as the centralised repository for all custom structural, validation, 
and runtime errors across the project layout. By isolating custom exceptions here, 
the codebase maintains strict type and value safety guardrails and produces clear, 
actionable stack traces during multi-dimensional data engineering or training failures.

Error Hierarchies Covered:
    - Tabular Data & Matrix Validation: Structural layout issues, duplicate columns, 
      or uninitialised dataframes.
    - Configuration & Pipeline Integration: Component mismatch boundaries, such as 
      failing to link the `AgnosticTabularPreprocessor` before pipeline execution.
    - Architectural Construction: Parameter, activation, hyperparameter, and layer 
      shape failures during model builder setup.
    - Compilation & Optimisation: Invalid configuration of loss functions, metrics, 
      or tracking optimisers.
    - Runtime Lifecycle Hooks: Ordering exceptions raised when operations violate the 
      strict execution sequence required by the builder or evaluation frameworks.

Note:
    Duplicate validations or overlapping definitions (such as `InvalidDatasetError`) 
    should be resolved within the file structure based on whether they inherit from 
    `TypeError` or `ValueError`.
"""


class DataFrameNotLoadedError(Exception):
    """A DataFrame Exception error when the Dataframe is not laoded"""

class DataValidationError(TypeError):
    """Raised when data matrices fail structural type checks."""

class AgnosticTabularPreprocessorError(Exception):
    """Raises an error of AgnosticTabularPreprocessor class hasn't be added to UniversalDataPipeline class"""


class NonUniqueColumnsError(ValueError):
    """Exception raised when a DataFrame contains duplicate column names.

    This error indicates that the tabular data structure violates the
    requirement of having strictly unique identifiers for every column.
    """

class InvalidConfigurationError(TypeError):
    """Raised when a configuration parameter receives an incorrect type."""
   

class InvalidActivationError(ValueError):
    """Raised when an unsupported or invalid activation function is requested."""


class InvalidDatasetError(TypeError):
    """Raised when the input training or validation datasets are not of the correct type."""
    
 
class InvalidPipelineError(TypeError):
    """Raised when an object is not a valid UniversalDataPipeline instance."""


class InvalidDatasetError(ValueError):
    """Raised when datasets fail type, shape, or alignment validation checks."""


class InvalidModelError(TypeError):
    """Raised when an object is not a valid Keras Model instance."""


class InvalidHyperparameterError(ValueError):
    """Raised when a hyperparameter (like epochs or batch size) has an invalid value."""


class InvalidCallbackError(TypeError):
    """Raised when an invalid object or list of objects is passed as a callback."""


class InvalidKerasVerboseOptionsError(TypeError):
    """Raises when an invalid verbose option is used as a parameter"""

class InvalidLossError(TypeError):
    """Raised when an unrecognized loss function component is assigned."""
    

class InvalidOptimizerError(TypeError):
    """Raised when an unapproved optimization function strategy is assigned."""


class InvalidMetricsError(TypeError):
    """Raised when tracking metrics fail framework compatibility checks."""

class ModelLifecycleError(RuntimeError):
    """Raised when a model building step is invoked out of sequential order."""


class ArchitectureNotFoundError(RuntimeError):
    """Raised when operations require an initialized architecture that does not exist."""
   

class MissingConfigurationError(RuntimeError):
    """Raised when vital hyperparameters are missing during initialization stages."""

class ModelNotFoundError(RuntimeError):
    """Raised when an operation requires an instantiated model object that is missing."""
    
class InvalidArchitectureError(TypeError):
    """Raised when model layer shapes or dimensions receive an incorrect type."""
    

class ModelNotCompiledError(Exception):
    """Raised when a Keras model has not been compiled before training."""


class InvalidModelHistoryError(TypeError):
    """Raised when model_history is not a Keras History instance."""


class IncorrectTransformerClassError(Exception):
    """
    Exception raised when an object fails to match the required 
    structural interface layout expected for data transformers.
    """
    from typing import Any

    def __init__(self, invalid_obj: Any, expected_type: str = "Transformer"):
        """
        Initialises the error with details about the invalid object class.

        Args:
            invalid_obj (Any): The instance that failed the type check.
            expected_type (str): The name of the interface blueprint expected.
        """
        actual_type_name = invalid_obj.__class__.__name__
        message = (
            f"Expected an instance of '{expected_type}', "
            f"but received an instance of '{actual_type_name}' instead."
        )
        super().__init__(message)


class MissingFeaturesError(ValueError):
    """Exception raised when required features are missing from a DataFrame.

    Args:
    
        missing_features (set): The collection of feature names that were expected but not found.
    """
    def __init__(self, missing_features: set):
        self.missing_features = missing_features
        super().__init__(f"Validation Failed: Missing required features: {missing_features}")




class ColumnOrderMismatchError(ValueError):
    """
    Exception raised when actual DataFrame columns do not match
    the expected column schema.

    This exception is raised when the actual column names, composition,
    or ordering differ from the expected schema. It can be used to
    enforce a consistent feature structure during data preprocessing,
    feature engineering, model fitting, and transformation.

    Attributes:
        expected (list[str]):
            The expected sequence of column names defining the
            required schema and column order.

        actual (list[str]):
            The actual sequence of column names received or produced
            during validation.

    Args:
        expected (list[str]):
            The expected column names in their required order.

        actual (list[str]):
            The actual column names in the order they were received.

    Raises:
        ValueError:
            This exception inherits from ValueError and is raised
            when the actual column structure does not match the
            expected structure.

    Example:
        >>> expected = ["MedInc", "HouseAge"]
        >>> actual = ["HouseAge", "MedInc"]
        >>> raise ColumnOrderMismatchError(
        ...     expected=expected,
        ...     actual=actual,
        ... )
        ColumnOrderMismatchError: Structure mismatch.
        Expected: ['MedInc', 'HouseAge']
        Got: ['HouseAge', 'MedInc']
    """

    def __init__(self, expected: list[str], actual: list[str]) -> None:
        """
        Initialise the exception with the expected and actual schemas.

        Args:
            expected (list[str]):
                The expected column names and their required order.

            actual (list[str]):
                The actual column names and their received order.
        """
        self.expected = expected
        self.actual = actual

        super().__init__(f"Structure mismatch.\n" f"Expected: {expected}\n" f"Got: {actual}")
