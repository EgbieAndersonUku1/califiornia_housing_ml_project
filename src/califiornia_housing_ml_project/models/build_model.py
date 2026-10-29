
"""Provides a secure builder interface for constructing and compiling deep learning models.

The ModelBuilder class implements a fluent builder pattern designed to safely construct, 
configure, and compile Keras sequential neural networks. The page isolates structural layer definitions, 
weight modifications, and optimiser settings from the core data orchestration blocks, this module 
acts as a specialised, decoupled layer blueprint.

Workflow Integration & Sequencing Guardrails:
    This class works closely with specialised configuration enums (such as `AllowedOptimizer`, 
    `AllowedLoss`, and `AllowedMetrics`) to prevent incorrect values from reaching the deep neural 
    layers. The built and compiled Keras network instances are designed to be consumed directly 
    by downstream evaluation modules such as `PipelineModelEvaluator`.

    Strict sequential construction must be observed to avoid runtime validation drops:
    1. `create_architecture()` must be called first to initialise the dense hidden layers, 
       apply L2 regularisation boundaries, and establish internal shape alignments.
    2. `add_optimizers()` must follow to safely register the training loss parameters, metrics, 
                         add a list of neurons e.g [126, 68, 32] and the underlying system
                         will automatically create three layers along with a output layer.
                         The number of layers created depends on number of neurons in the
                         list
       and mapping frameworks with strict gradient norm clipping thresholds.
    3. `build_and_compile()` can then be safely invoked to compile the underlying sequential model.
    4

The builder handles:
    - Enforcing rigorous datatype checks on shapes, activation enums, and dropout limits.
    - Injecting automatic gradient norm clipping (`clipnorm`) to prevent numerical runtime explosions.
    - Decoupling neural configuration vectors via localised tracking classes (`NeuralNetworkConfig`).
    - Standardising sequential state monitoring via flags (`_is_model_created`, `is_optimizer_params_added`).
"""


from __future__ import annotations

import math
import keras
from keras import layers
from enum import Enum
from typing_extensions import Literal
from typing import Any, Self
from pydantic import BaseModel, field_validator

from califiornia_housing_ml_project.errors.custom_errors import (
                                                                 ArchitectureNotFoundError,
                                                                 InvalidArchitectureError,
                                                                 MissingConfigurationError,
                                                                 InvalidConfigurationError, 
                                                                 InvalidActivationError, 
                                                                 InvalidLossError,
                                                                 InvalidMetricsError,
                                                                 InvalidOptimizerError,
                                                                 ModelNotFoundError,
                                                            
                                                                 )



# Permitted loss function string constants for model compilation.
ALLOWED_LOSSES = Literal[
    "mean_squared_error", "mse", "mean_absolute_error", "mae",
    "binary_crossentropy", "categorical_crossentropy", "sparse_categorical_crossentropy"
]


# Permitted optimizer string constants for the optimization pipeline.
ALLOWED_OPTIMIZERS = Literal["adam", "sgd", "rmsprop"]



# Permitted evaluation metric string constants for tracking training performance.
ALLOWED_METRICS = Literal[
    # For Regression (MSE, MAE)
    "mean_squared_error", "mse", 
    "mean_absolute_error", "mae", 
    "root_mean_squared_error", 
    "rmse",
    
    # For Binary Classification
    "binary_accuracy",
     "accuracy",
     "auc", 
     "precision",
     "recall",
    
    # For Multi-class Classification (Categorical & Sparse)
    "categorical_accuracy",
     "sparse_categorical_accuracy"
]




class AllowedMetrics(str, Enum):
    """Supported evaluation metrics for model training.

    Each member represents a standard evaluation metric option that can be passed 
    to a Keras model's compile method to monitor performance during training 
    and testing.

    Attributes:
        ACCURACY: Generic accuracy tracking shorthand string.
        ACC: Alternative generic accuracy shorthand string.
        MEAN_SQUARED_ERROR: Mean Squared Error (MSE) regression metric full string.
        MSE: Mean Squared Error (MSE) regression metric shorthand string.
        MEAN_ABSOLUTE_ERROR: Mean Absolute Error (MAE) regression metric full string.
        MAE: Mean Absolute Error (MAE) regression metric shorthand string.
        ROOT_MEAN_SQUARED_ERROR: Root Mean Squared Error (RMSE) regression metric full string.
        RMSE: Root Mean Squared Error (RMSE) regression metric shorthand string.
        MEAN_ABSOLUTE_PERCENTAGE_ERROR: Mean Absolute Percentage Error (MAPE) regression metric full string.
        MAPE: Mean Absolute Percentage Error (MAPE) regression metric shorthand string.
        BINARY_ACCURACY: Calculates the mean accuracy rate for binary classification problems.
        BINARY_CROSSENTROPY: Computes the cross-entropy metric between true labels and predictions.
        AUC: Area Under the Receiver Operating Characteristic (ROC) curve metric.
        PRECISION: Computes the precision score of the predictions relative to the labels.
        RECALL: Computes the recall score of the predictions relative to the labels.
        CATEGORICAL_ACCURACY: Calculates the mean accuracy rate for multi-class classification problems.
        SPARSE_CATEGORICAL_ACCURACY: Calculates multi-class accuracy when target labels are integers.
        CATEGORICAL_CROSSENTROPY: Computes categorical cross-entropy metric for one-hot encoded targets.
        SPARSE_CATEGORICAL_CROSSENTROPY: Computes categorical cross-entropy metric for integer targets.
    """
    # Generic / Context-Aware
    ACCURACY                       = "accuracy"
    ACC                            = "acc"

    # Regression Metrics
    MEAN_SQUARED_ERROR             = "mean_squared_error"
    MSE                            = "mse"
    MEAN_ABSOLUTE_ERROR            = "mean_absolute_error"
    MAE                            = "mae"
    ROOT_MEAN_SQUARED_ERROR        = "root_mean_squared_error"
    RMSE                           = "rmse"
    MEAN_ABSOLUTE_PERCENTAGE_ERROR = "mean_absolute_percentage_error"
    MAPE                           = "mape"

    # Binary Classification Metrics
    BINARY_ACCURACY                = "binary_accuracy"
    BINARY_CROSSENTROPY            = "binary_crossentropy"
    AUC                            = "auc"
    PRECISION                      = "precision"
    RECALL                         = "recall"

    # Multi-class Classification Metrics
    CATEGORICAL_ACCURACY           = "categorical_accuracy"
    SPARSE_CATEGORICAL_ACCURACY    = "sparse_categorical_accuracy"
    CATEGORICAL_CROSSENTROPY       = "categorical_crossentropy"
    SPARSE_CATEGORICAL_CROSSENTROPY = "sparse_categorical_crossentropy"



class AllowedLoss(str, Enum):
    """Supported loss functions for model training.

    Each member represents a structural loss criterion option that can be 
    mapped to its corresponding Keras loss value during the model compilation 
    and optimization pipeline setup.

    Attributes:
        BINARY_CROSSENTROPY: Binary cross-entropy loss for binary classification tasks.
        CATEGORICAL_CROSSENTROPY: Categorical cross-entropy loss for multi-class classification tasks.
        HUBER: Huber loss function, less sensitive to outliers than MSE.
        LOG_COSH: Logarithm of the hyperbolic cosine of the prediction error.
        MAE: Mean Absolute Error (MAE) loss function shorthand string.
        MAPE: Mean Absolute Percentage Error (MAPE) loss function shorthand string.
        MEAN_ABSOLUTE_ERROR: Mean Absolute Error (MAE) loss function full string.
        MEAN_ABSOLUTE_PERCENTAGE_ERROR: Mean Absolute Percentage Error (MAPE) loss function full string.
        MEAN_SQUARED_ERROR: Mean Squared Error (MSE) loss function full string.
        MSE: Mean Squared Error (MSE) loss function shorthand string.
        SPARSE_CATEGORICAL_CROSSENTROPY: Sparse categorical cross-entropy loss for integer-coded multi-class tasks.
    """

    BINARY_CROSSENTROPY             = "binary_crossentropy"
    CATEGORICAL_CROSSENTROPY        = "categorical_crossentropy"
    HUBER                           = "huber"
    LOG_COSH                        = "log_cosh"
    MAE                             = "mae"
    MAPE                            = "mape"
    MEAN_ABSOLUTE_ERROR             = "mean_absolute_error"
    MEAN_ABSOLUTE_PERCENTAGE_ERROR  = "mean_absolute_percentage_error"
    MEAN_SQUARED_ERROR              = "mean_squared_error"
    MSE                             = "mse"
    SPARSE_CATEGORICAL_CROSSENTROPY = "sparse_categorical_crossentropy"



class AllowedOptimizer(str, Enum):
    """Supported optimizers for model training.

    Each member represents a structural optimizer option that can be mapped 
    to its corresponding instantiated Keras optimizer object during the 
    compilation process.

    Attributes:
        ADAM: Represents the Adaptive Moment Estimation (Adam) optimizer.
        SGD: Represents the Stochastic Gradient Descent (SGD) optimizer.
        RMSPROP: Represents the Root Mean Square Propagation (RMSprop) optimizer.
    """

    ADAM    = "adam"
    SGD     = "sgd"
    RMSPROP = "rmsprop"


class Activations(str, Enum):
    """Enumeration of supported neural network activation functions.

    Attributes:
        RELU (str): Rectified Linear Unit activation.
        TANH (str): Hyperbolic Tangent activation.
        SIGMOID (str): Sigmoid activation.
        SOFTMAX (str): Softmax activation.
        LINEAR (str): Linear (identity) activation.
        SELU (str): Scaled Exponential Linear Unit activation.
        ELU (str): Exponential Linear Unit activation.
    """
    RELU     = "relu"
    TANH     = "tanh"
    SIGMOID  = "sigmoid"
    SOFTMAX  = "softmax"
    LINEAR   = "linear"
    SELU     = "selu"
    ELU      = "elu"
    



class NeuralNetworkConfig(BaseModel):
    """Configuration for a neural network's layer architecture.

    Validates the number of neurons assigned to each layer, ensuring
    all neuron counts are positive integers.

    Attributes:
        neurons: A list of positive integers representing the number
            of neurons in each layer, in order.
    """

    neurons: list[int]
   
    @field_validator("neurons")
    @classmethod
    def validate_neurons(cls, values):
        """Validate that all neuron counts are positive.

        Args:
            values: The list of neuron counts to validate.

        Returns:
            The validated list of neuron counts.

        Raises:
            ValueError: If any neuron count is zero or negative.
        """
        if any(n <= 0 for n in values):
            raise ValueError("Neuron counts must be positive")
        return values


class ModelBuilder:
    """Create and train a neural network using validated configuration.

    This class implements a builder pattern to safely construct, configure, 
    and compile a Keras sequential neural network model with strict guardrails 
    against numerical and architectural errors.

    Attributes:
        _model: The internal Keras Sequential model instance reference.
        _loss: The mapped loss function value used for training.
        _optimizer: The instantiated Keras optimizer object with configured 
            learning rate and gradient clipping bounds.
        _metrics: A list of evaluation metrics to track during model execution.
        is_optimizer_params_added: True if the optimizer settings have been 
            successfully registered.
        _is_model_created: True if the structural layers and network architecture 
            have been successfully instantiated.
    """

    def __init__(self):
        self._model                    = None
        self._loss                     = None
        self._optimizer                = None
        self._metrics                  = None
        self.is_optimizer_params_added = False
        self._is_model_created         = False

    @classmethod
    def _validate_model_config(cls, add_kernel_regularizer, activation: Activations):
        """Validates the model configuration parameters.

        Args:
            add_kernel_regularizer: A boolean indicating whether to add a kernel 
                regulariser.
            activation (Activations): An instance of the Activations Enum class 
                to use for the model.

        Raises:
            InvalidConfigurationError: If `add_kernel_regularizer` is not a boolean.
            InvalidActivationError: If `activation` is not an instance of the 
                Activations Enum.
        """
        if not isinstance(add_kernel_regularizer, bool):
            raise InvalidConfigurationError(
                f"Parameter 'add_kernel_regularizer' must be a boolean (True/False), "
                f"but received type '{type(add_kernel_regularizer).__name__}'."
            )
        
        if not isinstance(activation, Activations):
            raise InvalidActivationError(
                f"activation is not an instance of Activation Enum class. "
                f"Expected Activation Enum but got type {type(activation).__name__}"
                )

        
    def create_architecture(
                self,
                input_shape: int,
                neurons: list[int],
                activation: Activations = Activations.RELU,
                add_kernel_regularizer: bool = False,
                kernel_regularizer_value: float | None = 0.01,
                use_batch_normalization: bool = False,
                dropout_rate: float = 0.0,
                ensure_positive_output: bool = False
                 
        ) -> Self:
        """Build a neural network architecture safely without gradient explosion.

        Args:
            input_shape: The size of the input layer. Must be a positive integer.

            neurons: A list containing the number of hidden units for each dense layer.
                     This is need to create the Dense layers for the model

            activation: The activation function Enum to apply to the hidden layers. 
                         Defaults to Activations.RELU.

            add_kernel_regularizer: Whether to apply L2 regularisation to the weights. 
                Defaults to False.

            kernel_regularizer_value: The penalty factor for the regulariser. 
                Must be positive if regularisation is enabled. Defaults to 0.01.

            use_batch_normalization: Whether to inject a batch normalisation layer 
                after each hidden dense layer. Defaults to False.

            dropout_rate: The fraction of neurons to drop during training. Must 
                fall within the range [0, 1). Defaults to 0.0.

            ensure_positive_output: If True, forces a ReLU activation on the final 
                regression layer. Defaults to False.

        Returns:
            Self: The instance of the builder pattern class with the created model.

        Raises:
            InvalidConfigurationError: Passed up if `add_kernel_regularizer` or 
                `activation` fail the initial configuration check.

            InvalidActivationError: Passed up if the provided activation does not 
                match the expected enum.

            InvalidArchitectureError: If any structural parameters (shape, neurons, 
                regulariser boundaries, or dropout ranges) fail type and value validation.
        
        Example usage:
        >>> model = (ModelBuilder()
                     .create_architecture(neurons=[128, 64, 32], # 3 hidden layers + output layer will be created
                                input_shape=X_val.shape[1],
                                use_batch_normalization=True,  
                                dropout_rate=0.3,             
                                ensure_positive_output=False)
                    .add_optimizers(learning_rate=0.002) 
                    .build_and_compile())

        """

        self._validate_model_config(add_kernel_regularizer, activation)
        config = NeuralNetworkConfig(neurons=neurons)
        
        if (isinstance(input_shape, bool)
            or not isinstance(input_shape, int)
            or input_shape < 1
            ):
            raise InvalidArchitectureError(
                f"The 'input_shape' argument must be a positive integer, "
                f"but received {type(input_shape).__name__}."
            )
        
        regularizer = None

        if kernel_regularizer_value is not None:
            if isinstance(kernel_regularizer_value, bool):
                raise InvalidArchitectureError(
                    "The regularisation value cannot be a boolean."
                )
            if not isinstance(kernel_regularizer_value, (int, float)):
                raise InvalidArchitectureError(
                    "The regularisation value must be an int or float."
                )
            if not math.isfinite(kernel_regularizer_value):
                raise InvalidArchitectureError("The regularisation value must be finite.")
            
            if kernel_regularizer_value < 0.0:
                raise InvalidArchitectureError(
                        "The regularisation value must be non-negative."
                    )
            
        if add_kernel_regularizer:
            if kernel_regularizer_value is None:
                raise InvalidArchitectureError(
                    "A regularisation value is required when regularisation is enabled."
                )
            if isinstance(kernel_regularizer_value, bool):
                raise InvalidArchitectureError(
                    "The regularisation value cannot be a boolean."
                )
            if not isinstance(kernel_regularizer_value, (int, float)):
                raise InvalidArchitectureError(
                    "The regularisation value must be an int or float."
                )
            if not math.isfinite(kernel_regularizer_value):
                raise InvalidArchitectureError("The regularisation value must be finite.")
            if kernel_regularizer_value <= 0.0: 
                raise InvalidArchitectureError(
                    "The regularisation value must be greater than 0.0."
                )
                
            regularizer = keras.regularizers.l2(kernel_regularizer_value)


        if not isinstance(use_batch_normalization, bool):
            raise InvalidArchitectureError(
                  "The 'use_batch_normalization' argument must be a boolean, "
                    f"but received {type(use_batch_normalization).__name__}."
            )
       
        if isinstance(dropout_rate, bool) or not isinstance(dropout_rate, (int, float)):
            raise InvalidArchitectureError("The dropout rate must be a number, not a boolean.")

        if not math.isfinite(dropout_rate):
            raise InvalidArchitectureError("The dropout rate must be finite.")

        if not (0 <= dropout_rate < 1):
            raise InvalidArchitectureError("The dropout rate must be at least 0 and less than 1.")

        if not isinstance(ensure_positive_output, bool):
            raise InvalidArchitectureError(
                f"The 'ensure_positive_output' argument must be a boolean, "
                f"but received {type(ensure_positive_output).__name__}."
            )
        
        model = keras.Sequential()
        model.add(layers.Input(shape=(input_shape,)))

        # Build hidden layers with internal normalization guards
        for _, neuron in enumerate(config.neurons):
            model.add(layers.Dense(
                units=neuron, 
                activation=activation.value,
                kernel_regularizer=regularizer
            ))
            
            # Optionally add Batch Normalisation and Dropout after each hidden layer.
            if use_batch_normalization:
                model.add(layers.BatchNormalization())

            if dropout_rate > 0:
                model.add(layers.Dropout(rate=dropout_rate))

        # Add the custom final single-neuron output layer for regression
        output_activation = Activations.RELU.value if ensure_positive_output else None
       
        model.add(layers.Dense(1, activation=output_activation))


        self._is_model_created = True
        self._model            = model
        return self

    def add_optimizers(self, 
                       loss: ALLOWED_LOSSES = AllowedLoss.MSE, 
                       optimizer: ALLOWED_OPTIMIZERS = AllowedOptimizer.ADAM,
                       metrics: ALLOWED_METRICS = AllowedMetrics.MAE,
                       learning_rate: float =  0.001,
                       clipnorm: float = 1.0,
                        ) -> Self:
        """Configures the training loss, optimizer, and metrics for the model.

        Maps the configuration enums to their corresponding Keras classes and 
        instantiates the optimizer with strict gradient clipping parameters to 
        prevent numerical explosions in deep layers.

        Args:
            loss: The loss function configuration enum to use during training. 
                Defaults to AllowedLoss.MSE.

            optimizer: The training optimizer enum to map and instantiate. 
                Defaults to AllowedOptimizer.ADAM.

            metrics: The evaluation metric criteria enum to track. Defaults to 
                AllowedMetrics.MAE.

            learning_rate: The step size for the optimizer weight updates. 
                Defaults to 0.001.

            clipnorm: The maximum gradient norm threshold allowed before clipping. 
                Acts as a guardrail against gradient explosions. Defaults to 1.0.

        Returns:
            Self: The instance of the builder pattern class with optimizer 
                parameters configured.

        Raises:
            InvalidLossError: If the provided loss is not an instance of AllowedLoss.
            InvalidOptimizerError: If the optimizer does not match the 
                AllowedOptimizer interface.
            InvalidMetricsError: If the metrics object does not meet AllowedMetrics 
                criteria.
        """

        if not isinstance(loss, AllowedLoss):
            raise InvalidLossError(
                f"Invalid loss configuration. Expected type subclass of AllowedLoss, "
                f"but received variant object: '{type(loss).__name__}'."
            )

        if not isinstance(optimizer, AllowedOptimizer):
            raise InvalidOptimizerError(
                f"Invalid training optimizer. Must match AllowedOptimizer interface, "
                f"but received variant object: '{type(optimizer).__name__}'."
            )

        if not isinstance(metrics, AllowedMetrics):
            raise InvalidMetricsError(
                f"Invalid evaluation metric array. Must match AllowedMetrics criteria, "
                f"but received variant object: '{type(metrics).__name__}'."
            )

        # Keep the loss value as a string/object standard
        self._loss = loss.value

        # Map Enums to actual instantiated Keras optimizers with a clipnorm safety guardrail
        # This stops the 11.7+ / 12.4+ RMSE explosions in the deep networks
        
        optimizer_mapping = {
            AllowedOptimizer.ADAM: keras.optimizers.Adam,
            AllowedOptimizer.SGD: keras.optimizers.SGD,
            AllowedOptimizer.RMSPROP: keras.optimizers.RMSprop,
        }

        optimizer_class = optimizer_mapping[optimizer]

        self._optimizer = optimizer_class(
            learning_rate=learning_rate,
            clipnorm=clipnorm,
        )

        # Handle metrics gracefully (Keras accepts a list of metric strings or objects)
        self._metrics = [metrics.value]

        self.is_optimizer_params_added = True

        return self

    def build_and_compile(self, show_run_ouput: bool = False) -> Any:
        """Compiles the tracked Keras model instance using the configured settings.

        Verifies that the architecture generation has completed and that all 
        required optimization vectors are set up before calling the compile method 
        on the underlying sequential framework.

        Args:
            show_run_ouput: If True, prints or displays information regarding 
                the model summary or run pipeline state. Defaults to False.

        Returns:
            Any: The compiled Keras model instance or the pipeline tracking wrapper.

        Raises:
            ArchitectureNotFoundError: If the model creation sequence has not been 
                triggered or completed.

            MissingConfigurationError: If the optimizer, loss, or metric parameters 
                have not been registered.

            ModelNotFoundError: If the underlying model instance reference is broken 
                or unexpectedly set to None.
        """
        if not self._is_model_created:
            raise ArchitectureNotFoundError(
                "Compilation aborted: No active model instance found. "
                "Ensure the model generation sequence has completed successfully."
            )

        if not self.is_optimizer_params_added:
            raise MissingConfigurationError(
                "Compilation aborted: Missing required optimization settings parameter vectors."
            )

        if self._model is None:
            raise ModelNotFoundError(
                "Compilation failed: The underlying model instance tracking reference is empty (None). "
                "Ensure 'create_architecture()' was called and initialized the internal model storage container successfully."
            )

        if not self._is_model_created:
            raise ArchitectureNotFoundError(
                "Compilation aborted: No active model instance found. "
                "Ensure the model generation sequence has completed successfully."
            )

        if not self.is_optimizer_params_added:
            raise MissingConfigurationError(
                "Compilation aborted: Missing required optimization settings parameter vectors."
            )

        if self._model is None:
            raise ModelNotFoundError(
                "Compilation failed: The underlying model instance tracking reference is empty (None). "
                "Ensure 'create_architecture()' was called and initialized the internal model storage container successfully."
            )

        self._model.compile(
             
            optimizer=self._optimizer,
            loss=self._loss,
            metrics=self._metrics,
        )
        
        return self._model




