"""Provides performance benchmarking capabilities for compiled deep learning models.

The BenchMarkTrainedModels class acts as a performance evaluation matrix designed to aggregate, 
compare, and rank multiple trained Keras architectures. By consuming historical training logs 
and validating predictions against unseen feature sets, it creates a centralised, unified framework 
for identifying the most robust network layouts.

Workflow Integration & Sequencing Guardrails:
    This class operates downstream from your `ModelBuilder` or execution loops. It collects live 
    Keras models along with their corresponding training metadata wrappers.

    Strict functional ordering must be maintained to avoid structural layout faults or missing keys:
    1. `add_model_info()` must be called one or more times to register your model information pairs.
    2. `summarise_model_training_results()` must be invoked to initialize the base summary metrics 
       and establish the underlying results layout. Calling later metrics pipelines before this step 
       will result in an assignment error.
    3. `add_validation_rmse()` can then be executed to run active forward passes across data 
       matrices, calculating raw error scores and appending them to the existing summary dataset.

The benchmarking framework handles:
    - Verifying architectural types and ensuring framework-specific callback histories are valid.
    - Parsing historical epochs to automatically isolate optimal validation thresholds.
    - Executing automated batch predictions to calculate root-mean-squared errors.
    - Formatting performance dimensions into sorted, scannable data layouts.
"""

from typing import Self

from pandas import DataFrame, Series
from numpy import ndarray
from numpy import sqrt as np_sqrt
from keras.callbacks import History
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from califiornia_housing_ml_project.errors.custom_errors import InvalidModelError, DataValidationError, InvalidModelHistoryError


class BenchMarkTrainedModels:
    """Aggregates and benchmarks training and validation performance metrics across multiple models.

    This class serves as an evaluation registry that gathers Keras models, extracts epoch metrics 
    from their structural histories, and scores validation features to build a comprehensive 
    benchmarking report.

    Attributes:
        _histories: A list storing registered model profiles, where each profile is a tuple 
            containing the model name, its Keras History callback instance, and the raw model object.
        _result_df: A summary DataFrame tracking collected metrics across all registered models.
    """

    def __init__(self):
        self._histories = []
        self._result_df  = None

    def add_model_info(self, model_history: History, model_name : str, model: Keras.Model) -> None:
        """Registers a trained model's performance history and structural instance.

        Appends the model profile components to the internal tracking collection 
        after verifying that all arguments satisfy type requirements and framework 
        compatibility constraints.

        Args:
            model_history (History): The historical training logs callback instance 
                emitted by the Keras framework during training.
            model_name (str): A unique identifying string label assigned to this model.
            model (keras.Model): The active instantiated Keras neural network model 
                object associated with the training logs.

        Raises:
            InvalidModelHistoryError: If `model_history` is not an instance of the 
                Keras History callback wrapper class.
            TypeError: If `model_name` is not a string primitive type.
            InvalidModelError: If `model` is not a verified instance of a Keras model 
                architecture interface.
        """
        if not isinstance(model_history, History):
            raise InvalidModelHistoryError(
                "model_history must be an instance of keras.callbacks.History, "
                f"got {type(model_history).__name__}"
            )

        if not isinstance(model_name, str):
            raise TypeError(
                f"The model_name is not a string. Expected string got {type(model_name).__name__}"
            )

        if not isinstance(model, keras.Model):
            raise InvalidModelError(
                f"The model is not an instance of Keras. Got type {type(model).__name__}"
            )
        self._histories.append((model_name, model_history, model))

    def summarise_model_training_results(self):
        """Summarises the historical training performance across all registered profiles.

        Parses the internal history dictionary logs to extract epoch metrics, identify the 
        exact training cycle with the lowest validation loss, and map the corresponding 
        validation mean absolute error (MAE) score into a comprehensive results layout.

        Returns:
            Self: The instance of the benchmarking class with the summary DataFrame 
                initialised and cached in the `_result_df` attribute.
        """
        results = []

        for model_name, history, _ in self._histories:
            history_df = DataFrame(history.history)
            best_epoch_index = history_df["val_loss"].idxmin()

            results.append({
                "model": model_name,
                "epochs_completed": len(history_df),
                "best_epoch": best_epoch_index + 1,
                "validation_mae": float(
                    history_df.loc[best_epoch_index, "val_mae"]
                ),
            })

        self._result_df = DataFrame(results)
        return self

    def add_validation_rmse(self, X_val_prepared: ndarray, y_val: Series) -> Self:
        """Calculates validation RMSE for each model and appends it to the summary layout.

        Executes silent batch forward passes across the preprocessed features for every 
        registered model, flattens the resulting prediction arrays, and computes the 
        Root Mean Squared Error (RMSE) against the ground-truth targets before updating 
        the internal summary tracking container.

        Args:
            X_val_prepared (ndarray): A multi-dimensional NumPy array containing the 
                preprocessed validation feature split matrices.
            y_val (Series): A Pandas Series containing the unscaled ground-truth 
                validation target values.

        Returns:
            Self: The instance of the benchmarking class with an additional 
                'validation_rmse' metric column appended to the cached results DataFrame.

        Raises:
            DataValidationError: If `X_val_prepared` is not a NumPy ndarray, or if 
                `y_val` is not a Pandas Series instance.
        """
        if not isinstance(X_val_prepared, ndarray):
            raise DataValidationError(
                f"Validation features 'X_val_prepared' must be a NumPy ndarray, "
                f"but received type '{type(X_val_prepared).__name__}'."
            )

        if not isinstance(y_val, Series):
            raise DataValidationError(
                f"Validation targets 'y_val' must be a Pandas Series, "
                f"but received type '{type(y_val).__name__}'."
            )

        rmse_values = []

        for _, _, model in self._histories:
            y_val_pred = model.predict(
                X_val_prepared, verbose=0
            ).ravel()

            rmse = np_sqrt(mean_squared_error(y_val, y_val_pred))
            rmse_values.append(float(rmse))

        self._result_df["validation_rmse"] = rmse_values
        return self
        
    def display_trained_model_results(self) -> DataFrame:
        """Retrieves the aggregated performance summary tracking DataFrame.

        Returns:
            DataFrame: A Pandas DataFrame compiling the comparative benchmarking 
                metrics across all processed models.
        """
        return self._result_df
