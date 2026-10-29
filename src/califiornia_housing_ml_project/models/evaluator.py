from typing import Any, Self

from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

from califiornia_housing_ml_project.data.data_pipeline import UniversalDataPipeline
from califiornia_housing_ml_project.settings import N_JOBS, RANDOM_SEED
from califiornia_housing_ml_project.utils.validator import CheckModelType


class PipelineModelEvaluator:
    """
    Train, validate, and evaluate scikit-learn regression estimators.

    This class works with a UniversalDataPipeline to manage model
    training and performance evaluation without duplicating data
    preprocessing or splitting responsibilities.

    The evaluator supports three stages of model assessment:

    1. Training and validation:
       Clone and fit the supplied estimator on the prepared training
       split, generate predictions, and calculate regression metrics
       for both training and validation datasets.

    2. Cross-validation:
       Construct a scikit-learn Pipeline containing a fresh clone of
       the configured preprocessor and estimator. Perform K-fold
       cross-validation on the full raw training pool so that
       preprocessing is fitted independently within each fold.

    3. Final evaluation:
       After model selection, fit a fresh preprocessing-and-model
       pipeline on the complete raw training pool. Evaluate its
       predictions against the previously held-out test dataset.

    The evaluator does not modify the original estimator supplied
    during initialization. Instead, it uses scikit-learn cloning
    to create independent estimator instances.

    The UniversalDataPipeline must have a configured transformer,
    completed its train-test split, and prepared its training and
    validation data before the evaluator is used.

    This class is intended for scikit-learn-compatible regression
    estimators. It does not build or compile Keras neural networks.

    Args:
        model_instance (BaseEstimator):
            An unfitted scikit-learn-compatible regression estimator
            supporting fit() and predict().

        pipeline (UniversalDataPipeline):
            A prepared data pipeline containing the configured
            transformer, training and validation splits, and the
            held-out test dataset.

    Attributes:
        pipeline (UniversalDataPipeline):
            Data pipeline used to access the prepared datasets
            and preprocessing transformer.

    Notes:
        - Training and validation metrics include R², root mean
          squared error (RMSE), and mean absolute error (MAE).
        - Cross-validation uses raw training data to prevent
          preprocessing leakage between folds.
        - The held-out test dataset should only be evaluated after
          model selection and hyperparameter tuning are complete.
        - Final model training refits preprocessing on the full
          training pool rather than reusing preprocessing fitted
          on the smaller training split.
        - Calling the final evaluation methods is not required
          when comparing candidate models.

    Example:
        >>> from sklearn.linear_model import LinearRegression
        >>> from califiornia_housing_ml_project.models.evaluator import (
        ...     PipelineModelEvaluator,
        ... )
        >>>
        >>> # Assume data_pipeline has already been configured,
        >>> # split, and transformed.
        >>> evaluator = PipelineModelEvaluator(
        ...     model_instance=LinearRegression(),
        ...     pipeline=data_pipeline,
        ... )
        >>>
        >>> # Train and evaluate on training/validation data.
        >>> evaluator.fit()
        >>> evaluator.predict()
        >>> evaluator.evaluate()
        >>> validation_results = evaluator.get_results()
        >>>
        >>> # Optionally perform cross-validation.
        >>> evaluator.cross_validation()
        >>>
        >>> # Only after selecting the final model:
        >>> evaluator.fit_final_model()
        >>> evaluator.predict_final_test()
        >>> evaluator.evaluate_final_test()
        >>> final_results = evaluator.get_final_test_results()

    Warning:
        Repeatedly evaluating different candidate models against
        the held-out test dataset can introduce selection bias.
        Use training, validation, and cross-validation results
        for model selection, reserving the test dataset for the
        final assessment.
    """

    def __init__(self, model_instance: BaseEstimator, pipeline: UniversalDataPipeline):
        """Initializes the evaluator with an architecture instance and data pipeline.

        Args:
            model_instance: The target model definition or builder reference to 
                be evaluated.
            pipeline: An instantiated data processing pipeline containing prepared 
                feature arrays.
        """
        self._validate_fields(model_instance, pipeline)
        self._model_instance         = model_instance
        self.pipeline                = pipeline
        self._is_fitted              = False
        self._model                  = None
        self._model_train_prediction = None
        self._model_val_prediction  = None

        # Training metrics
        self._model_train_r2_score   = None
        self._model_train_rmse       = None
        self._model_train_mae        = None

        # Validation metrics
        self._model_val_r2_score    = None
        self._model_val_rmse        = None
        self._model_val_mae         = None

        # Cross-validation
        self._model_cv               = None
        # Final model and held-out test evaluation
        self._final_model            = None
        self._final_test_prediction  = None

        self._final_test_r2_score    = None
        self._final_test_rmse        = None
        self._final_test_mae         = None

    def _validate_fields(self, model_instance: Any, pipeline: UniversalDataPipeline,) -> None:
        """Validates that the provided fields meet the required types and interfaces.

        Args:
            model_instance: The model object to check. It must implement standard 
                scikit-learn or Keras API methods.
            pipeline: The data pipeline object to check.

        Raises:
            TypeError: If `pipeline` is not an instance of `UniversalDataPipeline`, 
                or if `model_instance` lacks a callable `fit` or `predict` method.
        """
        if not isinstance(pipeline, UniversalDataPipeline):
            raise TypeError(
                "pipeline must be an instance of "
                "UniversalDataPipeline. "
                f"Got {type(pipeline).__name__}."
            )

        if not isinstance(model_instance, BaseEstimator):
            raise TypeError(
                "Model instance is not a base instance of BaseEstimator"
            )

        if not callable(getattr(model_instance, "fit", None)):
            raise TypeError(
                "model_instance must provide a callable fit() method."
            )

        if not callable(getattr(model_instance, "predict", None)):
            raise TypeError(
                "model_instance must provide a callable predict() method."
            )

    def _check_is_fitted(self) -> None:
        """Verifies if the model training state is active before evaluation.

        Raises:
            RuntimeError: If `_is_fitted` is False, indicating the underlying 
                model has not been trained yet.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "The model has not been fitted. Call fit() first."
            )

    def _check_train_predictions_exist(self) -> None:
        """Verifies if the training target prediction arrays are fully cached.

        Raises:
            RuntimeError: If `_model_train_prediction` is None, indicating 
                that predictions have not been generated for the training set.
        """
        if self._model_train_prediction is None:
            raise RuntimeError(
                "Training predictions are unavailable. Call predict() first."
            )

    def _check_val_predictions_exist(self) -> None:
        """
        Verifies if the validation target prediction arrays are fully cached.
        
        Raises:
            RuntimeError: If `_model_val_prediction` is None, indicating 
                that predictions have not been generated for the validation set.
        """
        if self._model_val_prediction is None:
            raise RuntimeError(
                "Validation predictions are unavailable. Call predict() first."
            )

    def _reset_evaluation_results(self) -> None:
        """Resets all cached prediction vectors and calculated metric boundaries.

        Clears internal storage fields for training and validation outputs to prevent
        stale evaluations across consecutive fit or scoring pipelines.
        """
        self._model_train_prediction = None
        self._model_val_prediction  = None
        self._model_train_r2_score   = None
        self._model_train_rmse       = None
        self._model_train_mae        = None
        self._model_val_r2_score    = None
        self._model_val_rmse        = None
        self._model_val_mae         = None

    @property
    def model_name(self) -> str:
        """Retrieves the class name of the underlying model instance.

        Returns:
            str: The string representation of the model's class name.
        """
        return self._model_instance.__class__.__name__

    def fit(self):
        """Fits a cloned copy of the model instance on the training partition.

        Clears previous run states, isolates the architecture from external 
        mutations by cloning it, and executes the fit routing against the 
        pre-processed features from the data pipeline.

        Returns:
            Self: The instance of the evaluator class with the fitted model.
        """

        self._is_fitted = False
        self._model     = None
        self._model_cv  = None

        self._reset_evaluation_results()

        self._model = clone(self._model_instance).fit(
            self.pipeline.X_train_split,
            self.pipeline.y_train_split,
        )

        self._is_fitted = True
        return self

    def predict(self) -> Self:
        """
        Generates target predictions for both training and validation datasets.
        Executes the fitted model's prediction pipeline across the pre-processed
        feature matrices and invalidates any previously calculated performance metrics
        to prevent stale evaluation evaluations.

        Returns:
            Self: The instance of the evaluator class with cached prediction arrays.

        Raises:
            RuntimeError: If the model has not been successfully fitted yet.
        """
        self._check_is_fitted()

        self._model_train_prediction = self._model.predict(
            self.pipeline.X_train_split
        )

        self._model_val_prediction = self._model.predict(
            self.pipeline.X_val_split
        )

        # Predictions have changed, so previous metrics are invalid.
        self._model_train_r2_score = None
        self._model_train_rmse     = None
        self._model_train_mae      = None

        self._model_val_r2_score  = None
        self._model_val_rmse      = None
        self._model_val_mae       = None

        return self
    
    def train_r2_score(self) -> Self:
        """Calculates the R-squared score on the training data partition.

        Evaluates the coefficient of determination for the training predictions 
        against the true training targets and caches the result internally.

        Returns:
            Self: The instance of the evaluator class with the updated training 
                R-squared metric.

        Raises:
            RuntimeError: If training predictions do not exist.
        """
        self._check_train_predictions_exist()

        self._model_train_r2_score = r2_score(
             self.pipeline.y_train_split,
             self._model_train_prediction,
        )

        return self

    def train_rmse(self) -> Self:
        """Calculates the Root Mean Squared Error (RMSE) on the training partition.

        Evaluates the root mean squared deviation between the cached training predictions 
        and the ground-truth training targets, storing the result internally.

        Returns:
            Self: The instance of the evaluator class with the updated training 
                RMSE metric.

        Raises:
            RuntimeError: If training predictions do not exist.
        """
        self._check_train_predictions_exist()

        self._model_train_rmse = root_mean_squared_error(
            self.pipeline.inverse_transform(self.pipeline.y_train_split),
            self.pipeline.inverse_transform(self._model_train_prediction),
        )

        return self

    def train_mae(self) -> Self:
        """Calculates the Mean Absolute Error (MAE) on the training partition.

        Evaluates the average absolute differences between the cached training predictions 
        and the ground-truth training targets, storing the result internally.

        Returns:
            Self: The instance of the evaluator class with the updated training 
                MAE metric.

        Raises:
            RuntimeError: If training predictions do not exist.
        """
        self._check_train_predictions_exist()

        self._model_train_mae = mean_absolute_error(
            self.pipeline.inverse_transform(self.pipeline.y_train_split),
            self.pipeline.inverse_transform(self._model_train_prediction),
        )

        return self

    def val_r2_score(self) -> Self:
        """Calculates the R-squared score on the validation data partition.

        Evaluates the coefficient of determination for the validation predictions
        against the true validation targets and caches the result internally.

        Returns:
            Self: The instance of the evaluator class with the updated validation
            R-squared metric.

        Raises:
            RuntimeError: If validation predictions do not exist
        """
        self. _check_val_predictions_exist()

        self._model_val_r2_score = r2_score(
            self.pipeline.y_val_split,
            self._model_val_prediction,
        )

        return self

    def val_rmse(self) -> Self:
        """Calculates the Root Mean Squared Error (RMSE) on the validation partitionn.

        Evaluates the root mean squared deviation between the cached validation predictions
        and the ground-truth validation targets, storing the result internally.

        Returns:
            Self: The instance of the evaluator class with the updated validation
             RMSE metric.

        Raises:
            RuntimeError: If validation predictions do not exist.
        """
        self. _check_val_predictions_exist()

        self._model_val_rmse = root_mean_squared_error(
            self.pipeline.inverse_transform(self.pipeline.y_val_split),
            self.pipeline.inverse_transform(self._model_val_prediction),
        )

        return self

    def val_mae(self) -> Self:
        """Calculates the Mean Absolute Error (MAE) on the validation partition.

        Evaluates the average absolute differences between the cached validation predictions
        and the ground-truth validation targets, storing the result internally.

        Returns:
            Self: The instance of the evaluator class with the updated validation
            MAE metric.

        Raises:
           RuntimeError: If validation predictions do not exist.
        """
        self. _check_val_predictions_exist()

        self._model_val_mae = mean_absolute_error(
            self.pipeline.inverse_transform(self.pipeline.y_val_split),
            self.pipeline.inverse_transform(self._model_val_prediction),
        )

        return self

    def evaluate(self) -> Self:
        """Executes the complete evaluation pipeline for the fitted model.

        Generates training and validation predictions, calculates all core performance
        metrics (R-squared, RMSE, and MAE) across both data partitions, and 
        caches the results internally.

        Returns:
            Self: The instance of the evaluator class with all performance 
                metrics populated.

        Raises:
            RuntimeError: If the underlying model has not been successfully 
                fitted before evaluation.
        """
        self._check_is_fitted()

        self.predict()

        self.train_r2_score()
        self.train_rmse()
        self.train_mae()

        self.val_r2_score()
        self.val_rmse()
        self.val_mae()

        return self

    def get_results(self) -> dict[str, Any]:
        """Retrieves a nested dictionary of all compiled performance metrics.

        Extracts cached scoring values for both the training and validation
        partitions.

        Returns:
            dict[str, Any]: A dictionary containing structural validation results 
                partitioned by dataset split:
                
                - "train": Nested dictionary containing "r2", "rmse", and "mae".
                - "validation": Nested dictionary containing "r2", "rmse", and "mae".

        Raises:
            RuntimeError: If the model has not been fitted, indicating metrics 
                are uncalculated or stale.
        """

        self._check_is_fitted()

        return {
            "train": {
                "r2": self._model_train_r2_score,
                "rmse": self._model_train_rmse,
                "mae": self._model_train_mae,
            },
            "validation": {
                "r2": self._model_val_r2_score,
                "rmse": self._model_val_rmse,
                "mae": self._model_val_mae,
            },
        }

    @property
    def cv_results(self) -> Any:
        """Retrieves the cached cross-validation performance scores.

        Returns:
            Any: The stored cross-validation results array or dictionary payload.

        Raises:
            ValueError: If cross-validation has not been executed yet, leaving 
                the internal tracking reference uninitialized.
        """
        if self._model_cv is None:
            raise ValueError("No CV results found run cross_validation() first")
        return self._model_cv

    def cross_validation(
        self,
        scoring_metric: dict | str | None = None,
        n_jobs: int = N_JOBS,
        n_split: int = 5,
    ):
        """Executes cross-validation over the training dataset split.

        Constructs an isolated evaluation pipeline containing a cloned copy of 
        both the pipeline's data transformer and the raw model architecture, 
        then evaluates generalization performance using a shuffled K-Fold split.

        Args:
            scoring_metric: A string or dictionary strategy specification for 
                scikit-learn metric evaluation. Defaults to None.
            n_jobs: The number of CPU cores to run cross-validation jobs in parallel. 
                Defaults to N_JOBS.
            n_split: The number of K-Fold cross-validation folds. Must be at 
                least 2. Defaults to 5.

        Returns:
            Self: The instance of the evaluator class with populated cross-validation 
                results cached in the `_model_cv` attribute.

        Raises:
            TypeError: If `n_jobs` or `n_split` are not integers (or are booleans).
            ValueError: If `n_split` is configured to be less than 2.
            Exception: Passes through any type or structural execution exceptions 
                raised by the underlying `CheckModelType.run_check` routine.
        """

        if not isinstance(n_jobs, int) or isinstance(n_jobs, bool):
            raise TypeError(
                f"n_jobs must be an int. Got {type(n_jobs).__name__}."
            )

        if not isinstance(n_split, int) or isinstance(n_split, bool):
            raise TypeError(
                f"n_split must be an int. Got {type(n_split).__name__}."
            )

        if n_split < 2:
            raise ValueError("n_split must be at least 2.")

        CheckModelType.run_check(
            self._model_instance,
            scoring_metric=scoring_metric,
        )

        cv = KFold(
            n_splits=n_split,
            shuffle=True,
            random_state=RANDOM_SEED,
        )

        cv_pipeline = Pipeline([
            ("preprocessor", clone(self.pipeline.transformer)),
            ("model", clone(self._model_instance)),
        ])

        self._model_cv = cross_validate(
            cv_pipeline,
            self.pipeline.X_train,
            self.pipeline.y_train,
            scoring=scoring_metric,
            cv=cv,
            n_jobs=n_jobs,
            return_train_score=True,
        )

        return self
    
    def fit_final_model(self) -> Self:
        """Retrain the selected model on the full raw training pool.

        Clone the transformer and estimator so the original validation
        workflow remains unchanged. The held-out test set is not used.
        """
        self._final_model = None
        self._final_test_prediction = None
        self._final_test_r2_score = None
        self._final_test_rmse = None
        self._final_test_mae = None

        final_pipeline = Pipeline([
            ("preprocessor", clone(self.pipeline.transformer)),
            ("model", clone(self._model_instance)),
        ])

        final_pipeline.fit(
            self.pipeline.X_train,
            self.pipeline.y_train,
        )

        self._final_model = final_pipeline

        return self

    def predict_final_test(self) -> Self:
        """Generate predictions on the held-out test set.

        Requires the final preprocessing/model pipeline to be fitted.
        """
        if self._final_model is None:
            raise RuntimeError(
                "The final model has not been fitted. "
                "Call fit_final_model() first."
            )

        # Clear previous predictions and metrics before generating new ones.
        self._final_test_prediction = None
        self._final_test_r2_score = None
        self._final_test_rmse = None
        self._final_test_mae = None

        self._final_test_prediction = self._final_model.predict(
            self.pipeline.X_test
        )

        return self

    def evaluate_final_test(self) -> Self:
        """Calculate final metrics using the held-out test targets."""
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

    def get_final_test_results(self) -> dict[str, float]:
        """Return the final held-out test evaluation metrics."""
        if (
            self._final_test_r2_score is None
            or self._final_test_rmse is None
            or self._final_test_mae is None
        ):
            raise RuntimeError(
                "Final test metrics are unavailable. "
                "Call evaluate_final_test() first."
            )

        return {
            "r2_score": self._final_test_r2_score,
            "rmse": self._final_test_rmse,
            "mae": self._final_test_mae,
        }