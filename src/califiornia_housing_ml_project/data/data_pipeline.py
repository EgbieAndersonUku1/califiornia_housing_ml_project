# """Provides a reusable pipeline for preparing tabular datasets for machine learning.

# The UniversalDataPipeline class orchestrates structural split, extraction, and 
# transformation operations across multi-dimensional feature matrices while enforcing strict 
# data isolation boundaries to eliminate data leakage.

# Workflow Integration & Sequencing Guardrails:
#     This pipeline requires an external feature engineering and scaling engine passed directly 
#     to its transformation hooks. It is designed to consume a configured instance of 
#     `AgnosticTabularPreprocessor` (which packages target domain-specific mathematical functions, 
#     such as `housing_math_booster` or `geographic_math_booster`). 

#     The transformer must be registered via `set_transformer()` after initializing the dataset 
#     bounds but prior to calling the execution steps.

#     Strict operational ordering is enforced:
#     1. `run_test_split()` must be invoked first to establish baseline data partitions.
#     2. `fit_and_transform()` must be called to prepare the initial matrices.
#     3. `run_validation_split()` can then optionally be executed. Calling this out of order 
#        will raise a structural error, as it relies directly on the training arrays generated 
#        by the preceding workflow steps.

# The class coordinates:
#     - Partitioning raw dataset frames into stratified training and testing arrays.
#     - Providing an optional validation partition split step via `run_validation_split()` 
#       which isolates validation subsets entirely out of the prepared training data.
#     - Accepting and managing an active `AgnosticTabularPreprocessor` pipeline transformer.
#     - Sequentially applying isolated fit and transform workflows across separate data vectors.
#     - Monitoring execution states to ensure operations follow the correct order.
#     - Providing clean, cached access to the prepared arrays (`X_train_combined`, etc.).

# Example:
#     >>> pipeline = UniversalDataPipeline(X=X, y=y, test_size=0.2, random_state=42)
#     >>> pipeline.set_transformer(agnostic_tabular_preprocessor_instance)
#     >>> 
#     >>> # 1. Isolate test partition and transform data vectors
#     >>> pipeline.run_test_split()
#     >>> pipeline.fit_and_transform()
#     >>> 
#     >>> # 2. Generate validation sets using internal states (Requires steps above)
#     >>> X_train, X_val, y_train, y_val = pipeline.run_validation_split()
# """


from typing import Any

from numpy import ndarray
from numpy.typing import ArrayLike
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split

from califiornia_housing_ml_project.errors.custom_errors import (
    AgnosticTabularPreprocessorError,
    IncorrectTransformerClassError,
)
from califiornia_housing_ml_project.settings import RANDOM_SEED
from califiornia_housing_ml_project.utils.types import ScikitTransformer, is_transformer
from califiornia_housing_ml_project.utils.validator import validate_dataframe



class UniversalDataPipeline:
    """
    Coordinates the splitting and preparation of tabular data
    for machine learning.

    Manages training, validation, and test datasets, interfaces
    with a configured data transformer, and tracks processing
    states to ensure operations are performed in the correct order.
    """

    def __init__(
        self,
        X: DataFrame,
        y: DataFrame | Series,
        test_size: float,
        random_state: int = RANDOM_SEED
    ):
        """
        Initialise the data pipeline with features, target, and
        train-test splitting configuration.

        Args:
            X: Input features used to prepare the dataset.
            y: Target values corresponding to the input features.
            test_size: Proportion of the dataset reserved for testing.
            random_state: Seed used to make random splitting reproducible.
                Defaults to RANDOM_SEED.
        """
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state

        self.transformer = None

        # Internal states for tracking operational processing
        self._has_split = False
        self._has_transformed = False
        self._model = None
        self._is_transformer_fitted = False

        self._X_train: DataFrame | None = None
        self._X_test: DataFrame | None = None
        self._y_train: Series | DataFrame | None = None
        self._y_test: Series | DataFrame | None = None

        self._X_train_combined: ndarray | None = None
        self._X_test_combined: ndarray | None = None

        self._X_train_split: ndarray | None = None
        self._X_val_split: ndarray | None = None
        self._y_train_split: ndarray | None = None
        self._y_val_split: ndarray | None = None

        self._validation_split_done = False
        self._is_model_fitted = False
        self._X_train_raw_split = None
        self._X_val_raw_split = None
        self._y_train_raw_split = None
        self._y_val_raw_split = None

    def _verify_split(self) -> None:
        """
        Verify that the train-test split has been performed.

        Raises:
            RuntimeError: If run_test_split() has not been called.
        """
        if not self._has_split:
            raise RuntimeError(
                "run_test_split() must be called before accessing this state."
            )

    def _verify_transformer_is_set(self) -> None:
        """
        Verify that a data transformer has been configured.

        Raises:
            AgnosticTabularPreprocessorError: If no transformer has been set.
        """
        if self.transformer is None:
            raise AgnosticTabularPreprocessorError(
                "The tabular data transformer hasn't been set. "
                "Call set_transformer(...)"
            )


    def _verify_transformer_is_fitted(self) -> None:
        """
        Verify that the data transformer has been fitted.

        Raises:
            RuntimeError: If the transformer has not been fitted.
        """
        if not self._is_transformer_fitted:
            raise RuntimeError(
                "The transformer must be fitted before transforming new data. "
                "Call fit_and_transform() first."
            )

    def _verify_transform(self) -> None:
        """
        Verify that preprocessing has been completed successfully.

        Ensures transformed feature matrices are accessed only after
        fit_and_transform() has completed.
        
        Raises:
            RuntimeError: If preprocessing has not been completed.
        """
        if not self._has_transformed:
            raise RuntimeError(
                "fit_and_transform() must be called before accessing "
                "the transformed data."
            )


    def _verify_model_is_fitted(self) -> None:
        """
        Verify that a model has been fitted successfully.

        Checks that a model exists and that model fitting is complete
        before allowing access to the fitted model.

        Raises:
            RuntimeError: If no model is available or model fitting
                has not completed successfully.
        """
        if self._model is None:
            raise RuntimeError(
                "fit_model() must be executed before accessing the model state."
            )

        if not self._is_model_fitted:
            raise RuntimeError(
                "fit_model() must complete successfully before "
                "accessing the fitted model."
            )


    def _verify_validation_split_occurred(self) -> None:
        """
        Verify that the training-validation split has been completed.

        Raises:
            RuntimeError: If run_validation_split() has not been called
                successfully before accessing the validation data.
        """
        if not self._validation_split_done:
            raise RuntimeError(
                "run_validation_split() must be called before "
                "accessing the validation split data."
            )
    def _reset_state(self) -> None:
        """
        Reset the pipeline state and clear all stored processing results.

        Clears processing flags, the fitted model reference, and stored
        training, validation, and test data, including combined feature
        matrices.

        Notes:
            The configured transformer and original input data are retained.
            The pipeline's processing workflow can be run again from the
            initial state.
        """
        self._has_split              = False
        self._is_transformer_fitted  = False
        self._has_transformed        = False
        self._validation_split_done  = False
        self._is_model_fitted        = False
        self._model                  = None
        self._X_train                = None
        self._X_test                 = None
        self._y_train                = None
        self._y_test                 = None
        self._X_train_split          = None
        self._X_val_split            = None
        self._y_train_split          = None
        self._y_val_split            = None
        self._X_train_combined       = None
        self._X_test_combined        = None
        self._X_train_raw_split     = None
        self._X_val_raw_split       = None
        self._y_train_raw_split     = None
        self._y_val_raw_split       = None


    def set_transformer(self, transformer_instance: ScikitTransformer) -> None:
        """
        Configure the pipeline's data transformer.

        Sets the transformer instance and invalidates transformed feature
        matrices and the stored model associated with the previous
        transformer.

        Args:
            transformer_instance: Transformer instance to use for
                preprocessing.

        Raises:
            IncorrectTransformerClassError: If the supplied instance does
                not satisfy the expected transformer interface.
        """
        if not is_transformer(transformer_instance):
            raise IncorrectTransformerClassError(transformer_instance)
       
        self.transformer = transformer_instance

        # The new transformer has not been fitted by this pipeline.
        self._is_transformer_fitted = False
        self._has_transformed       = False

        # Remove results and model associated with the previous transformer.
        self._X_train_combined = None
        self._X_test_combined  = None
        self._model            = None


    @property
    def X_train_split(self) -> ndarray:
        """
        Return the training feature matrix after the validation split.

        Returns:
            ndarray: Training features for model development.

        Raises:
            RuntimeError: If run_validation_split() has not been called
                successfully.
        """
        self._verify_validation_split_occurred()
        return self._X_train_split
    
    @property
    def X_val_split(self) -> ndarray:
        """Return the validation feature matrix after the validation split.

        Verify that the validation split has been performed before
        exposing the validation features.

        Returns:
            ndarray: The validation feature matrix.

        Raises:
            RuntimeError: If run_validation_split() has not been called.
        """
        self._verify_validation_split_occurred()
        return self._X_val_split

    @property
    def y_train_split(self) -> ndarray:
        """Return the training target values after the validation split.

        Verify that the validation split has been performed before
        exposing the training targets.

        Returns:
            ndarray: The target values corresponding to the training
                feature matrix.

        Raises:
            RuntimeError: If run_validation_split() has not been called.
        """
        self._verify_validation_split_occurred()
        return self._y_train_split

    @property
    def y_val_split(self) -> ndarray:
        """Return the validation target values after the validation split.

        Verify that the validation split has been performed before
        exposing the validation targets.

        Returns:
            ndarray: The target values corresponding to the validation
                feature matrix.

        Raises:
            RuntimeError: If run_validation_split() has not been called.
        """
        self._verify_validation_split_occurred()
        return self._y_val_split

    @property
    def X_train_combined(self) -> ndarray:
        """Return the combined training feature matrix after preprocessing.

        Verify that the transformation step has been completed before
        exposing the combined training features.

        Returns:
            ndarray: The horizontally stacked training feature matrix.

        Raises:
            RuntimeError: If the transformation step has not been performed.
        """
        self._verify_transform()
        return self._X_train_combined
    
    @property
    def X_train(self) -> ndarray:
        """Return the original training feature matrix.

        Returns:
            ndarray: The training features before preprocessing
                and feature combination.
        """
        return self._X_train

    @property
    def X_test(self) -> ndarray:
        """Return the original test feature matrix.

        Returns:
            ndarray: The test features before preprocessing
                and feature combination.
        """
        return self._X_test

    @property
    def y_train(self) -> Any:
        """Return the training target values.

        Verify that the initial data split has been performed before
        exposing the training targets.

        Returns:
            Any: The target values corresponding to the training features.

        Raises:
            RuntimeError: If the initial data split has not been performed.
        """
        self._verify_split()
        return self._y_train
    
    @property
    def y_test(self) -> Any:
        """Return the test target values.

        Verify that the initial data split has been performed before
        exposing the test targets.

        Returns:
            Any: The target values corresponding to the test features.

        Raises:
            RuntimeError: If the initial data split has not been performed.
        """
        self._verify_split()
        return self._y_test

    @property
    def X_test_combined(self) -> ndarray:
        """Return the combined test feature matrix after preprocessing.

        Verify that the transformation step has been completed before
        exposing the combined test features.

        Returns:
            ndarray: The horizontally stacked test feature matrix.

        Raises:
            RuntimeError: If the transformation step has not been performed.
        """
        self._verify_transform()
        return self._X_test_combined

    def run_test_split(self) -> None:
        """Create training and test partitions and invalidate dependent state.

        Verify that a transformer has been set before splitting the data.
        Reset the existing state before creating the new partitions.

        Raises:
            RuntimeError: If a transformer has not been set.
        """
        self._verify_transformer_is_set()
        self._reset_state()

        (
            self._X_train,
            self._X_test,
            self._y_train,
            self._y_test,
        ) = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        self._has_split = True
    
    @property
    def trained_model(self) -> Any:
        """Return the trained model after verifying that it has been fitted.

        Verify that the model has been fitted before exposing the model
        instance.

        Returns:
            Any: The fitted model instance.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._verify_model_is_fitted()
        return self._model

    def fit_and_transform(self) -> None:
        """Split raw training pool, fit on training subset, transform all splits.

        Validation and held-out test observations never contribute to fitted
        feature imputation, encoding, or scaling statistics.
        """
        self._verify_transformer_is_set()
        self._verify_split()

        self._is_transformer_fitted = False
        self._has_transformed = False
        self._validation_split_done = False
        self._X_train_combined = None
        self._X_test_combined = None
        self._X_train_split = None
        self._X_val_split = None
        self._y_train_split = None
        self._y_val_split = None
        self._model = None
        self._is_model_fitted = False

        (
            self._X_train_raw_split,
            self._X_val_raw_split,
            self._y_train_raw_split,
            self._y_val_raw_split,
        ) = train_test_split(
            self._X_train,
            self._y_train,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        self.transformer.fit(self._X_train_raw_split)
        self._is_transformer_fitted = True

        self._X_train_split = self.transformer.transform(self._X_train_raw_split)
        self._X_val_split = self.transformer.transform(self._X_val_raw_split)
        self._X_test_combined = self.transformer.transform(self._X_test)

        self._X_train_combined = self.transformer.transform(self._X_train)
        self._has_transformed = True

    
    def clear_training_data(self) -> None:
        """Remove stored datasets while preserving fitted preprocessing.

        Clear raw and prepared data partitions to reduce the size of the
        serialized pipeline. Preserve the fitted feature transformer,
        target scaler, and inference methods for deployment.

        Reset dataset-processing flags because the stored partitions
        are no longer available.
        """

        self.X = None
        self.y = None
        self._X_train = None
        self._X_test = None
        self._y_train = None
        self._y_test = None
        self._X_train_combined = None
        self._X_test_combined = None
        self._X_val_split = None
        self._X_train_split = None
        self._y_val_split = None
        self._y_train_split = None
        self._has_split = False
        self._has_transformed = False
        self._validation_split_done = False
        self._X_train_raw_split = None
        self._X_val_raw_split = None
        self._y_train_raw_split = None
        self._y_val_raw_split = None

    def transform_new_data(self, X_new: DataFrame) -> ndarray:
        """Transform new feature data using the fitted transformer.

        Validate the input DataFrame and verify that a transformer has
        been set and fitted before transforming the new data.

        Args:
            X_new (DataFrame): The new feature data to transform.

        Returns:
            ndarray: The transformed feature matrix.

        Raises:
            ValueError: If X_new is not a valid DataFrame.
            RuntimeError: If a transformer has not been set or fitted.
        """
        validate_dataframe(X_new)
        self._verify_transformer_is_set()
        self._verify_transformer_is_fitted()
        return self.transformer.transform(X_new)

    def fit_model(self, model_instance: Any) -> Any:
        """Fit a model using the pipeline's transformed training features.

        Retrieve the transformed training feature matrix and corresponding
        target values, then fit the supplied estimator and store the
        fitted model instance.

        Args:
            model_instance (Any): An unfitted scikit-learn-compatible
                estimator.

        Returns:
            Any: The fitted model instance.

        Raises:
            RuntimeError: If the required data split or transformation
                has not been performed.
        """
      
        self._verify_validation_split_occurred()
        features = self.X_train_split
        labels = self.y_train_split
        self._is_model_fitted = False

        history = model_instance.fit(features, labels)
        self._is_model_fitted = True
        self._model = history
        return history

    def get_feature_names_out(self):
        """Return the output feature names generated by the transformer.

        Returns:
            ndarray: The feature names corresponding to the transformed
                feature matrix.
        """
        return self.transformer.get_feature_names_out()

    def tf_dataset_tensorflow(
        self,
        split: str = "train",
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> Any:
        """Create a batched TensorFlow dataset from a data partition.

        Retrieve the feature matrix and target values for the specified
        partition, then construct a TensorFlow dataset with optional
        shuffling, batching, and prefetching. Shuffling is disabled
        automatically for validation and test partitions.

        Args:
            split (str): The data partition to use. Must be "train",
                "validation", or "test". Defaults to "train".
            batch_size (int): The number of samples per batch.
                Defaults to 32.
            shuffle (bool): Whether to shuffle the training dataset.
                Defaults to True. Ignored for validation and test
                partitions, which are not shuffled.

        Returns:
            tf.data.Dataset: The batched and prefetched TensorFlow dataset
                containing the selected features and target values.

        Raises:
            ValueError: If split is not "train", "validation", or "test".
            RuntimeError: If the required data split or transformation
                has not been performed.
        """
        import tensorflow as tf

        if split == "train":
            self._verify_validation_split_occurred()
            features = self.X_train_split
            labels = self.y_train_split
        elif split == "validation":
            features = self._X_val_split
            labels = self._y_val_split
            shuffle = False
        elif split == "test":
            features = self.X_test_combined
            labels = self.transformer.transform_target(self.y_test)
            shuffle = False
        else:
            raise ValueError(
                "Split must be 'train', 'validation', or 'test'."
            )

        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(features))

        return dataset.batch(batch_size).prefetch(
            buffer_size=tf.data.AUTOTUNE
        )
    
    def run_validation_split(self):
        """Return preprocessed training/validation features and scaled targets.

        Raw splitting and feature fitting occur in fit_and_transform().
        The target scaler is fitted only on training-subset targets.
        """
        self._verify_transform()
        self._verify_transformer_is_fitted()

        if self._X_train_split is None or self._X_val_split is None:
            raise RuntimeError("Call fit_and_transform() before validation split.")

        self._y_train_split = self.transformer.fit_transform_target(
            self._y_train_raw_split
        )
        self._y_val_split = self.transformer.transform_target(
            self._y_val_raw_split
        )
        self._validation_split_done = True

        return (
            self._X_train_split,
            self._X_val_split,
            self._y_train_split,
            self._y_val_split,
        )

    def inverse_transform(self, y_scaled: ArrayLike) -> ndarray:
        """Convert scaled target values back to their original scale.

        Args:
            y_scaled: Scaled target values, provided as an array-like
                object such as a NumPy array or list.

        Returns:
            A one-dimensional NumPy array containing target values
            in their original scale.

        Raises:
            RuntimeError: If the underlying preprocessor has not been fitted.
            ValueError: If the input cannot be converted into a valid
                numeric array or the scaler rejects its shape or values.
        """
        return self.transformer.inverse_transform_target(y_scaled)



