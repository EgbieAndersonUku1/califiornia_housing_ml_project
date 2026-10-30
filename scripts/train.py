"""Train and evaluate models using the California Housing dataset."""

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import pandas as pd

from califiornia_housing_ml_project.data.data_pipeline import UniversalDataPipeline
from califiornia_housing_ml_project.data.preprocessing import AgnosticTabularPreprocessor
from califiornia_housing_ml_project.models.evaluator import PipelineModelEvaluator
from califiornia_housing_ml_project.settings import RANDOM_SEED



PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "housing.csv"

TARGET_COLUMN = "median_house_value"


def main() -> None:
    """Load the dataset and configure preprocessing."""
    housing = pd.read_csv(DATA_PATH)

    X = housing.drop(columns=[TARGET_COLUMN])
    y = housing[TARGET_COLUMN]

    numerical_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    preprocessor = AgnosticTabularPreprocessor(
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
    )

    pipeline = UniversalDataPipeline(
        X=X,
        y=y,
        test_size=0.2,
        random_state=RANDOM_SEED,
    )

    pipeline.set_transformer(preprocessor)
    pipeline.run_test_split()
    pipeline.fit_and_transform()
    pipeline.run_validation_split()

    evaluator = PipelineModelEvaluator(
        model_instance=LinearRegression(),
        pipeline=pipeline,
    )

    evaluator.fit().predict().evaluate()
  
    scoring = {
        "r2": "r2",
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
    }

    evaluator.cross_validation(
        scoring_metric=scoring,
        n_split=5,
        n_jobs=1,
    )

    print("Model evaluation results:")
    print(evaluator.get_results())

    print(f"Training features: {pipeline.X_train_split.shape}")
    print(f"Validation features: {pipeline.X_val_split.shape}")
    print(f"Training targets: {pipeline.y_train_split.shape}")
    print(f"Validation targets: {pipeline.y_val_split.shape}")

    print(f"Dataset shape: {housing.shape}")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Numerical columns: {preprocessor.numerical_columns}")
    print(f"Categorical columns: {preprocessor.categorical_columns}")


    scoring = {
        "r2": "r2",
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
    }

    evaluator.cross_validation(
        scoring_metric=scoring,
        n_split=5,
        n_jobs=1,
    )

    cv_results = evaluator.cv_results

    print("\nCross-validation results:")
    print(f"Mean R²: {cv_results['test_r2'].mean():.4f}")
    print(f"Mean RMSE: {-cv_results['test_rmse'].mean():,.2f}")
    print(f"Mean MAE: {-cv_results['test_mae'].mean():,.2f}")


    print("\nRandom Forest evaluation:")

    rf_evaluator = PipelineModelEvaluator(
      RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
        pipeline,
    )

    rf_evaluator.fit().evaluate()

    print(rf_evaluator.get_results())

    print("\nRandom Forest cross-validation:")

    rf_evaluator.cross_validation(
        scoring_metric=scoring,
        n_split=5,
        n_jobs=1,
    )

    rf_cv = rf_evaluator.cv_results

    print(f"Mean R²: {rf_cv['test_r2'].mean():.4f}")
    print(f"Mean RMSE: {-rf_cv['test_rmse'].mean():,.2f}")
    print(f"Mean MAE: {-rf_cv['test_mae'].mean():,.2f}")

    # Final evaluation of the selected Random Forest model
    print("\nFinal Random Forest held-out test evaluation:")

    final_results = rf_evaluator.fit_final_model().predict_final_test().evaluate_final_test().get_final_test_results()

    print(f"Test R²: {final_results['r2_score']:.4f}")
    print(f"Test RMSE: ${final_results['rmse']:,.2f}")
    print(f"Test MAE: ${final_results['mae']:,.2f}")

if __name__ == "__main__":
    main()

