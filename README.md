# California Housing Price Prediction — End-to-End Machine Learning Project

An end-to-end deep learning project for predicting median housing values across California districts, from data exploration and preprocessing to neural network training, experiment tracking, and deployment.

The project uses **TensorFlow/Keras**, **Scikit-learn**, **MLflow**, **FastAPI**, **Docker**, and **Streamlit** to demonstrate a complete machine learning workflow.

---

## Table of Contents

1. [Project Overview](#project-overview)

2. [Project Objectives](#project-objectives)

3. [Technology Stack](#technology-stack)

4. [Dataset](#dataset)

5. [Data Preprocessing](#data-preprocessing)

6. [Neural Network Architectures](#neural-network-architectures)

7. [Model Evaluation](#model-evaluation)

8. [Project Structure](#project-structure)

9. [Installation](#installation)

10. [Scikit-learn Model Training and Evaluation](#scikit-learn-model-training-and-evaluation)
11. [MLflow Experiment Tracking](#mlflow-experiment-tracking)

12. [Running the FastAPI Backend with Docker](#running-the-fastapi-backend-with-docker)

13. [Example API Prediction](#example-api-prediction)

14. [Running the Streamlit Frontend](#running-the-streamlit-frontend)

15. [Saved Model Artifacts](#saved-model-artifacts)

16. [End-to-End Verification](#end-to-end-verification)

17. [Business Insights](#business-insights)

18. [Project Limitations](#project-limitations)

19. [Future Improvements](#future-improvements)

20. [Conclusion](#conclusion)

---

## Project Overview

This project develops a machine learning application for estimating median housing values across California districts.

Three neural network architectures were trained and evaluated using a reusable data preprocessing pipeline. The best-performing model was selected based on validation performance and evaluated on a held-out test dataset.

The final model was saved and deployed using a Dockerized FastAPI backend, with an interactive Streamlit frontend for generating predictions.

The project demonstrates the complete machine learning lifecycle:

**Data Exploration → Preprocessing → Model Training → Evaluation → Experiment Tracking → Model Persistence → API Deployment → Web Application**

## Project Objectives

- Explore and understand the California Housing dataset.
- Develop a reusable data cleaning and preprocessing workflow.
- Train and compare three neural network regression models.
- Evaluate predictive performance using MAE, RMSE, and R².
- Track experiments and model metrics using MLflow.
- Save and reload the selected model and preprocessing artifacts.
- Deploy predictions through a REST API using FastAPI.
- Containerize the API using Docker.
- Build an interactive prediction application using Streamlit.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming language | Python 3.14 |
| Data processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine learning | Scikit-learn |
| Deep learning | TensorFlow, Keras |
| Experiment tracking | MLflow, SQLite |
| Backend API | FastAPI, Uvicorn |
| Containerization | Docker |
| Frontend | Streamlit |
| Artifact persistence | Joblib, Keras |
| Development tools | VS Code, Jupyter Notebook |

## Dataset

The project uses the California Housing dataset, which contains district-level geographical, demographic, and housing-related information.

The original dataset contains **20,640 observations**, nine input features, and one target variable.

### Input Features

| Feature | Description |
|---|---|
| `longitude` | Geographic longitude |
| `latitude` | Geographic latitude |
| `housing_median_age` | Median age of housing units |
| `total_rooms` | Total rooms in the district |
| `total_bedrooms` | Total bedrooms in the district |
| `population` | District population |
| `households` | Number of households |
| `median_income` | Median income |
| `ocean_proximity` | Proximity to the ocean |

**Target variable:** `median_house_value`

### Dataset Filtering (Earlier Neural Network Experiments)

Observations with median housing values of **$500,000 or more** were excluded during preparation.

- Original observations: 20,640
- Excluded observations: 992
- Remaining observations: 19,648

This filtering decision is important when interpreting the model's applicability to higher-valued districts.

### Dataset Splits (Earlier Neural Network Experiments)

| Split | Observations |
|---|---:|
| Training | 12,574 |
| Validation | 3,144 |
| Held-out test | 3,930 |
| **Total** | **19,648** |

## Data Preprocessing

The project uses a custom reusable preprocessing pipeline to prepare the data consistently for training and inference.

The pipeline handles missing numerical values, encodes categorical variables, and scales numerical inputs and the target variable.

The earlier neural network workflow transforms the nine input features into **17 numerical features**; the scikit-learn workflow described below produces **13 transformed features**.

The fitted preprocessing components are saved so that the deployed application can apply the same transformations to new prediction requests.

## Neural Network Architectures

Three fully connected neural network architectures were developed and compared.

| Model | Hidden Layer Architecture |
|---|---|
| Model 1 | 64 → 32 |
| Model 2 | 128 → 64 |
| Model 3 | 64 → 32 → 16 |

### Training Configuration

| Parameter | Value |
|---|---|
| Epochs | 100 |
| Batch size | 32 |
| Learning rate | 0.001 |
| Loss function | Mean Absolute Error |
| L2 regularization | 0.0001 |
| Dropout rate | 0.1 |
| Batch normalization | Enabled |

The models were trained on preprocessed features and evaluated using a separate validation dataset.

## Model Evaluation

Three evaluation metrics were used:

- **MAE (Mean Absolute Error):** Average absolute difference between predicted and observed values.
- **RMSE (Root Mean Squared Error):** Measures prediction error while giving greater weight to large errors.
- **R² (Coefficient of Determination):** Measures the proportion of target variation explained by the model.

### Validation Results

| Model | MAE (USD) | RMSE (USD) | R² |
|---|---:|---:|---:|
| Model 1 | $36,094.70 | $65,816.36 | 0.5308 |
| Model 2 | $43,624.96 | $64,545.77 | 0.5488 |
| **Model 3** | **$36,022.84** | **$54,510.57** | **0.6782** |

Model 3 achieved the lowest validation MAE and was selected as the final model.

### Held-Out Test Results

Model 3 was selected based on validation performance and subsequently evaluated on the held-out test dataset.

| Metric | Result |
|---|---:|
| MAE | **$36,925.39** |
| RMSE | **$53,749.17** |
| R² | **0.6987** |

The model explains approximately **69.87%** of the variation in median housing values within the filtered test dataset.

The held-out test MAE is approximately 2.5% higher than the validation MAE, suggesting reasonably consistent performance on unseen observations.

Although the model captures useful relationships in the dataset, individual predictions may contain substantial errors.

## Project Structure

```text

califiornia_housing_ml_project/

├── artifacts/

│   ├── housing_model.keras

│   ├── preprocessor.joblib

│   └── feature_columns.joblib

├── backend/

│   └── app.py

├── frontend/

│   └── app.py

├── src/

│   └── ...

├── Dockerfile

├── requirements.txt

├── mlflow.db

└── README.md

```

This is a simplified structure. Additional source modules, notebooks, and data directories may be present in the repository.

## Installation

### 1. Clone the Repository

```bash

git clone https://github.com/abbieRU/W2_ml_project

cd califiornia_housing_ml_project

```

Replace `<your-repository-url>` with the actual repository URL.

### 2. Create a Virtual Environment

```bash

python -m venv .venv

```

Activate the environment on Windows PowerShell:

```powershell

.\.venv\Scripts\Activate.ps1

```

### 3. Install Dependencies

```bash

python -m pip install -r requirements.txt

```


### Dataset Download and Setup

This project uses the [California Housing Prices dataset on Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices).

The dataset is not included in this GitHub repository. To reproduce the training experiments:

1. Visit the Kaggle dataset page and click **Download**.
2. Extract the downloaded ZIP file.
3. Create the following directory in the project root if it does not already exist:

   `data/raw/`

4. Copy `housing.csv` into that directory.

Your project should have the following structure:

```text
califiornia_housing_ml_project/
├── data/
│   └── raw/
│       └── housing.csv
├── scripts/
│   └── train.py
├── src/
├── tests/
└── README.md
```

After installing the project dependencies with:

```powershell
uv sync
```

Run the training and evaluation pipeline:

```powershell
uv run python .\scripts\train.py
```

The script trains and evaluates Linear Regression and Random Forest models using the California Housing dataset.

**Note:** The `data/raw/` directory is excluded from version control through `.gitignore`, so the dataset must be downloaded separately.


## Scikit-learn Model Training and Evaluation

The project also includes a reusable scikit-learn training workflow implemented in `scripts/train.py`.

This workflow evaluates Linear Regression and Random Forest models using the California Housing dataset. It is separate from the neural network experiments and deployment workflow described elsewhere in this README.

### Running the Training Script

From the project root, install the project dependencies using `uv`:

```powershell

uv sync

```

Ensure the dataset is available at:

```text

data/raw/housing.csv

```

Run the training and evaluation script:

```powershell

uv run python .\scripts\train.py

```

The script:

1. Loads the California Housing dataset.

2. Separates the `median_house_value` target from the input features.

3. Identifies numerical and categorical features.

4. Creates training, validation, and held-out test splits.

5. Fits preprocessing transformations using training data only.

6. Trains and evaluates Linear Regression and Random Forest models.

7. Performs five-fold cross-validation on the training pool.

8. Reports R², RMSE, and MAE, with error metrics expressed in US dollars.

The held-out test set is not used during model selection or hyperparameter tuning.

### Dataset Used in the Scikit-learn Experiments

The current workflow uses 20,640 observations, including eight numerical features and one categorical feature. After preprocessing, the input data contains 13 transformed features.

Unlike the earlier neural network experiments described above, this workflow does not exclude observations with housing values of $500,000 or more.

### Model Comparison

The following results were obtained using five-fold cross-validation:

| Model | Mean R² | Mean RMSE | Mean MAE |
|---|---:|---:|---:|
| Linear Regression | 0.6477 | $68,604.16 | $49,667.16 |
| Random Forest (default leaf size) | 0.8171 | $49,445.29 | $32,328.53 |
| Random Forest (`min_samples_leaf=3`) | 0.8171 | $49,438.33 | $32,390.36 |
| Random Forest (`min_samples_leaf=5`) | 0.8131 | $49,982.35 | $32,869.96 |

Random Forest substantially outperformed Linear Regression. The configuration using `min_samples_leaf=3` was selected as a reasonable balance between predictive performance and model complexity.

These are cross-validation results, not final held-out test results.

### Final Random Forest Held-Out Test Results

After model selection, the Random Forest model with `n_estimators=100` and `min_samples_leaf=3` was retrained on the full training pool and evaluated once on the held-out test dataset.

| Metric | Result |
|---|---:|
| R² | **0.8157** |
| RMSE | **$49,147.47** |
| MAE | **$31,686.45** |

The model explains approximately **81.57% of the variation** in median housing values on the held-out test dataset.

Its test performance is consistent with the five-fold cross-validation results, providing evidence of generalization to unseen observations.

The held-out test dataset was not used during model selection or hyperparameter tuning.

These results relate to the scikit-learn Random Forest workflow and are separate from the earlier neural network deployment results.


### Running Automated Tests

From the project root:

```powershell

uv run pytest -q

```

The test suite currently contains 43 passing tests covering the reusable machine learning components.

### Training Script Location

```text

scripts/

└── train.py

```

The training script uses the reusable preprocessing, data pipeline, and model evaluation components in `src/califiornia_housing_ml_project/`.

## MLflow Experiment Tracking

MLflow is used to track model architectures, training parameters, validation metrics, and saved model artifacts.

The project uses a SQLite tracking database.

Start the MLflow UI from the project root:

```bash

python -m mlflow ui --backend-store-uri sqlite:///mlflow.db

```

Open:

`http\://localhost:5000`

**Experiment name:** `California Housing Neural Networks`

The experiment contains separate runs for the three retrained neural network models.

## Running the FastAPI Backend with Docker

The trained model is served through a FastAPI REST API running inside a Docker container.

### 1. Build the Docker Image

```bash

docker build -t housing-api .

```

### 2. Start the API Container

```bash

docker run -d --name housing-api-container -p 8000:8000 housing-api

```

If a container with the same name already exists, remove it first:

```bash

docker rm -f housing-api-container

```

### 3. Check API Health

Open:

`http\://localhost:8000/health`

Expected response:

```json

{

  "status": "ok"

}

```

### 4. Open the API Documentation

FastAPI provides interactive documentation at:

`http\://localhost:8000/docs`

Use the `/predict` endpoint to test predictions.

## Example API Prediction

**Endpoint:** `POST /predict`

Example request:

```json

{

  "longitude": -122.23,

  "latitude": 37.88,

  "housing_median_age": 41,

  "total_rooms": 880,

  "total_bedrooms": 129,

  "population": 322,

  "households": 126,

  "median_income": 8.3252,

  "ocean_proximity": "NEAR BAY"

}

```

Example response:

```json

{

  "predicted_house_value": 340722.38,

  "currency": "USD",

  "model": "model_3"

}

```

The prediction is expressed in US dollars and represents an estimated district median housing value.

## Running the Streamlit Frontend

The Streamlit frontend provides an interactive interface for entering housing characteristics and generating predictions.

Keep the Dockerized FastAPI backend running.

From the project root, open a separate terminal and run:

```bash

python -m streamlit run frontend/app.py

```

Open the application:

`http\://localhost:8501`

Enter the nine housing features and submit the form.

Streamlit sends the request to the FastAPI backend, which preprocesses the inputs, generates a prediction using Model 3, and returns the result.

## Saved Model Artifacts

| Artifact | Purpose |
|---|---|
| `housing_model.keras` | Stores the selected trained neural network |
| `preprocessor.joblib` | Stores the fitted preprocessing pipeline |
| `feature_columns.joblib` | Stores the expected feature names and order |

These artifacts allow the application to generate predictions without retraining the model.

## End-to-End Verification

The deployed application was tested using the same housing features through both the FastAPI documentation interface and Streamlit.

| Interface | Model Used | Predicted Value |
|---|---|---:|
| Local FastAPI (port 8001) | Model 3 | **$340,722.38** |
| Docker FastAPI (port 8000) | Model 3 | **$340,722.38** |
| Streamlit frontend | Model 3 | **$340,722.38** |

**Result: Both interfaces returned exactly the same prediction using Model 3.**

This confirms that the Streamlit frontend successfully communicates with the Dockerized FastAPI backend and that the deployed application uses the selected neural network consistently for the tested input.

The API health endpoint was also verified successfully.

## Business Insights

### 1. District Characteristics Provide Predictive Information

Geographical, demographic, and housing-related features provide useful signals for estimating district median housing values.

### 2. Larger Networks Do Not Always Perform Better

Model 2 used wider hidden layers but performed worse than Models 1 and 3 on the validation dataset.

Model 3 achieved the strongest validation performance, demonstrating the importance of evaluating architectures rather than relying on network size alone.

### 3. Consistent Preprocessing Is Essential

Saving the fitted preprocessing pipeline helps ensure that training and deployment use consistent feature transformations.

### 4. Deployment Makes Models Accessible

FastAPI, Docker, and Streamlit allow the trained model to be accessed through a user-friendly prediction application.

## Project Limitations

- **Target filtering:** The model was trained only on districts with median housing values below $500,000. Its performance on higher-valued districts has not been established.

**Model 3** achieved a held-out test R² of approximately **0.6987**, with an MAE of **$36,925.39** and an RMSE of **$53,749.17**.- **District-level estimates:** Predictions should not be interpreted as individual property appraisals.

- **Historical data:** The model may not reflect current housing market conditions without retraining on updated information.
- **Limited features:** The dataset does not contain many property-specific characteristics or current economic indicators.

## Future Improvements

- Extend comparisons between neural networks and tree-based regression algorithms.
- Perform systematic hyperparameter optimization.
- Explore additional geographical and socioeconomic features.
- Investigate alternative approaches to the upper target-value cap.
- Add automated tests for preprocessing and API predictions.
- Implement model monitoring and deployment automation.

## Conclusion

**Model 3** achieved a held-out test R² of approximately **0.6987**, with an MAE of **$36,925.39** and an RMSE of **$53,749.17**.

The final model was successfully deployed through a Dockerized FastAPI backend and connected to an interactive Streamlit frontend.

The resulting application provides a practical foundation for district-level housing value estimation and further machine learning development.
