"""Centralised configuration registry for pipeline parameters and storage paths.

This module hosts global constant settings used across the data engineering, 
model building, and evaluation lifecycles. Maintaining these variables in a 
single file ensures reproducible seeds and uniform operational structures.

Configuration Settings:
    - ARTIFACTS_FOLDER_NAME: The default directory name where training exports, 
      serialised models, and evaluation report dataframes are stored.
    - MODEL_URI: A parameterised tracking string template for resolving unique 
      model artifact assets within an MLflow run ecosystem.
    - RANDOM_SEED: The global random state integer applied across all shuffling, 
      splitting, and layer initialization routines to maintain absolute execution 
      reproducibility.
    - N_JOBS: The default number of parallel CPU worker cores to allocate for 
      multiprocessing tasks, such as parallel cross-validation folds.
"""

ARTIFACTS_FOLDER_NAME = "artifacts"
MODEL_URI = "runs:/{run_id}/{artifact_path}"
RANDOM_SEED = 42
N_JOBS = 1
