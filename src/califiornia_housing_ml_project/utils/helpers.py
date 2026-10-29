from os import getcwd
from pathlib import Path

from pandas import DataFrame, Index, Series, read_csv

from califiornia_housing_ml_project.utils.validator import validate_dataframe


def load_csv_as_dataframe(file_name: str) -> DataFrame:
    """Loads a CSV file into a pandas DataFrame from designated directories.

    The function searches for the file in the 'data/raw/' directory first. 
    If not found, it checks the parent root directory.

    Args:
        file_name: The name of the CSV file to load (including extension).

    Returns:
        DataFrame: The loaded pandas DataFrame.

    Raises:
       TypeError: If `file_name` is not a string.
        ValueError: If `file_name` does not end with a '.csv' extension.
        FileNotFoundError: If the file cannot be found in 'data/raw/' or root.

    Example:
        >>> load_csv_as_dataframe("example.csv")
        ... col1  col2  col3
        ...  0     1     2
    """

    if not isinstance(file_name, str):
        raise TypeError(f"The file path must be a string. Got type {type(file_name).__name__}")

    if not file_name.endswith(".csv"):
        raise ValueError("The file is not a csv file. Check extenstion and try again")

    parent           = Path(getcwd()).parent
    source_file_path = Path(parent / "data" / "raw" / f"{file_name}")

    if source_file_path.exists():
        return read_csv(source_file_path)

    file_path = Path(parent / f"{file_name}")

    if not file_path.exists():
        raise FileNotFoundError(f"The csv file {file_name} couldn't be located in the parent folder ({parent}) or in the /data/raw/ folder")
    
    return read_csv(file_path)



def get_entire_row(df: DataFrame, column_name: str, min_row=True) -> DataFrame:
    """
    Takes a DataFrame and a column name and returns the entire row
    containing the minimum value in that column if the `min_row` is
    set to True else returns the maximum value in that colum

    Args:
        df (DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to search.
        min_row : bool: Defaults to True, returns the minimum value for that
                    row. if set to False returns the maximun value in that row

    Raises:
        TypeError:
            If df is not a DataFrame.
            If column_name is not a string.

    Example:
        >>> get_entire_row(df, "Age")
    """

    validate_dataframe(df)

    if not isinstance(column_name, str):
        raise TypeError(
            f"Column name is not a string. "
            f"Expected a string, got type {type(column_name).__name__}"
        )

    if not isinstance(min_row, bool):
        raise TypeError(
            f"The min_row parameter is not a bool. Got type {type(min_row).__name__}"
        )

    if min_row:
        return df.loc[[df[column_name].idxmin()]]
    return df.loc[[df[column_name].idxmax()]]





def _verbose_comment(not_found_msg: str, found_msg: str, verbose_mode: bool, value: int) -> None:
    """Print a conditional message based on verbose mode and a numeric value.

    Args:
        not_found_msg: The message to print if `value` is 0 or less.
        found_msg: The message to print if `value` is greater than 0.
        verbose_mode: If False, the function returns immediately without printing.
        value: The count or value determining which message to display.

    Raises:
        TypeError: If `verbose_mode` is not a boolean.
    """
    if not isinstance(verbose_mode, bool):
        raise TypeError(
            f"The parameter verbose_mode is not a bool. Got type {type(verbose_mode).__name__}"
        )    

    if not verbose_mode:
        return
    
    if value > 0:
        print(found_msg)
    else:
        print(not_found_msg)


def count_missing_values(df: DataFrame, verbose_mode: bool = True) -> int:
    """Count all missing cells in a DataFrame.

    Args:
        df: The pandas DataFrame to inspect for missing values.
        verbose_mode: If True, prints a summary message about the missing values found.
            Defaults to True.

    Returns:
        The total number of missing (null) cells in the DataFrame.
    """
    validate_dataframe(df)
    
    num_missing_values = df.isnull().sum().sum()

    _verbose_comment(
        not_found_msg="There are no missing values.",
        found_msg=f"Number of missing values found: {num_missing_values}",
        value=num_missing_values,
        verbose_mode=verbose_mode
    )
  
    return num_missing_values


def count_duplicated_rows(df: DataFrame, verbose_mode: bool = True) -> int:
    """Count rows that duplicate an earlier row exactly.

    Args:
        df: The pandas DataFrame to check for duplicate rows.
        verbose_mode: If True, prints a summary message about the duplicates found.
            Defaults to True.

    Returns:
        The total number of duplicated rows in the DataFrame.
    """
    validate_dataframe(df)
    
    num_duplicated_rows = df.duplicated().sum()

    _verbose_comment(
        not_found_msg="There are no duplicated rows.",
        found_msg=f"Number of duplicated rows found: {num_duplicated_rows}",
        value=num_duplicated_rows,
        verbose_mode=verbose_mode
    )

    return num_duplicated_rows



def identify_missing_data(df: DataFrame) -> tuple[Series, list[str]]:
    """
    Identifies columns in a DataFrame that contain missing values.

    This function scans the provided DataFrame to locate any columns with null, 
    NaN, or missing data. It filters out columns that are already fully complete, 
    returning only the names and missing row counts of the columns that require 
    attention or data cleaning.

    Parameters
    ----------
    df : pd.DataFrame
        The pandas DataFrame to scan for missing values.

    Returns
    -------
    tuple[pd.Series, list[str]]
        A tuple containing two elements:
        - pd.Series: A series where the index represents the column names with 
          missing data, and the values represent the total count of missing rows.
        - list[str]: A list containing only the names of the columns that have 
          one or more missing values.

    Examples Usage
    --------
    >>> import pandas as pd
    >>> data = {'housing_median_age': [30, 40, None], 'total_bedrooms': [100, None, 150]}
    >>> df = pd.DataFrame(data)
    >>> counts, names = identify_missing_data(df)
    >>> print(names)
    ['housing_median_age', 'total_bedrooms']
    """
    validate_dataframe(df)

    missing_counts       = df.isnull().sum()
    only_missing         = missing_counts[missing_counts > 0]
    missing_column_names = only_missing.index.tolist()
    
    return only_missing, missing_column_names




def get_numerical_and_categorical_columns(
    df: DataFrame, 
    categorical_exclude_cols: list[str] | None = None,
    numerical_exclude_cols: list[str] | None = None,
) -> tuple[Index, Index]:
    """
    Splits a DataFrame into numerical and categorical columns.
    
    Args:
        - df (pd.Datatframe): The input pandas DataFrame.
        - categorical_exclude_cols (list) : A list of column names (e.g., targets, IDs) to remove from the category results.
        - numerical_exclude_cols (list) : A list of column names (e.g., targets, IDs) to remove from the numerical results
    
    Returns:
    - Tuple of (numerical_columns_index, categorical_columns_index)
    """
    validate_dataframe(df)

    if categorical_exclude_cols is not None and not isinstance(categorical_exclude_cols, list):
        raise TypeError(
                        "Categorical_exclude_cols is not a list ",
                        f"Got type {type(categorical_exclude_cols).__name__}" 
                        )

    if numerical_exclude_cols is not None and not isinstance(numerical_exclude_cols, list):
        raise TypeError(
                        "Numerical_exclude_cols is not a list ",   
                         f"numerical_exclude_cols got type {type(numerical_exclude_cols).__name__}"    
                        )

    numerical_columns   = df.select_dtypes(include=["number"]).columns
    categorical_columns = df.select_dtypes(exclude=["number"]).columns

    if categorical_exclude_cols:
        cat_to_drop         = categorical_columns.intersection(categorical_exclude_cols)
        categorical_columns = categorical_columns.drop(cat_to_drop)

    if numerical_exclude_cols:
        num_to_drop       = numerical_columns.intersection(numerical_exclude_cols)
        numerical_columns = numerical_columns.drop(num_to_drop)

    return numerical_columns, categorical_columns



def verbose_message(msg: str, verbose_mode: bool) -> None:
    """Print a conditional message based on verbose mode and a numeric value.

    Args:
        msg: The message to print if in verbose mode.
        verbose_mode: If False, the function returns immediately without printing.

    Raises:
        TypeError: If `verbose_mode` is not a boolean.
    """
    if not isinstance(verbose_mode, bool):
        raise TypeError(
            f"The parameter verbose_mode is not a bool. Got type {type(verbose_mode).__name__}"
        )    

    if not verbose_mode:
        return

    if verbose_mode:
        print(msg)
    