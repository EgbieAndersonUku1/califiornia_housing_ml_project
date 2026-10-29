"""
This module provides structural type definitions, type guards, and custom 
exceptions for handling Scikit-Learn style data transformers across the pipeline.
"""

from typing import Any, Protocol, TypeGuard

import numpy as np
import pandas as pd


class ScikitTransformer(Protocol):
    """
    A structural blueprint (Protocol) defining the minimum required interface 
    for a Scikit-Learn style data transformer.

    Any class that implements these exact methods with matching signatures 
    will automatically satisfy this type hint.
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> ScikitTransformer:
        """Fit the transformer to the input training data layout."""
        ...

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform the input data layout into its processed state."""
        ...


def is_transformer(obj: Any) -> TypeGuard[ScikitTransformer]:
    """
    Validates whether an object adheres to the ScikitTransformer protocol outline.

    Args:
        obj (Any): The object instance to check.

    Returns:
        TypeGuard[ScikitTransformer]: True if the object has both callable 'fit' 
        and 'transform' methods, False otherwise.
    """
    return (
        hasattr(obj, "fit") and callable(obj.fit) and
        hasattr(obj, "transform") and callable(obj.transform)
    )


