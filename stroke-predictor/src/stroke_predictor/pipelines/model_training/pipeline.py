"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import split_data, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_data,
            inputs=["scaled_data"],
            outputs=["X_resampled", "X_test", "y_resampled", "y_test"],
            name="split_data_node",
        ),
        node(
            func=train_model,
            inputs=["X_resampled", "X_test", "y_resampled", "y_test"],
            outputs="model",
            name="train_model_node",
        ),
    ])
