from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import clean_data, scale_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=clean_data,
            inputs=["stroke_data"],
            outputs="cleaned_data",
            name="clean_data_node",
        ),
        node(
            func=scale_data,
            inputs=["cleaned_data"],
            outputs="scaled_data",
            name="scale_data_node",
        ),
    ])
