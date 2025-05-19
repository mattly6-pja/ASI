from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import clean_data, scale_data, preprocess_stroke_data, preprocess_diabetes_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        # node(
        #     func=preprocess_stroke_data,
        #     inputs=["stroke_data"],
        #     outputs="preprocessed_data",
        #     name="preprocess_data_node",
        # ),
        node(
            func=preprocess_diabetes_data,
            inputs=["diabetes_data"],
            outputs="preprocessed_data_diabetes",
            name="preprocess_data_node_diabetes",
        ),
    ])
