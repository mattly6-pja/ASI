from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import split_data_stroke, train_model_stroke, split_data_diabetes, train_model_diabetes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        # node(
        #     func=split_data,
        #     inputs=["scaled_data"],
        #     outputs="split_output",
        #     name="split_data_node",
        # ),
        # node(
        #     func=split_data_stroke,
        #     inputs=["preprocessed_data"],
        #     outputs="split_output",
        #     name="split_data_node",
        # ),
        # node(
        #     func=train_model_stroke,
        #     inputs="split_output",
        #     outputs="model",
        #     name="train_model_node",
        # ),
        node(
            func=split_data_diabetes,
            inputs=["preprocessed_data_diabetes"],
            outputs="split_output_diabetes",
            name="split_data_node_diabetes",
        ),
        node(
            func=train_model_diabetes,
            inputs="split_output_diabetes",
            outputs="model",
            name="train_model_node_diabetes",
        ),
    ])
