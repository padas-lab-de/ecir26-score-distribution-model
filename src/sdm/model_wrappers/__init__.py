from typing import Tuple

from .AbstractModelWrapper import AbstractModelWrapper

# from .E5MultilingualWrapper import E5MultilingualWrapper
# from .JinaV3Wrapper import JinaV3Wrapper
from .SnowflakeWrapper import SnowflakeWrapper


def get_model_wrapper(model_name: str) -> Tuple[AbstractModelWrapper, int]:
    """
    Attempts to return the model wrapper used to create embeddings to be evaluated.

    Args:
        model_name: The model to use.

    Returns:
        The model wrapper and its maximum embedding dimension, if implemented.
    """
    # if model_name == "jinav3":
    #     return JinaV3Wrapper(), 1024
    # elif model_name == "e5":
    #     return E5MultilingualWrapper(), 1024
    if model_name == "snowflakev2":
        return SnowflakeWrapper(), 768
    elif model_name == "snowflake":
        return SnowflakeWrapper("Snowflake/snowflake-arctic-embed-m"), 768
    else:
        raise NotImplementedError(f"Model {model_name} not supported!")
