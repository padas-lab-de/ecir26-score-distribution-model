from typing import List, Union

from transformers import AutoModel

from sdm.model_wrappers import AbstractModelWrapper
from sdm.model_wrappers.utils import construct_document


class JinaV3Wrapper(AbstractModelWrapper):
    def __init__(
        self,
        pretrained_model_name="jinaai/jina-embeddings-v3",
    ):
        super().__init__()
        self.pretrained_model_name = pretrained_model_name
        self.encoder = None

    def _lazy_loading(self):
        self.encoder = AutoModel.from_pretrained(
            self.pretrained_model_name, trust_remote_code=True
        )
        self.encoder.cuda()
        self.encoder.eval()

    def get_instructions(self):
        if self.encoder is None:
            self._lazy_loading()
        return [
            self.encoder._task_instructions[x]
            for x in ["retrieval.query", "retrieval.passage"]
        ]

    def encode_queries(
        self,
        sentences: Union[str, List[str]],
        *args,
        **kwargs,
    ):
        if self.encoder is None:
            self._lazy_loading()
        return self.encoder.encode(sentences, *args, task="retrieval.query", **kwargs)

    def encode_corpus(
        self,
        sentences: Union[str, List[str]],
        *args,
        **kwargs,
    ):
        if self.encoder is None:
            self._lazy_loading()
        _sentences = [construct_document(sentence) for sentence in sentences]
        return self.encoder.encode(
            _sentences, *args, task="retrieval.passage", **kwargs
        )
