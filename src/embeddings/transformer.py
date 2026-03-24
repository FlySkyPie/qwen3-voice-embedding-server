import librosa
import torch
import intel_extension_for_pytorch as ipex
from typing import Annotated
from transformers import AutoModel, AutoProcessor
from fastapi import Depends

from src.embeddings.decorator import Singleton


@Singleton
class EmbeddingTransformer:
    def __init__(self):
        processor = AutoProcessor.from_pretrained(
            "marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B",
            trust_remote_code=True,
        )
        model = AutoModel.from_pretrained(
            "marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B",
            trust_remote_code=True,
        )
        model.eval()

        self.processor = processor
        self.model = model

    def embedding(self, audio_apth: str):
        audio, sr = librosa.load(audio_apth, sr=None, mono=True)
        inputs = self.processor(audio, sampling_rate=sr)

        with torch.no_grad():
            embedding = self.model(**inputs).last_hidden_state  # (1, 2048)
            return embedding[0]


def get_transformer():
    return EmbeddingTransformer.instance()


TransformerDep = Annotated[EmbeddingTransformer, Depends(get_transformer)]
