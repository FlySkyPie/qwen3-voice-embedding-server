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
        self.device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
        print("xpu" if torch.xpu.is_available() else "cpu")

        processor = AutoProcessor.from_pretrained(
            "marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B",
            trust_remote_code=True,
        )
        model = AutoModel.from_pretrained(
            "marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B",
            trust_remote_code=True,
            dtype=torch.bfloat16,
        ).to(self.device)
        model.eval()

        self.processor = processor
        self.model = model

        self.model = ipex.optimize(self.model, dtype=torch.bfloat16)
        self.model.eval()

    def embedding(self, audio_apth: str):
        audio, sr = librosa.load(audio_apth, sr=None, mono=True)
        inputs = self.processor(audio, sampling_rate=sr,return_tensors="pt")

        # inputs = {k: v.to(device=self.device, dtype=torch.bfloat16) for k, v in inputs.items()}
        # inputs = {k: v.to(device=self.device) for k, v in inputs.items()}
        inputs = {
            k: v.to(device=self.device, dtype=torch.bfloat16) 
            if torch.is_floating_point(v) else v.to(device=self.device)
            for k, v in inputs.items()
        }


        with torch.no_grad():
            # with torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
            with torch.amp.autocast('xpu', enabled=True, dtype=torch.bfloat16):
                print("start inference")
                embedding = self.model(**inputs).last_hidden_state  # (1, 2048)
                return embedding[0]


def get_transformer():
    return EmbeddingTransformer.instance()


TransformerDep = Annotated[EmbeddingTransformer, Depends(get_transformer)]
