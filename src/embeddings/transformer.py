import librosa
import numpy as np
import onnxruntime as ort
from typing import Annotated
from fastapi import Depends
from huggingface_hub import hf_hub_download

from src.embeddings.decorator import Singleton


@Singleton
class EmbeddingTransformer:
    def __init__(self):
        model_path = hf_hub_download(
            repo_id="marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B-onnx",
            filename="speaker_encoder_fp32.onnx",
        )

        session = ort.InferenceSession(
            model_path,
            providers=["WebGpuExecutionProvider"],
        )

        self.session = session

    def embedding(self, audio_path: str):
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=24000,
            n_fft=1024,
            hop_length=256,
            n_mels=128,
            fmin=0,
            fmax=12000,
        )

        mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))
        mel = mel.T[np.newaxis, ...]  # (1, time, 128)

        # Run inference
        embedding = self.session.run(None, {"mel_spectrogram": mel.astype(np.float32)})[0]
        return embedding[0]


def get_transformer():
    return EmbeddingTransformer.instance()


TransformerDep = Annotated[EmbeddingTransformer, Depends(get_transformer)]
