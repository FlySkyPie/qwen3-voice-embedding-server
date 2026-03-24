import librosa
import torch
import intel_extension_for_pytorch as ipex
from transformers import AutoModel, AutoProcessor

def embedding_audio(audio_apth: str):
    processor = AutoProcessor.from_pretrained(
        "marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B", trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        "marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B", trust_remote_code=True,
    )
    model.eval()

    audio, sr = librosa.load(audio_apth, sr=None, mono=True)
    inputs = processor(audio, sampling_rate=sr)

    with torch.no_grad():
        embedding = model(**inputs).last_hidden_state  # (1, 2048)
        print("embedding",embedding)
        return embedding[0]