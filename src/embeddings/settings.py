from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    hf_repo : str = "marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B-onnx"
    hf_file : str = "speaker_encoder_fp32.onnx"


settings = Settings()