# Qwen3 Voice Embedding Server

![](./docs/PCA.webp)

This project used to wrapping `marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B-onnx` model as OpenAI-compatible API.

## Notice

Consider original design of OpenAI-compatible API not support multimodal (for `/embeddings` endpoint), I using RFC 2397 to transfer Audio file.

## Install

### Dokcer Compose

```yaml
services:
  qwen3-voice-embedding-server:
    image: ghcr.io/flyskypie/qwen3-voice-embedding-server:0.1.2
    devices:
      - /dev/dri/:/dev/dri/
    ports:
      - 8000:8000
    environment:
      - HF_HOME=/cache
    volumes:
      - ./cache:/cache
```

### Usage

```shell
curl -X POST http://localhost:8000/embeddings \
     -H "Content-Type: application/json" \
     -d '{
           "model": "ANY",
           "input": "data:audio/mpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
         }'
```
