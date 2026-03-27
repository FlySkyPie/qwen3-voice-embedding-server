# Qwen3 Voice Embedding Server

This project used to wrapping `marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B-onnx` model as OpenAI-compatible API.

## Notice

Consider original design of OpenAI-compatible API not support multimodal (for `/embeddings` endpoint), I using RFC 2397 to transfer Audio file.
