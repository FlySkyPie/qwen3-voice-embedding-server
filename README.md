# Qwen3 Voice Embedding Server

This project used to wrapping `marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B` model as OpenAI-compatible API.

## Notice

This project is customized for my environment, it's using IPEX (Intel-Extension-for-PyTorch) instead of normal ML GPU SDK CUDA.

Consider original design of OpenAI-compatible API not support multimodal, I using RFC 2397 to transfer Audio file.
