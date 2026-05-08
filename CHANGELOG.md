# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive code quality audit and improvements
- CHANGELOG.md for version tracking
- CONTRIBUTING.md for developer workflow

### Changed
- Reorganized import order in platform.py for clarity

### Fixed
- Null reference handling in _is_stateful_model()

## [0.19.1] - 2026-05-08

### Added
- Initial release as vLLM 0.19.1 plugin
- OpenVINO backend for vLLM with V1 engine support
- CPU and GPU device support
- PagedAttention optimization for attention-only models
- Stateful model support (Gemma-4, Qwen3.5 hybrid)
- KV cache quantization (u8, i8, f16, bf16, f32)
- Performance tuning environment variables for CPU deployments
- Podman-based testing workflow

### Removed
- V0 engine support (complete removal)

### Notes
- Single-developer project (belonghim)
- Minimal test infrastructure by design (vLLM runtime required)
- Focus on upstream vLLM compatibility patterns
