# Contributing to vllm-openvino

This is a single-developer open-source project. Contributions and feedback are welcome.

## Development Setup

### Prerequisites

- Python >= 3.10
- vLLM 0.19.1 (or target version)
- OpenVINO >= 2026.1.0
- Linux x86-64 with AVX2+ support

### Installation from Source

Clone the repository and install in development mode:

```bash
git clone https://github.com/belonghim/vllm-openvino.git
cd vllm-openvino

# Install vLLM with CPU-only PyTorch to avoid CUDA dependencies
VLLM_TARGET_DEVICE="empty" PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu" pip install -e .

# Uninstall incompatible Triton (may be pulled by vLLM)
pip uninstall -y triton
```

## Code Quality

### Validation (Required)

Before submitting a PR, ensure all Python files pass syntax validation:

```bash
# Single file
python3 -m py_compile vllm_openvino/platform.py

# All files
find vllm_openvino -name "*.py" -exec python3 -m py_compile {} +
```

**No exceptions**: All PRs must pass `py_compile` validation.

### Style Conventions

- Follow existing patterns in `vllm_openvino/` — do not introduce new style
- No type-ignore suppressions (`as any`, `@ts-ignore`, `# type: ignore`) except where explicitly documented
- Prefer explicit error handling over silent failures
- Docstrings for public functions (class methods, module-level functions)

## Testing

### Local Testing (Podman Source Mount)

Test your changes in isolation using a Podman container with source code mounted.

**Quick start** (see `docs/podman-testing.md` for detailed guide):

```bash
# Build container (if using local image)
podman build -f Containerfile -t vllm-openvino .

# Run with source mount (CPU limited to 8 for reproducible comparisons)
podman run --replace -d --name vllm-server -p 8080:8080 --cpus=8 \
  -v /path/to/vllm-openvino/vllm_openvino:/opt/app-root/vllm_openvino \
  -v /path/to/models:/models:Z \
  vllm-openvino \
  --port 8080 --model <model_dir> --max-model-len 4096

# Test API
curl -s http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "model", "prompt": "Hello", "max_tokens": 10}' | jq .
```

**Test matrix** (recommended):
- [ ] Single request (warm + cold)
- [ ] Concurrent requests (5+ simultaneous)
- [ ] Different device types (CPU, GPU if available)
- [ ] Different models (if applicable)

### Regression Testing

After changes to core modules (`platform.py`, `worker_v1/`, `model_executor/`), run the full test suite:

See `docs/podman-testing.md` for comprehensive testing steps.

## Commits

### Commit Policy

- **No Co-authored-by trailers** — all commits are recorded under the author's account
- **Rationale**: Clarity of contribution attribution; AI agents are tools, not contributors
- Use descriptive, actionable commit messages

### Commit Message Format

```
<type>: <subject>

<body (optional)>
```

**Types**: `fix`, `feat`, `docs`, `refactor`, `test`, `perf`, `chore`

**Examples**:
```
fix: null check in _is_stateful_model

refactor: consolidate statistics imports in model_loader/openvino.py

docs: add OpenVINO 2026.x deprecation warning
```

## Pull Request Process

1. **Create a feature branch** off `main`
   ```bash
   git checkout -b fix/my-issue-description
   ```

2. **Make changes** and pass validation:
   ```bash
   python3 -m py_compile vllm_openvino/**/*.py
   ```

3. **Test locally** (Podman source mount — see Testing section)

4. **Commit with clear message**:
   ```bash
   git commit -m "fix: describe your change"
   ```

5. **Push and open PR**:
   ```bash
   git push origin fix/my-issue-description
   ```

6. **PR Description**:
   - Link related issues: `Fixes #123`
   - Describe the motivation and implementation
   - Highlight any breaking changes
   - Note if changes affect performance or API

7. **Review and merge**:
   - Maintainer will review for:
     - Compatibility with vLLM and OpenVINO versions
     - Adherence to plugin boundary (no vLLM core modifications)
     - Alignment with project minimalism principles
     - Test coverage (Podman validation)

## Architecture Principles

Before proposing changes, review the design principles in `AGENTS.md`:

1. **Upstream Pattern Following** — Match vLLM's V1 engine patterns exactly
2. **Plugin Boundary Respect** — Use only vLLM plugin interfaces; don't modify core
3. **Minimalism** — No premature abstractions, no unused features
4. **Problem-Focused** — Solve real user problems, not hypothetical ones

See `docs/decisions.md` for rationales behind rejected features.

## Troubleshooting

### Import Errors

If `import openvino` fails:
- Verify OpenVINO >= 2026.1.0 is installed
- Try: `python3 -c "import openvino; print(openvino.__version__)"`

### Podman Build Failures

- Ensure `Containerfile` is in the repo root
- Check available disk space for image layers
- Try: `podman system prune` to clean dangling images

### Model Loading Issues

- Verify model is in OpenVINO IR format (`.xml` + `.bin`)
- Check file paths and permissions: `ls -la /path/to/model/`
- See `docs/compatibility.md` for supported architectures

## Questions or Feedback

- Open an issue on GitHub with detailed context
- Reference relevant documentation: `docs/`, `README.md`, `AGENTS.md`
- For security issues, contact the maintainer privately

---

**Thanks for contributing!** Your help makes vllm-openvino better.
