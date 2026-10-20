# Fails if a vllm.v1 symbol imported by the plugin no longer resolves. Run from the repo root inside the target vLLM environment.
import ast, importlib, pathlib, sys
bad = []
for f in pathlib.Path("vllm_openvino").rglob("*.py"):
    for n in ast.walk(ast.parse(f.read_text())):
        if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("vllm.v1"):
            try:
                m = importlib.import_module(n.module)
            except Exception as e:
                bad.append(f"{f}: {n.module}: {e!r}")
                continue
            for a in n.names:
                if not hasattr(m, a.name):
                    try:
                        importlib.import_module(f"{n.module}.{a.name}")
                    except Exception:
                        bad.append(f"{f}: {n.module}.{a.name}")
print("\n".join(bad) or "all vllm.v1 imports resolve")
sys.exit(1 if bad else 0)
