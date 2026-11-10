# Contributing

Gracias por contribuir a `edge-qwen-slo`.

## Desarrollo local

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q
```

## Pull requests

- Un cambio por PR.
- Incluye benchmark antes/después si tocas serving.
- Actualiza `results/README.md` si cambias números.
- Mantén el SLO documentado (≥100 tok/s, TTFT ≤500ms, VRAM ≤7.5GB).
```

Create requirements.txt with pytest.<|end|><|start|>Assistant to=bash<|channel|>analysis <|constrain|>xml<|message|><rfl:parameter name="command">
cat > "C:\Users\user\Documents\Default Project\edge-qwen-slo\requirements.txt" <<'EOF'
pytest>=8.0
pyyaml>=6.0
EOF