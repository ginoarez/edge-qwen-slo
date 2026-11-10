# Servir Qwen3 4B con speculative decoding (DFlash) (hardware-agnostic)
# Requiere llama.cpp compilado con CUDA.

$Target = "Qwen3-4B-Q4_K_M.gguf"
$Draft  = "Qwen3-4B-DFlash.gguf"

llama-server `
  -m $Target `
  -md $Draft `
  --spec-type draft-dflash `
  --spec-draft-n-max 15 `
  -fa on --jinja `
  -c 8192 `
  --port 8080
