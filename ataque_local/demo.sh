#!/bin/bash
# Demonstração lado a lado: AlexNet (esquerda) x EfficientNet (direita), cada uma contra
# seu próprio servidor local, com o mesmo --seed (mesma sequência de CAPTCHAs).
#
# Uso (na raiz do repositório):
#   bash ataque_local/demo.sh            # 15 tentativas, 700 ms entre ações
#   bash ataque_local/demo.sh 30 1200    # tentativas, atraso em ms
set -euo pipefail
cd "$(dirname "$0")/.."

TENTATIVAS="${1:-15}"
LENTO="${2:-700}"
PY=".venv/bin/python"
CKPT="resultados_bernardo/checkpoints_finais"
SAIDA="${TMPDIR:-/tmp}/captcha_demo"

PIDS=()
trap 'kill "${PIDS[@]}" 2>/dev/null || true' EXIT

for porta in 5050 5051; do
    if curl -s -o /dev/null "http://127.0.0.1:$porta/"; then
        echo "A porta $porta já está em uso (servidor antigo?). Encerre-o e rode de novo." >&2
        exit 1
    fi
    "$PY" -m ataque_local.servidor --seed 42 --porta "$porta" >/dev/null 2>&1 &
    PIDS+=($!)
done
for porta in 5050 5051; do
    until curl -s -o /dev/null "http://127.0.0.1:$porta/"; do sleep 0.3; done
done

"$PY" -m ataque_local.robo --modelo AlexNetCaptcha --ckpt "$CKPT/AlexNetCaptcha.ckpt" \
    --url http://127.0.0.1:5050/ --tentativas "$TENTATIVAS" --visivel --lento "$LENTO" \
    --saida "$SAIDA" 2>&1 | sed -u 's/^/[AlexNet]      /' &
R1=$!
"$PY" -m ataque_local.robo --modelo EfficientNetCaptcha --ckpt "$CKPT/EfficientNetCaptcha.ckpt" \
    --url http://127.0.0.1:5051/ --tentativas "$TENTATIVAS" --visivel --lento "$LENTO" --janela_x 580 \
    --saida "$SAIDA" 2>&1 | sed -u 's/^/[EfficientNet] /' &
R2=$!
wait $R1 $R2
