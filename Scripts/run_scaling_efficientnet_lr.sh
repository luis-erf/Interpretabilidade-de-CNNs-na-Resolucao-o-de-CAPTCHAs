#!/bin/bash
#SBATCH --job-name=captcha-effnet-scaling-lr
#SBATCH --partition=short-simple
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out

# Refaz o escalonamento EfficientNet B0-B3 com uma taxa de aprendizado menor.
# No job 30276 (lr 1e-3), 5 das 8 rodadas não convergiram (val_loss parada em
# 5*ln(36) = 17,92, acurácia ~1/36). Suspeita: lr 1e-3 alto demais para as versões maiores.
#
# Todas as versões, inclusive a B0, usam o mesmo lr, para a comparação continuar justa.
# O CSV de saída leva o lr no nome e não sobrescreve o scaling_efficientnet_results.csv original.
#
# Estimativa: ~105-170 s por época conforme a versão; 8 treinos x até 30 épocas dá no máximo ~9 h.
#
# Uso (na raiz do repositório, com dataset_captcha/ e o ambiente já criados):
#   mkdir -p logs && sbatch Scripts/run_scaling_efficientnet_lr.sh
# Outro lr:
#   LR=1e-4 sbatch Scripts/run_scaling_efficientnet_lr.sh
# Sanity check rápido antes do job real:
#   TEST=--test bash Scripts/run_scaling_efficientnet_lr.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

source .venv/bin/activate

# Temporários fora do NFS da home (mesmo motivo do run_alexnet_efficientnet.sh)
for base in /dev/shm /tmp; do
    if TMP_JOB="$(mktemp -d "$base/captcha-XXXXXX" 2>/dev/null)"; then
        export TMPDIR="$TMP_JOB"
        trap 'rm -rf "$TMP_JOB"' EXIT
        break
    fi
done
echo "TMPDIR=${TMPDIR:-padrão}"

DADOS="${DADOS:-dataset_captcha}"
TEST="${TEST:-}"
LR="${LR:-3e-4}"
SAIDA="scaling_efficientnet_results_lr${LR}.csv"
nvidia-smi -L || true

if [[ -e "$SAIDA" && -z "$TEST" ]]; then
    echo "ERRO: $SAIDA já existe; apague ou mova antes para não sobrescrever resultados." >&2
    exit 1
fi

echo "=== Escalonamento EfficientNet B0-B3 | lr $LR ==="
python -m Scripts.scaling_efficientnet $TEST \
    --diretorio_dados "$DADOS" --learning_rate "$LR" --saida "$SAIDA"

echo "=== Concluído ==="
