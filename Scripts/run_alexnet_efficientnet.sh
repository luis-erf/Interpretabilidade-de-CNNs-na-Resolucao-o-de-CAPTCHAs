#!/bin/bash
#SBATCH --job-name=captcha-alex-effnet
#SBATCH --partition=short-simple
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out

# Experimentos que faltam para AlexNet e EfficientNet:
#   1. Tunagem da AlexNet           -> tune_results_alexnet.csv
#   2. Treino final AlexNet + EffNet -> train_final_alexnet_efficientnet.csv + checkpoints_finais/
#   3. Escalonamento EfficientNet B0-B3 -> scaling_efficientnet_results.csv
#
# Uso (na raiz do repositório, com dataset_captcha/ e o ambiente já criados):
#   mkdir -p logs && sbatch Scripts/run_alexnet_efficientnet.sh
# Rodar só algumas etapas (ex: escalonamento em um job separado, em paralelo):
#   ETAPAS="tune train" sbatch Scripts/run_alexnet_efficientnet.sh
#   ETAPAS="scaling"    sbatch Scripts/run_alexnet_efficientnet.sh
# Sanity check rápido antes do job real:
#   TEST=--test bash Scripts/run_alexnet_efficientnet.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

source .venv/bin/activate

DADOS="${DADOS:-dataset_captcha}"
TEST="${TEST:-}"
ETAPAS="${ETAPAS:-tune train scaling}"
nvidia-smi -L || true

if [[ " $ETAPAS " == *" tune "* ]]; then
echo "=== 1/3 Tunagem AlexNet ==="
python -m Scripts.tune_models $TEST --modelos AlexNetCaptcha \
    --diretorio_dados "$DADOS" --saida tune_results_alexnet.csv
fi

if [[ " $ETAPAS " == *" train "* ]]; then
echo "=== 2/3 Treino final AlexNet + EfficientNet ==="
python -m Scripts.train_final $TEST --modelos AlexNetCaptcha EfficientNetCaptcha \
    --diretorio_dados "$DADOS" --params_csv tune_results_alexnet.csv \
    --pasta_ckpt checkpoints_finais --saida train_final_alexnet_efficientnet.csv
fi

if [[ " $ETAPAS " == *" scaling "* ]]; then
echo "=== 3/3 Escalonamento EfficientNet B0-B3 ==="
python -m Scripts.scaling_efficientnet $TEST \
    --diretorio_dados "$DADOS" --saida scaling_efficientnet_results.csv
fi

echo "=== Concluído ==="
