#!/bin/bash
#SBATCH --job-name=captcha-effnet-tune
#SBATCH --partition=short-simple
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out

# Tunagem da EfficientNet-B0 e treino final com os hiperparâmetros encontrados:
#   1. Tunagem (Optuna, mesmas configurações da AlexNet) -> tune_results_efficientnet.csv
#   2. Treino final com os melhores hiperparâmetros     -> train_final_efficientnet_tunada.csv
#                                                          + checkpoints_efficientnet_tunada/
# O checkpoint vai para uma pasta separada para não sobrescrever o
# checkpoints_finais/EfficientNetCaptcha.ckpt (sem tunagem) usado nos mapas de XAI.
#
# Estimativa: ~105 s por época na fatia MIG 1g.10gb (treino final de 13/11), então
# 50 trials x até 15 épocas dá no máximo ~22 h, mais ~1 h de treino final.
#
# Uso (na raiz do repositório, com dataset_captcha/ e o ambiente já criados):
#   mkdir -p logs && sbatch Scripts/run_tune_efficientnet.sh
# Rodar só uma etapa:
#   ETAPAS="train" sbatch Scripts/run_tune_efficientnet.sh
# Menos trials:
#   N_TRIALS=30 sbatch Scripts/run_tune_efficientnet.sh
# Sanity check rápido antes do job real:
#   TEST=--test bash Scripts/run_tune_efficientnet.sh

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
ETAPAS="${ETAPAS:-tune train}"
N_TRIALS="${N_TRIALS:-50}"
nvidia-smi -L || true

# Não sobrescreve resultados de rodadas anteriores
if [[ -z "$TEST" ]]; then
    for saida in tune_results_efficientnet.csv train_final_efficientnet_tunada.csv; do
        etapa=$([[ $saida == tune_* ]] && echo tune || echo train)
        if [[ " $ETAPAS " == *" $etapa "* && -e "$saida" ]]; then
            echo "ERRO: $saida já existe; apague ou mova antes para não sobrescrever resultados." >&2
            exit 1
        fi
    done
fi

if [[ " $ETAPAS " == *" tune "* ]]; then
echo "=== 1/2 Tunagem EfficientNet-B0 ($N_TRIALS trials) ==="
python -m Scripts.tune_models $TEST --modelos EfficientNetCaptcha \
    --diretorio_dados "$DADOS" --n_trials "$N_TRIALS" \
    --saida tune_results_efficientnet.csv
# O Trainer da tunagem grava um .ckpt por trial em checkpoints/, que nenhuma etapa usa
rm -rf checkpoints
fi

if [[ " $ETAPAS " == *" train "* ]]; then
echo "=== 2/2 Treino final EfficientNet-B0 tunada ==="
python -m Scripts.train_final $TEST --modelos EfficientNetCaptcha \
    --diretorio_dados "$DADOS" --params_csv tune_results_efficientnet.csv \
    --pasta_ckpt checkpoints_efficientnet_tunada \
    --saida train_final_efficientnet_tunada.csv
fi

echo "=== Concluído ==="
