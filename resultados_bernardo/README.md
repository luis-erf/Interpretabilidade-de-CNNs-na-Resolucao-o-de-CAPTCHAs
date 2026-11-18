# Resultados – AlexNet e EfficientNet (Bernardo)

Experimentos rodados no cluster APUANA (CIn/UFPE), partição `short-simple`, GPU A100 (fatia MIG `1g.10gb`),
com `Scripts/run_alexnet_efficientnet.sh` (jobs SLURM 30275 e 30276, 13/11/2026).
Dataset: `dataset_captcha` (30.000 imagens, 5 caracteres, cópia do Drive), split 70/15/15 com seed 42.

## Conteúdo

| Caminho | O que é |
|---|---|
| `tune_results_alexnet.csv` | Melhores hiperparâmetros da AlexNet (Optuna, 50 trials, até 15 épocas cada) |
| `train_final_alexnet_efficientnet.csv` | Treino final (até 30 épocas, pesos da melhor época de validação) |
| `scaling_efficientnet_results.csv` | Escalonamento EfficientNet B0–B3, modos `profundidade_largura` e `composto` |
| `logs/` | Saídas dos jobs SLURM (`captcha-sanity-30272` é o teste rápido com `--test`) |
| `mapas_xai/` | Grad-CAM, Integrated Gradients e LIME em 10 imagens de teste por modelo (`Scripts/gerar_mapas.py`) |
| `checkpoints_finais/` | Pesos treinados (`.ckpt`). **Não vão para o git** (156 MB e 51 MB); ficam só localmente e no APUANA |

## Treino final

| Modelo | Parâmetros | Épocas | Acc. treino | Acc. val | Acc. teste |
|---|---|---|---|---|---|
| AlexNetCaptcha (tunada) | 13,0 M | 30 | 99,77% | 94,95% | **94,76%** |
| EfficientNetCaptcha (B0, padrão) | 4,2 M | 30 | 99,95% | 99,01% | **98,99%** |

- AlexNet: lr ≈ 7,9e-5, weight decay ≈ 5,0e-6, dropout 0,34, camadas densas 2048×2048, `pool_width` 9.
  Gap treino–val de 4,8 pontos (algum overfitting).
- EfficientNet usa os valores padrão: ela não está no espaço de busca do `tune_models.py`.

## Escalonamento EfficientNet (acurácia de teste)

| Versão | Profundidade + largura | Composto |
|---|---|---|
| B0 | 98,99% | 98,99% |
| B1 | ❌ 3,0% | 98,62% |
| B2 | ❌ 2,9% | ❌ 2,8% |
| B3 | ❌ 2,9% | ❌ 3,1% |

**As rodadas com ❌ não convergiram**: perda de validação parada em 17,92 (= 5 · ln 36, chute aleatório)
e acurácia ≈ 1/36. Não medem o efeito do escalonamento e precisam ser refeitas. Suspeita (não confirmada):
Adam com lr 1e-3 fixo para todas as versões é alto demais para as versões mais profundas.
