# Interpretabilidade de CNNs na Resolução de CAPTCHAs

Projeto de pesquisa e experimentação que avalia **Redes Neurais Convolucionais (CNNs)** na resolução automática de CAPTCHAs alfanuméricos e aplica **técnicas de explicabilidade (XAI)** para entender como cada arquitetura toma decisões caractere a caractere, além de validar o desempenho de ponta a ponta em um **ataque controlado em ambiente local**.

Link do dataset: https://drive.google.com/drive/folders/1-IQtCVqA4bQ3mQxz8aaJUgfi2H-jGksL?usp=drive_link

---

## O que o projeto faz

Os CAPTCHAs do projeto são imagens RGB de `160×60` pixels geradas pela biblioteca `captcha` (`ImageCaptcha`), contendo **5 caracteres** alfanuméricos (`A–Z` e `0–9`, totalizando **36 classes** por posição).

Em vez de segmentar os caracteres manualmente antes da classificação, cada modelo recebe a imagem inteira e prevê os 5 caracteres simultaneamente com saída no formato `(batch, 5, 36)`. O treinamento é feito com **PyTorch Lightning**, somando a `CrossEntropyLoss` ao longo das 5 posições e medindo a acurácia por caractere.

O fluxo do projeto cobre seis frentes:

1. **Geração e preparação do dataset (`Dataset/`)**: geração de 30.000 imagens rotuladas no nome do arquivo (`<TEXTO>_<INDICE>.png`), normalização padrão ImageNet e divisão determinística (`random_seed=42`) em **70% treino** (21.000), **15% validação** (4.500) e **15% teste** (4.500).
2. **Análise Exploratória de Dados (`eda/`)**: notebook e artefatos estáticos investigando integridade dos arquivos, balanceamento dos splits, distribuição de caracteres, contraste/tinta, faixas horizontais ocupadas por cada posição (com sobreposição em ~92,1% das imagens), glifos parecidos (como `0` e `O`) e padrões de erro.
3. **Modelagem com 6 arquiteturas de CNN (`Models/`)**: implementações adaptadas para imagens `60×160` com extrator convolucional exposto em `self.features`:
   - `AlexNetCaptcha` ([Models/AlexNet.py](Models/AlexNet.py))
   - `VGGCaptcha` ([Models/Vgg.py](Models/Vgg.py))
   - `ResNetCaptcha` ([Models/ResNet.py](Models/ResNet.py))
   - `InceptionCaptcha` ([Models/Inception.py](Models/Inception.py))
   - `EfficientNetCaptcha` ([Models/EfficientNet.py](Models/EfficientNet.py), suportando variantes `b0` a `b7`)
   - `ConvNeXtCaptcha` ([Models/ConvNext.py](Models/ConvNext.py))
4. **Tunagem, treino final e escalonamento (`Scripts/`)**:
   - Busca de hiperparâmetros com **Optuna** (`tune_models.py`).
   - Treino final com `EarlyStopping`, `ModelCheckpoint` e salvamento dos pesos da melhor época em `.ckpt` (`train_final.py`).
   - Experimento de escalonamento da `EfficientNet` (`b0` a `b3`), comparando o crescimento apenas em profundidade/largura (`60×160`) contra o escalonamento composto com aumento de resolução (`scaling_efficientnet.py`).
5. **Explicabilidade / XAI (`XAI/` e `Scripts/gerar_mapas.py`)**: geração de mapas de atribuição por posição (`0` a `4`) sobre imagens do conjunto de teste usando **Grad-CAM**, **Integrated Gradients** e **LIME**.
6. **Ataque controlado em servidor local (`ataque_local/`)**: servidor web local em **Flask** que gera CAPTCHAs inéditos em tempo real e um robô em **Playwright** que lê a imagem na página, infere a resposta com um checkpoint treinado e submete o formulário, medindo a **taxa de sucesso por CAPTCHA inteiro** (todos os 5 caracteres corretos) — métrica mais rigorosa que a acurácia por caractere.

---

## Organização do repositório

```text
.
├── Dataset/
│   ├── Gerar_dataset.py            # Gera imagens ImageCaptcha (160x60, 5 caracteres)
│   └── Datamodule.py               # CaptchaDataset e CaptchaDataModule (split 70/15/15)
├── dataset_captcha/                # Pasta padrão com as 30.000 imagens .png (ignorada no Git)
├── Models/
│   ├── AlexNet.py                  # AlexNetCaptcha
│   ├── ConvNext.py                 # ConvNeXtCaptcha
│   ├── EfficientNet.py             # EfficientNetCaptcha (b0–b7)
│   ├── Inception.py                # InceptionCaptcha
│   ├── ResNet.py                   # ResNetCaptcha
│   └── Vgg.py                      # VGGCaptcha
├── Scripts/
│   ├── tune_models.py              # Tunagem de hiperparâmetros com Optuna -> CSV
│   ├── train_final.py              # Treino final, avaliação e exportação de checkpoints (.ckpt)
│   ├── scaling_efficientnet.py     # Comparação EfficientNet B0–B3 (profundidade/largura vs. composto)
│   └── gerar_mapas.py              # Gera painéis Grad-CAM, Integrated Gradients e LIME
├── XAI/
│   └── explainers.py               # Implementação de grad_cam, integrated_gradients e lime
├── eda/
│   ├── analise_exploratoria.ipynb  # Notebook completo da EDA
│   ├── 01_*.png ... 10_*.png       # Gráficos gerados pela análise exploratória
│   └── *.csv                       # Resumos e estatísticas extraídas (resumo_eda.csv, etc.)
├── ataque_local/
│   ├── servidor.py                 # Servidor Flask local (127.0.0.1:5050)
│   ├── robo.py                     # Robô Playwright que resolve CAPTCHAs usando um .ckpt
│   ├── demo.sh                     # Demonstração lado a lado (AlexNet vs. EfficientNet)
│   ├── README.md                   # Detalhes e resultados do ataque local
│   └── resultados/                 # CSVs de tentativas e imagens de erro por modelo
├── requirements.txt                # Dependências Python do projeto
├── LICENSE                         # Licença MIT
└── README.md
```

---

## Como rodar o código

> Execute todos os comandos a partir da **raiz do repositório** usando `python -m ...` para que os imports entre pastas funcionem corretamente.

### 1. Ambiente e dependências

Crie um ambiente virtual e instale os pacotes listados em [requirements.txt](requirements.txt):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Se for rodar o módulo de ataque controlado (`ataque_local/`), instale também o navegador Chromium utilizado pelo Playwright:

```bash
playwright install chromium
```

### 2. Obter ou gerar o dataset

Você pode baixar o dataset pronto pelo link do Google Drive no topo deste README e extrair as imagens em `dataset_captcha/`, ou gerar um novo conjunto de 30.000 imagens localmente:

```bash
python -m Dataset.Gerar_dataset
```

### 3. Análise exploratória (EDA)

Os gráficos e tabelas já estão salvos em `eda/`. Para reproduzir ou inspecionar a análise, abra o notebook [eda/analise_exploratoria.ipynb](eda/analise_exploratoria.ipynb) em um ambiente Jupyter (ou VS Code).

### 4. Tunagem de hiperparâmetros (Optuna)

Para validar rapidamente o pipeline (2 mini-lotes, 1 época, 2 trials):

```bash
python -m Scripts.tune_models --test
```

Para executar a tunagem completa e salvar os melhores hiperparâmetros em CSV:

```bash
python -m Scripts.tune_models \
    --modelos AlexNetCaptcha VGGCaptcha ResNetCaptcha InceptionCaptcha EfficientNetCaptcha ConvNextCaptcha \
    --n_trials 50 \
    --max_epochs 15 \
    --patience 3 \
    --saida tune_results.csv
```

> Sem `--modelos`, o script tuna apenas `VGGCaptcha` e `ConvNextCaptcha` por padrão. Outras opções úteis: `--diretorio_dados`, `--batch_size`, `--num_workers`, `--seed`.

### 5. Treino final e geração de checkpoints

Para checagem rápida (`--test`, sem salvar checkpoint):

```bash
python -m Scripts.train_final --test
```

Para treinar os modelos e salvar os melhores pesos (`.ckpt`) em `checkpoints_finais/` e as métricas em `train_final_results.csv`:

```bash
python -m Scripts.train_final \
    --params_csv tune_results.csv \
    --pasta_ckpt checkpoints_finais \
    --max_epochs 30 \
    --patience 5 \
    --saida train_final_results.csv
```

- Se `--params_csv` não for informado (ou se um modelo não estiver no CSV), são usados os hiperparâmetros padrão da classe.
- Use `--modelos <NomeModelo ...>` para treinar apenas arquiteturas específicas. Os nomes aceitos são: `AlexNetCaptcha`, `VGGCaptcha`, `ConvNextCaptcha`, `EfficientNetCaptcha`, `ResNetCaptcha` e `InceptionCaptcha`.

### 6. Experimento de escalonamento da EfficientNet (`b0` a `b3`)

Teste rápido:

```bash
python -m Scripts.scaling_efficientnet --test
```

Execução completa nos modos `profundidade_largura` e `composto`:

```bash
python -m Scripts.scaling_efficientnet --saida scaling_efficientnet_results.csv
```

É possível filtrar versões e modos com `--versoes b0 b1 b2 b3`, `--modos profundidade_largura composto` e ajustar `--learning_rate`, `--max_epochs` e `--batch_size`.

### 7. Gerar mapas de explicabilidade (XAI)

Requer um checkpoint `.ckpt` gerado pelo `train_final.py` (ou baixado da release `checkpoints-v1` do repositório, quando disponível).

Teste rápido (1 imagem do conjunto de teste e 10 perturbações no LIME):

```bash
python -m Scripts.gerar_mapas --test \
    --modelo EfficientNetCaptcha \
    --ckpt checkpoints_finais/EfficientNetCaptcha.ckpt
```

Execução completa (10 imagens do conjunto de teste, salvando figuras em `mapas/<modelo>/`):

```bash
python -m Scripts.gerar_mapas \
    --modelo EfficientNetCaptcha \
    --ckpt checkpoints_finais/EfficientNetCaptcha.ckpt \
    --n_imagens 10 \
    --metodos gradcam ig lime \
    --amostras_lime 1000 \
    --saida mapas
```

### 8. Ataque controlado em servidor local (`ataque_local/`)

> **Uso restrito:** o robô aceita exclusivamente URLs em `127.0.0.1` ou `localhost`, funcionando apenas contra o servidor local deste projeto.

Em um terminal, inicie o servidor local (`http://127.0.0.1:5050`):

```bash
python -m ataque_local.servidor --seed 42
```

Em outro terminal, execute o robô:

```bash
# Avaliação em modo headless (navegador invisível)
python -m ataque_local.robo \
    --modelo EfficientNetCaptcha \
    --ckpt checkpoints_finais/EfficientNetCaptcha.ckpt \
    --tentativas 100

# Demonstração visual (abre o Chromium e desacelera as ações em ms)
python -m ataque_local.robo \
    --modelo AlexNetCaptcha \
    --ckpt checkpoints_finais/AlexNetCaptcha.ckpt \
    --tentativas 10 --visivel --lento 600
```

As tentativas ficam registradas em `ataque_local/resultados/<modelo>/tentativas.csv` e as imagens em que o modelo errou são salvas em `ataque_local/resultados/<modelo>/erros/`.

Também há o script `ataque_local/demo.sh` para demonstração lado a lado entre `AlexNetCaptcha` e `EfficientNetCaptcha` (configure nele ou garanta os checkpoints no caminho definido na variável `CKPT` dentro do script). Consulte [ataque_local/README.md](ataque_local/README.md) para mais detalhes e resultados coletados.

---

## Licença

Este projeto está licenciado sob a [Licença MIT](LICENSE).
