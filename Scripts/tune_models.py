import argparse
import logging
import warnings

import optuna
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch

from Dataset.Datamodule import CaptchaDataModule
from Models.AlexNet import AlexNetCaptcha
from Models.ConvNext import ConvNeXtCaptcha
from Models.Inception import InceptionCaptcha
from Models.ResNet import ResNetCaptcha
from Models.Vgg import VGGCaptcha

# 1. Silencia avisos de deprecamento e warnings gerais do Python/Lightning
warnings.filterwarnings("ignore")

# 2. Silencia os logs informativos do Optuna (exibe apenas se houver ERRO)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 3. Silencia os logs informativos do PyTorch Lightning
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

"""
======================================================================================
SCRIPT DE TUNAGEM DE HIPERPARÂMETROS - MODELOS DE CAPTCHA
======================================================================================

Este script realiza a busca automatizada dos melhores hiperparâmetros para as 
arquiteturas de Redes Neurais Convolucionais utilizando Optuna e PyTorch Lightning.

--------------------------------------------------------------------------------------
COMO EXECUTAR:
--------------------------------------------------------------------------------------
1. Modo Normal (Tunagem Completa):
   Executa a busca completa no dataset inteiro utilizando dezenas de iterações.
   Recomendado rodar em GPU (ex: cluster APUANA).
   Comando: python -m Scripts.tune_models

2. Modo de Teste (Sanity Check):
   Limita o dataset a 2 mini-lotes, executa apenas 1 época e 2 iterações no Optuna.
   Ideal para debugar o código em segundos e verificar se há erros de sintaxe 
   antes de comprometer horas de processamento.
   Comando: python -m Scripts.tune_models --test

3. Opções úteis:
   --modelos ResNetCaptcha InceptionCaptcha   Tuna apenas os modelos escolhidos
   --diretorio_dados caminho/da/pasta         Pasta do dataset (padrão: dataset_captcha)
   --saida tune_results.csv                   Nome/caminho do CSV de saída
   --n_trials 50                              Quantidade de iterações por modelo
   --batch_size 32                            Tamanho do lote
   --num_workers 2                            Número de workers no DataModule
   --max_epochs 15                            Máximo de épocas por trial
   --patience 3                               Paciência do EarlyStopping
   --seed 42                                  Semente aleatória global
======================================================================================
"""

# =====================================================================
# 1. ESPAÇO DE BUSCA E MAPEAMENTO DE MODELOS
# =====================================================================
# Nomes alinhados exatamente com a chave do dicionário MODELOS do train_final.py
SEARCH_SPACE = {
    "AlexNetCaptcha": {
        # Faixa menor que a dos demais modelos, pois a AlexNet não tem BatchNorm
        "learning_rate": {"type": "log_float", "low": 1e-5, "high": 1e-3},
        "weight_decay": {"type": "log_float", "low": 1e-6, "high": 1e-2},
        "dropout_rate": {"type": "float", "low": 0.2, "high": 0.6},
        "fc_sizes_str": {"type": "categorical", "choices": ["1024_1024", "2048_2048", "4096_4096"]},
        "pool_width": {"type": "categorical", "choices": [3, 9]}
    },
    "VGGCaptcha": {
        "learning_rate": {"type": "log_float", "low": 1e-4, "high": 1e-2},
        "dropout_rate": {"type": "float", "low": 0.2, "high": 0.6},
        "fc_sizes_str": {"type": "categorical", "choices": ["512", "1024", "512_512"]}
    },
    "ConvNextCaptcha": {  # 'x' minúsculo para alinhar com train_final.py
        "learning_rate": {"type": "log_float", "low": 1e-4, "high": 1e-2},
        "weight_decay": {"type": "log_float", "low": 1e-5, "high": 1e-2},
        "expansion_ratio": {"type": "categorical", "choices": [2, 4, 6]}
    },
    "ResNetCaptcha": {
        "learning_rate": {"type": "log_float", "low": 1e-4, "high": 1e-2},
        "weight_decay": {"type": "log_float", "low": 1e-6, "high": 1e-2},
        "dropout_rate": {"type": "float", "low": 0.1, "high": 0.5},
        "pool_width": {"type": "categorical", "choices": [1, 4, 5]}
    },
    "InceptionCaptcha": {
        "learning_rate": {"type": "log_float", "low": 1e-4, "high": 1e-2},
        "weight_decay": {"type": "log_float", "low": 1e-6, "high": 1e-2},
        "dropout_rate": {"type": "float", "low": 0.2, "high": 0.6},
        "pool_width": {"type": "categorical", "choices": [1, 4, 5]}
    }
}

MODELOS_MAP = {
    "AlexNetCaptcha": AlexNetCaptcha,
    "VGGCaptcha": VGGCaptcha,
    "ConvNextCaptcha": ConvNeXtCaptcha,
    "ResNetCaptcha": ResNetCaptcha,
    "InceptionCaptcha": InceptionCaptcha,
}


def parse_fc_sizes(fc_str):
    """Função auxiliar para converter a escolha da camada densa da VGG de string para lista."""
    return [int(x) for x in str(fc_str).split('_')]


def objective(trial, model_name, datamodule, args):
    """
    Função objetivo que o Optuna tenta otimizar (minimizar a perda de validação).
    """
    params = {}
    for param_name, config in SEARCH_SPACE[model_name].items():
        p_type = config["type"]
        if p_type == "log_float":
            params[param_name] = trial.suggest_float(param_name, config["low"], config["high"], log=True)
        elif p_type == "float":
            params[param_name] = trial.suggest_float(param_name, config["low"], config["high"])
        elif p_type == "categorical":
            params[param_name] = trial.suggest_categorical(param_name, config["choices"])

    # Cópia para instanciação do modelo sem remover a chave 'fc_sizes_str' do trial.params
    model_kwargs = params.copy()

    if model_name in ("AlexNetCaptcha", "VGGCaptcha"):
        model_kwargs["fc_sizes"] = parse_fc_sizes(model_kwargs.pop("fc_sizes_str"))
        model = MODELOS_MAP[model_name](**model_kwargs)
    elif model_name in MODELOS_MAP:
        model = MODELOS_MAP[model_name](**model_kwargs)
    else:
        raise ValueError(f"Modelo {model_name} não reconhecido.")

    # Configurar Early Stopping com a paciência vinda dos argumentos
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        mode='min',
        verbose=False
    )

    trainer_kwargs = {
        "max_epochs": args.max_epochs if not args.test else 1,
        "callbacks": [early_stop],
        "enable_progress_bar": False,
        "logger": False,
        "enable_model_summary": False
    }

    if args.test:
        trainer_kwargs["limit_train_batches"] = 2
        trainer_kwargs["limit_val_batches"] = 2

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, datamodule=datamodule)

    best_val_loss = early_stop.best_score
    if best_val_loss is None:
        return float('inf')

    return best_val_loss.item()


def run_tuning(models_to_tune, datamodule, args):
    """
    Executa a tunagem para uma lista de modelos e retorna um dicionário com os melhores resultados.
    """
    melhores_resultados = {}
    sampler_deterministico = optuna.samplers.TPESampler(seed=args.seed)

    for model_name in models_to_tune:
        print(f"\n{'='*50}")
        print(f" Iniciando Tunagem para: {model_name} ")
        print(f"{'='*50}")

        study = optuna.create_study(
            direction="minimize",
            study_name=f"Tuning_{model_name}",
            sampler=sampler_deterministico
        )

        trials_to_run = 2 if args.test else args.n_trials

        study.optimize(
            lambda trial: objective(trial, model_name, datamodule, args),
            n_trials=trials_to_run
        )

        melhores_resultados[model_name] = study.best_params

        print(f"\n--> Tunagem de {model_name} concluída!")
        print(f"--> Menor Val Loss: {study.best_value:.4f}")
        print(f"--> Melhores Hiperparâmetros: {study.best_params}")

    return melhores_resultados


if __name__ == "__main__":
    # =====================================================================
    # CONTROLE PRINCIPAL VIA TERMINAL (ARGPARSE COMPLETO)
    # =====================================================================
    parser = argparse.ArgumentParser(description="Script de Tunagem de Modelos para CAPTCHA")
    parser.add_argument('--test', action='store_true', help="Ativa o modo de teste rápido (Sanity Check)")
    parser.add_argument('--diretorio_dados', default="dataset_captcha", help="Pasta com as imagens do dataset")
    parser.add_argument('--modelos', nargs='+', choices=list(SEARCH_SPACE), help="Tuna apenas os modelos escolhidos")
    parser.add_argument('--saida', default="tune_results.csv", help="Caminho do CSV de saída")
    parser.add_argument('--n_trials', type=int, default=50, help="Quantidade de tentativas do Optuna por modelo")
    parser.add_argument('--batch_size', type=int, default=32, help="Tamanho do lote (batch size)")
    parser.add_argument('--num_workers', type=int, default=2, help="Número de workers do DataModule")
    parser.add_argument('--max_epochs', type=int, default=15, help="Número máximo de épocas por trial")
    parser.add_argument('--patience', type=int, default=3, help="Paciência do EarlyStopping")
    parser.add_argument('--seed', type=int, default=42, help="Semente aleatória global (seed)")
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)

    modelos_para_tunar = args.modelos if args.modelos else ["VGGCaptcha", "ConvNextCaptcha"]

    print(f"Modo de Teste (Sanity Check): {'ATIVADO' if args.test else 'DESATIVADO'}")

    datamodule = CaptchaDataModule(
        diretorio_dados=args.diretorio_dados,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_seed=args.seed
    )

    resultados_finais = run_tuning(
        models_to_tune=modelos_para_tunar,
        datamodule=datamodule,
        args=args
    )

    print("\n\n" + "#"*50)
    print(" RESUMO FINAL DA TUNAGEM ")
    print("#"*50)
    for modelo, hiperparametros in resultados_finais.items():
        print(f"\nModelo: {modelo}")
        for param, valor in hiperparametros.items():
            print(f"  - {param}: {valor}")

    # =====================================================================
    # EXPORTAÇÃO PARA CSV (FORMATO ESPERADO PELO TRAIN_FINAL.PY)
    # =====================================================================
    # As chaves principais (nomes dos modelos) viram as COLUNAS.
    # Os nomes dos hiperparâmetros viram as LINHAS (índice da primeira coluna).
    df = pd.DataFrame(resultados_finais)
    df.to_csv(args.saida)
    print(f"\n[OK] Resultados salvos com sucesso em '{args.saida}'")