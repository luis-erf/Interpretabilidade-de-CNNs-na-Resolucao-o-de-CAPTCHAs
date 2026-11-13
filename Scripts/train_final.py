import argparse
import os
import shutil
import tempfile
import time
import warnings
import logging

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from Dataset.Datamodule import CaptchaDataModule
from Models.AlexNet import AlexNetCaptcha
from Models.ConvNext import ConvNeXtCaptcha
from Models.EfficientNet import EfficientNetCaptcha
from Models.Vgg import VGGCaptcha
from Models.ResNet import ResNetCaptcha
from Models.Inception import InceptionCaptcha


# Silencia avisos e logs informativos (mesmo padrão do tune_models.py)
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

"""
======================================================================================
TREINO FINAL - MODELOS DE CAPTCHA (baseline de acurácia + checkpoints para o XAI)
======================================================================================

Treina cada modelo com os melhores hiperparâmetros encontrados na tunagem e avalia
treino, validação e teste com os pesos da melhor época de validação. O checkpoint
dessa melhor época é guardado em `--pasta_ckpt`, pois a Fase 3 (Grad-CAM, Integrated
Gradients e LIME) consome os modelos já treinados sem re-treinar.

--------------------------------------------------------------------------------------
COMO EXECUTAR:
--------------------------------------------------------------------------------------
1. Treino completo (recomendado em GPU, ex: cluster APUANA):
   python -m Scripts.train_final --diretorio_dados dataset_captcha_2/dataset_captcha

2. Modo de Teste (Sanity Check): 2 mini-lotes e 1 época por treino.
   python -m Scripts.train_final --test --diretorio_dados dataset_captcha_2/dataset_captcha

3. Opções úteis:
   --modelos ResNetCaptcha InceptionCaptcha   Treina apenas os modelos escolhidos
   --params_csv tune_results.csv              CSV gerado pelo tune_models.py. Sem ele,
                                              usa os valores padrão de cada modelo.
   --max_epochs 30   --patience 5   --batch_size 32   --saida arquivo.csv

O CSV de resultados é atualizado a cada treino (mesmas colunas do scaling_efficientnet.py).
======================================================================================
"""

RANDOM_SEED = 42

MODELOS = {
    "AlexNetCaptcha": AlexNetCaptcha,
    "VGGCaptcha": VGGCaptcha,
    "ConvNextCaptcha": ConvNeXtCaptcha,
    "EfficientNetCaptcha": EfficientNetCaptcha,
    "ResNetCaptcha": ResNetCaptcha,
    "InceptionCaptcha": InceptionCaptcha,
}

# Hiperparâmetros que o pandas converte para float ao ler o CSV (por causa dos NaN)
PARAMS_INTEIROS = {"pool_width", "expansion_ratio"}


def carregar_hiperparametros(caminho_csv, modelo):
    """Lê os melhores hiperparâmetros de um modelo no CSV da tunagem (uma coluna por modelo)."""
    if caminho_csv is None:
        return {}

    df = pd.read_csv(caminho_csv, index_col=0)
    if modelo not in df.columns:
        print(f"--> '{modelo}' não está em {caminho_csv}; usando valores padrão.")
        return {}

    params = {}
    for nome, valor in df[modelo].dropna().items():
        if nome == "fc_sizes_str":
            # Mesma conversão da objective() do tune_models.py. O float() cobre o caso em que
            # o pandas lê a coluna como numérica e "512" chega aqui como 512.0
            params["fc_sizes"] = [int(float(x)) for x in str(valor).split('_')]
        elif nome in PARAMS_INTEIROS:
            params[nome] = int(float(valor))
        else:
            params[nome] = float(valor)
    return params


def treinar_modelo(nome_modelo, args):
    """Treina um modelo e devolve um dicionário com as métricas."""
    # Reinicia a seed para que todos os treinos partam do mesmo estado
    pl.seed_everything(RANDOM_SEED, workers=True)

    datamodule = CaptchaDataModule(
        diretorio_dados=args.diretorio_dados,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_seed=RANDOM_SEED,
    )

    params = carregar_hiperparametros(args.params_csv, nome_modelo)
    print(f"--> Hiperparâmetros: {params if params else 'padrão'}")
    model = MODELOS[nome_modelo](**params)
    parametros = sum(p.numel() for p in model.parameters() if p.requires_grad)

    early_stop = EarlyStopping(monitor='val_loss', patience=args.patience, mode='min', verbose=False)

    with tempfile.TemporaryDirectory() as pasta_ckpt:
        checkpoint = ModelCheckpoint(dirpath=pasta_ckpt, monitor='val_loss', mode='min', save_top_k=1)

        trainer_kwargs = {
            "max_epochs": args.max_epochs if not args.test else 1,
            "callbacks": [early_stop, checkpoint],
            "enable_progress_bar": False,
            "logger": False,
            "enable_model_summary": False,
            "deterministic": "warn",
        }

        if args.test:
            trainer_kwargs["limit_train_batches"] = 2
            trainer_kwargs["limit_val_batches"] = 2
            trainer_kwargs["limit_test_batches"] = 2

        trainer = pl.Trainer(**trainer_kwargs)

        inicio = time.perf_counter()
        trainer.fit(model, datamodule=datamodule)
        tempo_treino = time.perf_counter() - inicio

        epocas_treinadas = trainer.current_epoch
        melhor_val_loss = checkpoint.best_model_score.item()
        melhor_ckpt = checkpoint.best_model_path

        # validate() no conjunto de treino mede a acurácia sem dropout, comparável à validação
        train_metrics = trainer.validate(model, dataloaders=datamodule.train_dataloader(),
                                         ckpt_path=melhor_ckpt, verbose=False)[0]
        val_metrics = trainer.validate(model, datamodule=datamodule,
                                       ckpt_path=melhor_ckpt, verbose=False)[0]
        test_metrics = trainer.test(model, datamodule=datamodule,
                                    ckpt_path=melhor_ckpt, verbose=False)[0]

        # Guarda os pesos da melhor época para a etapa de XAI (fora do modo de teste)
        if not args.test:
            os.makedirs(args.pasta_ckpt, exist_ok=True)
            shutil.copy(melhor_ckpt, os.path.join(args.pasta_ckpt, f"{nome_modelo}.ckpt"))

    train_acc = train_metrics['val_acc']
    val_acc = val_metrics['val_acc']

    return {
        "modelo": nome_modelo,
        "parametros": parametros,
        "tempo_treino_s": round(tempo_treino, 1),
        "epocas_treinadas": epocas_treinadas,
        "melhor_val_loss": round(melhor_val_loss, 4),
        "train_acc": round(train_acc, 4),
        "val_acc": round(val_acc, 4),
        "test_acc": round(test_metrics['test_acc'], 4),
        "gap_treino_val": round(train_acc - val_acc, 4),
    }


if __name__ == "__main__":
    # =====================================================================
    # CONTROLE PRINCIPAL VIA TERMINAL
    # =====================================================================
    parser = argparse.ArgumentParser(description="Treino final dos modelos de CAPTCHA")
    parser.add_argument('--test', action='store_true', help="Ativa o modo de teste rápido (Sanity Check)")
    parser.add_argument('--modelos', nargs='+', default=list(MODELOS), choices=list(MODELOS))
    parser.add_argument('--diretorio_dados', default="dataset_captcha")
    parser.add_argument('--params_csv', default=None, help="CSV de hiperparâmetros gerado pelo tune_models.py")
    parser.add_argument('--pasta_ckpt', default="checkpoints_finais")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--saida', default="train_final_results.csv")
    args = parser.parse_args()

    print(f"Modo de Teste (Sanity Check): {'ATIVADO' if args.test else 'DESATIVADO'}")

    resultados = []
    for nome_modelo in args.modelos:
        print(f"\n{'='*50}")
        print(f" Treinando {nome_modelo}")
        print(f"{'='*50}")

        resultado = treinar_modelo(nome_modelo, args)
        resultados.append(resultado)

        print(f"--> Parâmetros: {resultado['parametros']:,} | Tempo: {resultado['tempo_treino_s']}s "
              f"| Épocas: {resultado['epocas_treinadas']}")
        print(f"--> Acurácia  treino: {resultado['train_acc']:.4f} | val: {resultado['val_acc']:.4f} "
              f"| teste: {resultado['test_acc']:.4f}")

        # Salva a cada treino para não perder resultados em caso de interrupção
        pd.DataFrame(resultados).to_csv(args.saida, index=False)

    print("\n\n" + "#"*50)
    print(" RESUMO FINAL DO TREINO ")
    print("#"*50)
    print(pd.DataFrame(resultados).to_string(index=False))
    print(f"\nResultados salvos em: {os.path.abspath(args.saida)}")
