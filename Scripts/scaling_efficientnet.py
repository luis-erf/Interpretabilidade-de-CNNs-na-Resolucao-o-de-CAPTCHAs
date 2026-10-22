import argparse
import warnings
import logging
import os
import tempfile
import time

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd

from Dataset.Datamodule import CaptchaDataModule
from Models.EfficientNet import EfficientNetCaptcha, scale_values


# Silencia avisos e logs informativos (mesmo padrão do tune_models.py)
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

"""
======================================================================================
EXPERIMENTO DE ESCALONAMENTO - EFFICIENTNET B0 A B3
======================================================================================

Treina as versões B0, B1, B2 e B3 da EfficientNet para medir o impacto do
escalonamento no desempenho sobre os CAPTCHAs. Cada versão é treinada em dois modos:

  - "profundidade_largura": cresce apenas profundidade (alpha^phi) e largura (beta^phi).
                            As imagens mantêm o tamanho original (60x160).
  - "composto":             escalonamento composto completo do artigo. Além de profundidade
                            e largura, a resolução cresce na proporção resolução/224
                            (ex: B3 -> 300/224 = 1.34 -> imagens 80x214).

Comparar os dois modos mostra quanto a resolução contribui sozinha.

Para uma comparação justa, todos os treinos usam a mesma seed, a mesma taxa de
aprendizado e o mesmo critério de parada. A avaliação no teste usa os pesos da
melhor época de validação.

--------------------------------------------------------------------------------------
COMO EXECUTAR:
--------------------------------------------------------------------------------------
1. Experimento completo (recomendado em GPU, ex: cluster APUANA):
   Comando: python -m Scripts.scaling_efficientnet

2. Modo de Teste (Sanity Check): 2 mini-lotes e 1 época por treino.
   Comando: python -m Scripts.scaling_efficientnet --test

3. Opções úteis:
   --versoes b0 b1        Treina apenas as versões escolhidas
   --modos composto       Treina apenas um dos modos
   --max_epochs 30        Limite de épocas por treino
   --learning_rate 1e-3   Taxa de aprendizado (fixa para todas as versões)
   --saida arquivo.csv    Nome do CSV de resultados

O CSV é atualizado ao fim de cada treino, então resultados parciais não se perdem
se o job for interrompido.

--------------------------------------------------------------------------------------
COLUNAS DO CSV DE RESULTADOS:
--------------------------------------------------------------------------------------
  versao, modo, resolucao      -> configuração do treino
  parametros                   -> número de parâmetros treináveis
  tempo_treino_s               -> tempo total do trainer.fit (segundos)
  epocas_treinadas             -> épocas executadas até o Early Stopping
  melhor_val_loss              -> menor perda de validação alcançada
  train_acc, val_acc, test_acc -> acurácia por caractere com os pesos da melhor época
  gap_treino_val               -> train_acc - val_acc (indicador de overfitting)
======================================================================================
"""

RANDOM_SEED = 42
ALTURA_ORIGINAL, LARGURA_ORIGINAL = 60, 160
RESOLUCAO_BASE = 224  # Resolução da B0 no artigo original


def calcular_resolucao(versao, modo):
    """Retorna (altura, largura) das imagens para a versão e o modo escolhidos."""
    if modo == "profundidade_largura":
        return ALTURA_ORIGINAL, LARGURA_ORIGINAL

    _, resolucao, _ = scale_values[versao]
    fator = resolucao / RESOLUCAO_BASE
    return round(ALTURA_ORIGINAL * fator), round(LARGURA_ORIGINAL * fator)


def treinar_versao(versao, modo, args):
    """Treina uma versão em um modo e devolve um dicionário com as métricas."""
    # Reinicia a seed para que todos os treinos partam do mesmo estado
    pl.seed_everything(RANDOM_SEED, workers=True)

    altura, largura = calcular_resolucao(versao, modo)

    datamodule = CaptchaDataModule(
        diretorio_dados=args.diretorio_dados,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_seed=RANDOM_SEED,
        resize_dims=(altura, largura),
    )

    model = EfficientNetCaptcha(version=versao, learning_rate=args.learning_rate)
    parametros = sum(p.numel() for p in model.parameters() if p.requires_grad)

    early_stop = EarlyStopping(monitor='val_loss', patience=args.patience, mode='min', verbose=False)

    # Os checkpoints ficam numa pasta temporária, apagada ao fim do treino,
    # apenas para recuperar os pesos da melhor época na avaliação
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

        # Avalia treino, validação e teste com os pesos da melhor época.
        # validate() é usado no conjunto de treino para medir a acurácia sem
        # dropout/stochastic depth, permitindo comparar com a validação.
        train_metrics = trainer.validate(model, dataloaders=datamodule.train_dataloader(),
                                         ckpt_path=melhor_ckpt, verbose=False)[0]
        val_metrics = trainer.validate(model, datamodule=datamodule,
                                       ckpt_path=melhor_ckpt, verbose=False)[0]
        test_metrics = trainer.test(model, datamodule=datamodule,
                                    ckpt_path=melhor_ckpt, verbose=False)[0]

    train_acc = train_metrics['val_acc']
    val_acc = val_metrics['val_acc']

    return {
        "versao": versao,
        "modo": modo,
        "resolucao": f"{altura}x{largura}",
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
    parser = argparse.ArgumentParser(description="Experimento de escalonamento da EfficientNet (B0 a B3)")
    parser.add_argument('--test', action='store_true', help="Ativa o modo de teste rápido (Sanity Check)")
    parser.add_argument('--versoes', nargs='+', default=["b0", "b1", "b2", "b3"], choices=list(scale_values))
    parser.add_argument('--modos', nargs='+', default=["profundidade_largura", "composto"],
                        choices=["profundidade_largura", "composto"])
    parser.add_argument('--diretorio_dados', default="dataset_captcha")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--saida', default="scaling_efficientnet_results.csv")
    args = parser.parse_args()

    print(f"Modo de Teste (Sanity Check): {'ATIVADO' if args.test else 'DESATIVADO'}")

    resultados = []
    for modo in args.modos:
        for versao in args.versoes:
            altura, largura = calcular_resolucao(versao, modo)
            print(f"\n{'='*50}")
            print(f" Treinando EfficientNet-{versao.upper()} | modo: {modo} | imagens {altura}x{largura}")
            print(f"{'='*50}")

            resultado = treinar_versao(versao, modo, args)
            resultados.append(resultado)

            print(f"--> Parâmetros: {resultado['parametros']:,} | Tempo: {resultado['tempo_treino_s']}s "
                  f"| Épocas: {resultado['epocas_treinadas']}")
            print(f"--> Acurácia  treino: {resultado['train_acc']:.4f} | val: {resultado['val_acc']:.4f} "
                  f"| teste: {resultado['test_acc']:.4f}")

            # Salva a cada treino para não perder resultados em caso de interrupção
            pd.DataFrame(resultados).to_csv(args.saida, index=False)

    print("\n\n" + "#"*50)
    print(" RESUMO FINAL DO ESCALONAMENTO ")
    print("#"*50)
    print(pd.DataFrame(resultados).to_string(index=False))
    print(f"\nResultados salvos em: {os.path.abspath(args.saida)}")
