import argparse
import os
import warnings
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch

from Dataset.Datamodule import CaptchaDataModule, idx2char
from Models.ConvNext import ConvNeXtCaptcha
from Models.EfficientNet import EfficientNetCaptcha
from Models.Vgg import VGGCaptcha
from Models.ResNet import ResNetCaptcha
from Models.Inception import InceptionCaptcha
from XAI import explainers


warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

"""
======================================================================================
GERAÇÃO DE MAPAS DE EXPLICABILIDADE (Fase 3) - Grad-CAM, Integrated Gradients e LIME
======================================================================================

Carrega um checkpoint treinado (gerado pelo train_final.py) e, para imagens do
CONJUNTO DE TESTE, salva uma figura por imagem: uma linha por caractere (posição) e
uma coluna por método. Não re-treina nada.

COMO EXECUTAR:
   python -m Scripts.gerar_mapas --modelo ResNetCaptcha \
       --ckpt checkpoints_finais/ResNetCaptcha.ckpt \
       --diretorio_dados dataset_captcha_2/dataset_captcha

MODO DE TESTE (Sanity Check):
   python -m Scripts.gerar_mapas --test --modelo ResNetCaptcha \
       --ckpt checkpoints_finais/ResNetCaptcha.ckpt \
       --diretorio_dados dataset_captcha_2/dataset_captcha

Opções úteis:
   --test                    Modo rápido (Sanity Check): 1 imagem e 10 amostras de LIME
   --n_imagens 10            Quantidade de imagens do teste a explicar
   --metodos gradcam ig lime Métodos a aplicar (padrão: os três)
   --saida mapas             Pasta de saída (uma subpasta por modelo)
   --amostras_lime 1000     Perturbações do LIME (é o método mais lento)
======================================================================================
"""

RANDOM_SEED = 42

MODELOS = {
    "VGGCaptcha": VGGCaptcha,
    "ConvNextCaptcha": ConvNeXtCaptcha,
    "EfficientNetCaptcha": EfficientNetCaptcha,
    "ResNetCaptcha": ResNetCaptcha,
    "InceptionCaptcha": InceptionCaptcha,
}


def gerar_figura(model, imagem, rotulo, metodos, args, caminho):
    """Uma linha por caractere: imagem original + um mapa por método."""
    num_chars = model.num_chars
    with torch.no_grad():
        predicao = model(imagem.unsqueeze(0))[0].argmax(dim=1)

    original = explainers.desnormalizar(imagem)
    fig, eixos = plt.subplots(num_chars, 1 + len(metodos), figsize=(3.2 * (1 + len(metodos)), 1.4 * num_chars),
                              squeeze=False)

    for pos in range(num_chars):
        classe = predicao[pos].item()
        certo = classe == rotulo[pos].item()
        eixos[pos][0].imshow(original)
        eixos[pos][0].set_ylabel(f"pos {pos}\nprev '{idx2char[classe]}'\n{'OK' if certo else 'ERRO'}", fontsize=7)

        for col, metodo in enumerate(metodos, start=1):
            if metodo == "gradcam":
                mapa = explainers.grad_cam(model, imagem, pos)
            elif metodo == "ig":
                mapa = explainers.integrated_gradients(model, imagem, pos)
            else:
                mapa = explainers.lime(model, imagem, pos, num_amostras=args.amostras_lime)
            eixos[pos][col].imshow(original)
            eixos[pos][col].imshow(mapa, cmap="jet", alpha=0.5)
            if pos == 0:
                eixos[pos][col].set_title({"gradcam": "Grad-CAM", "ig": "Integrated Gradients", "lime": "LIME"}[metodo],
                                          fontsize=8)

    texto_real = "".join(idx2char[i.item()] for i in rotulo)
    texto_prev = "".join(idx2char[i.item()] for i in predicao)
    fig.suptitle(f"real: {texto_real} | previsto: {texto_prev}", fontsize=9)
    for eixo in eixos.ravel():
        eixo.set_xticks([])
        eixo.set_yticks([])
    fig.tight_layout()
    fig.savefig(caminho, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera mapas Grad-CAM, IG e LIME sobre o conjunto de teste")
    parser.add_argument('--test', action='store_true', help="Ativa o modo de teste rápido (Sanity Check)")
    parser.add_argument('--modelo', required=True, choices=list(MODELOS))
    parser.add_argument('--ckpt', required=True, help="Checkpoint treinado (.ckpt)")
    parser.add_argument('--diretorio_dados', default="dataset_captcha")
    parser.add_argument('--n_imagens', type=int, default=10)
    parser.add_argument('--metodos', nargs='+', default=["gradcam", "ig", "lime"], choices=["gradcam", "ig", "lime"])
    parser.add_argument('--amostras_lime', type=int, default=1000)
    parser.add_argument('--saida', default="mapas")
    args = parser.parse_args()

    pl.seed_everything(RANDOM_SEED, workers=True)

    if args.test:
        print("Modo de Teste (Sanity Check): ATIVADO")
        args.n_imagens = min(args.n_imagens, 1)
        args.amostras_lime = 10
    else:
        print("Modo de Teste (Sanity Check): DESATIVADO")

    model = MODELOS[args.modelo].load_from_checkpoint(args.ckpt, map_location="cpu")
    model.eval()

    # Mesmo split (seed 42) do treino: as imagens vêm do conjunto de teste
    datamodule = CaptchaDataModule(diretorio_dados=args.diretorio_dados, random_seed=RANDOM_SEED, num_workers=0)
    datamodule.setup()

    pasta = os.path.join(args.saida, args.modelo)
    os.makedirs(pasta, exist_ok=True)

    for i in range(min(args.n_imagens, len(datamodule.dataset_teste))):
        imagem, rotulo = datamodule.dataset_teste[i]
        caminho = os.path.join(pasta, f"teste_{i:03d}.png")
        gerar_figura(model, imagem, rotulo, args.metodos, args, caminho)
        print(f"--> {caminho}")
