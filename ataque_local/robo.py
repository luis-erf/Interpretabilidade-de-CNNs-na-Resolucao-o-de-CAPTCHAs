"""
======================================================================================
ROBÔ DE ATAQUE (Playwright + modelo treinado) contra o servidor LOCAL de CAPTCHA
======================================================================================

Abre a página do `servidor.py`, baixa a imagem do CAPTCHA, prevê o texto com um
checkpoint do `train_final.py`, digita a resposta, envia e lê se passou. Mede a
taxa de sucesso por CAPTCHA INTEIRO (os 5 caracteres certos), que é o que importa
num ataque real, ao contrário da acurácia por caractere do treino.

Uso permitido: SOMENTE contra o servidor local deste projeto. O robô recusa
qualquer URL que não seja 127.0.0.1/localhost.

COMO EXECUTAR (na raiz do repositório, com o servidor rodando em outro terminal):
   python -m ataque_local.servidor --seed 42
   python -m ataque_local.robo --modelo EfficientNetCaptcha \
       --ckpt resultados_bernardo/checkpoints_finais/EfficientNetCaptcha.ckpt --tentativas 200

Demonstração visual (navegador aberto e passos mais lentos):
   python -m ataque_local.robo --modelo AlexNetCaptcha \
       --ckpt resultados_bernardo/checkpoints_finais/AlexNetCaptcha.ckpt --tentativas 10 --visivel --lento 600

Saída em ataque_local/resultados/<modelo>/:
   tentativas.csv   uma linha por tentativa (previsto, correto, acertou, confiança, tempos)
   erros/           imagens que o modelo errou, para gerar mapas de XAI depois
======================================================================================
"""
import argparse
import csv
import io
import os
import time
import warnings
from urllib.parse import urljoin, urlparse

import torch
from PIL import Image
from playwright.sync_api import sync_playwright
from torchvision import transforms

from Dataset.Datamodule import idx2char
from Models.AlexNet import AlexNetCaptcha
from Models.ConvNext import ConvNeXtCaptcha
from Models.EfficientNet import EfficientNetCaptcha
from Models.Inception import InceptionCaptcha
from Models.ResNet import ResNetCaptcha
from Models.Vgg import VGGCaptcha

warnings.filterwarnings("ignore")

MODELOS = {
    "AlexNetCaptcha": AlexNetCaptcha,
    "VGGCaptcha": VGGCaptcha,
    "ConvNextCaptcha": ConvNeXtCaptcha,
    "EfficientNetCaptcha": EfficientNetCaptcha,
    "ResNetCaptcha": ResNetCaptcha,
    "InceptionCaptcha": InceptionCaptcha,
}

# Mesmo pré-processamento do CaptchaDataModule (sem resize, normalização ImageNet)
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def prever(model, png_bytes):
    imagem = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    with torch.no_grad():
        probs = model(TRANSFORM(imagem).unsqueeze(0))[0].softmax(dim=1)  # (num_chars, num_classes)
    confianca, indices = probs.max(dim=1)
    texto = "".join(idx2char[i.item()] for i in indices)
    return texto, confianca.min().item()


def garantir_local(url):
    if urlparse(url).hostname not in {"127.0.0.1", "localhost"}:
        raise SystemExit("O robô só ataca o servidor local do projeto (127.0.0.1/localhost).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ataque controlado ao servidor local de CAPTCHA")
    parser.add_argument("--modelo", required=True, choices=list(MODELOS))
    parser.add_argument("--ckpt", required=True, help="Checkpoint treinado (.ckpt)")
    parser.add_argument("--url", default="http://127.0.0.1:5050/")
    parser.add_argument("--tentativas", type=int, default=100)
    parser.add_argument("--visivel", action="store_true", help="Abre o navegador na tela (demonstração)")
    parser.add_argument("--lento", type=int, default=0, help="Atraso em ms entre ações do navegador")
    parser.add_argument("--saida", default="ataque_local/resultados")
    args = parser.parse_args()
    garantir_local(args.url)

    torch.set_num_threads(max(1, os.cpu_count() // 2))
    model = MODELOS[args.modelo].load_from_checkpoint(args.ckpt, map_location="cpu")
    model.eval()

    pasta = os.path.join(args.saida, args.modelo)
    pasta_erros = os.path.join(pasta, "erros")
    os.makedirs(pasta_erros, exist_ok=True)
    caminho_csv = os.path.join(pasta, "tentativas.csv")

    acertos = 0
    with sync_playwright() as p, open(caminho_csv, "w", newline="") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(["tentativa", "previsto", "correto", "acertou", "confianca_min",
                           "tempo_inferencia_ms", "tempo_total_ms"])

        navegador = p.chromium.launch(headless=not args.visivel, slow_mo=args.lento)
        pagina = navegador.new_page(viewport={"width": 520, "height": 640})
        pagina.goto(args.url)

        for n in range(1, args.tentativas + 1):
            inicio = time.perf_counter()

            # Baixa exatamente a imagem exibida na página (mesmo id, mesmos bytes)
            src = pagina.locator("#captcha").get_attribute("src")
            png = pagina.request.get(urljoin(pagina.url, src)).body()

            t0 = time.perf_counter()
            previsto, confianca = prever(model, png)
            tempo_inferencia = (time.perf_counter() - t0) * 1000

            pagina.fill("#resposta", previsto)
            pagina.click("#enviar")
            resultado = pagina.locator("#resultado")
            resultado.wait_for()
            acertou = resultado.get_attribute("data-ok") == "1"
            correto = resultado.get_attribute("data-correto")
            tempo_total = (time.perf_counter() - inicio) * 1000

            acertos += acertou
            if not acertou:
                with open(os.path.join(pasta_erros, f"{correto}_{n:04d}.png"), "wb") as f:
                    f.write(png)  # nome no formato do dataset: <texto>_<i>.png
            escritor.writerow([n, previsto, correto, int(acertou), f"{confianca:.4f}",
                               f"{tempo_inferencia:.1f}", f"{tempo_total:.1f}"])
            print(f"[{n:4d}/{args.tentativas}] previsto {previsto} | correto {correto} | "
                  f"{'PASSOU' if acertou else 'FALHOU'} | taxa {100 * acertos / n:.1f}%")

        navegador.close()

    print(f"\n{args.modelo}: {acertos}/{args.tentativas} CAPTCHAs passados "
          f"({100 * acertos / args.tentativas:.1f}%). Detalhes em {caminho_csv}")
