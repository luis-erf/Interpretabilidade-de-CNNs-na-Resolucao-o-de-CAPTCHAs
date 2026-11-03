import numpy as np
import torch
import torch.nn.functional as F
from lime import lime_image
from skimage.segmentation import slic

"""
======================================================================================
EXPLICADORES (XAI) - Grad-CAM, Integrated Gradients e LIME
======================================================================================

Os três métodos explicam UM caractere por vez: o modelo devolve logits de formato
(batch, num_chars, num_classes) e cada explicação é calculada para a posição `pos`
(0 a num_chars-1) e para a classe `alvo` (por padrão, a classe predita nessa posição).

Todos recebem uma imagem normalizada de formato (3, H, W) e devolvem um mapa (H, W)
normalizado em [0, 1], pronto para sobrepor à imagem original.

Requisitos dos modelos: atributo `features` (extrator convolucional, alvo do Grad-CAM)
e `forward(x) -> (batch, num_chars, num_classes)`. Vale para todos os modelos do projeto.
======================================================================================
"""

MEDIA_IMAGENET = [0.485, 0.456, 0.406]
DESVIO_IMAGENET = [0.229, 0.224, 0.225]


def desnormalizar(imagem, media=MEDIA_IMAGENET, desvio=DESVIO_IMAGENET):
    """(3, H, W) normalizado -> (H, W, 3) numpy em [0, 1], para exibição."""
    media = torch.tensor(media).view(3, 1, 1)
    desvio = torch.tensor(desvio).view(3, 1, 1)
    return (imagem.cpu() * desvio + media).clamp(0, 1).permute(1, 2, 0).numpy()


def _normalizar_mapa(mapa):
    mapa = mapa - mapa.min()
    maximo = mapa.max()
    return mapa / maximo if maximo > 0 else mapa


def _classe_alvo(model, imagem, pos, alvo):
    if alvo is not None:
        return alvo
    with torch.no_grad():
        return model(imagem.unsqueeze(0))[0, pos].argmax().item()


def grad_cam(model, imagem, pos, alvo=None):
    """
    Grad-CAM (Selvaraju et al., 2017) na saída de `model.features`.
    Pondera cada canal pelo gradiente médio do logit alvo e aplica ReLU.
    """
    model.eval()
    alvo = _classe_alvo(model, imagem, pos, alvo)
    ativacoes = {}

    def hook(_, __, saida):
        saida.retain_grad()
        ativacoes["mapa"] = saida

    handle = model.features.register_forward_hook(hook)
    try:
        model.zero_grad()
        logits = model(imagem.unsqueeze(0))
        logits[0, pos, alvo].backward()
    finally:
        handle.remove()

    mapa = ativacoes["mapa"]
    pesos = mapa.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((pesos * mapa).sum(dim=1, keepdim=True)).detach()
    cam = F.interpolate(cam, size=imagem.shape[1:], mode="bilinear", align_corners=False)
    return _normalizar_mapa(cam[0, 0].numpy())


def integrated_gradients(model, imagem, pos, alvo=None, passos=50, baseline=None):
    """
    Integrated Gradients (Sundararajan et al., 2017) com aproximação por soma de Riemann.
    A baseline padrão é a imagem totalmente preta (zero em pixels, não normalizada).
    Devolve a magnitude da atribuição somada nos canais de cor.
    """
    model.eval()
    alvo = _classe_alvo(model, imagem, pos, alvo)

    if baseline is None:
        media = torch.tensor(MEDIA_IMAGENET).view(3, 1, 1)
        desvio = torch.tensor(DESVIO_IMAGENET).view(3, 1, 1)
        baseline = (0.0 - media) / desvio  # preto no espaço normalizado

    alphas = torch.linspace(0, 1, passos + 1)[1:].view(-1, 1, 1, 1)
    interpolados = (baseline + alphas * (imagem - baseline)).requires_grad_(True)

    logits = model(interpolados)
    logits[:, pos, alvo].sum().backward()

    gradiente_medio = interpolados.grad.mean(dim=0)
    atribuicao = (imagem - baseline) * gradiente_medio
    return _normalizar_mapa(atribuicao.abs().sum(dim=0).detach().numpy())


def lime(model, imagem, pos, alvo=None, num_amostras=1000, num_segmentos=60, seed=42):
    """
    LIME (Ribeiro et al., 2016) com superpixels SLIC. Trata o modelo como caixa-preta.
    Devolve o peso do modelo linear de cada superpixel espalhado sobre seus pixels.
    """
    model.eval()
    alvo = _classe_alvo(model, imagem, pos, alvo)
    media = torch.tensor(MEDIA_IMAGENET).view(1, 3, 1, 1)
    desvio = torch.tensor(DESVIO_IMAGENET).view(1, 3, 1, 1)

    def prever(imagens_np):
        # LIME entrega (N, H, W, 3) em [0, 1]; o modelo espera tensores normalizados
        x = torch.from_numpy(imagens_np).float().permute(0, 3, 1, 2)
        x = (x - media) / desvio
        with torch.no_grad():
            return F.softmax(model(x)[:, pos, :], dim=1).numpy()

    explicador = lime_image.LimeImageExplainer(random_state=seed)
    explicacao = explicador.explain_instance(
        desnormalizar(imagem).astype(np.double),
        prever,
        labels=(alvo,),
        top_labels=None,
        num_samples=num_amostras,
        segmentation_fn=lambda img: slic(img, n_segments=num_segmentos, compactness=10, start_label=0),
        random_seed=seed,
        batch_size=64,
    )

    pesos = dict(explicacao.local_exp[alvo])
    mapa = np.zeros(explicacao.segments.shape, dtype=np.float32)
    for segmento, peso in pesos.items():
        mapa[explicacao.segments == segmento] = peso
    return _normalizar_mapa(np.clip(mapa, 0, None))
