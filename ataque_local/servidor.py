"""
======================================================================================
SERVIDOR LOCAL DE CAPTCHA (alvo controlado para o ataque com Playwright)
======================================================================================

Página de login fictícia protegida por CAPTCHA de texto, gerado pela MESMA biblioteca
e no MESMO formato do dataset (ImageCaptcha 160x60, 5 caracteres A-Z/0-9). Serve só
como alvo do `robo.py` e para demonstração: escuta apenas em 127.0.0.1.

COMO EXECUTAR (na raiz do repositório):
   python -m ataque_local.servidor              -> http://127.0.0.1:5050
   python -m ataque_local.servidor --seed 42    -> sequência de CAPTCHAs reproduzível
   python -m ataque_local.servidor --porta 6000

Rotas:
   GET  /                   formulário com um CAPTCHA novo + placar de tentativas
   GET  /captcha/<id>.png   imagem do CAPTCHA
   POST /verificar          confere a resposta (cada CAPTCHA vale uma única tentativa)
   POST /zerar              zera o placar
======================================================================================
"""
import argparse
import io
import random
import string
import threading
import uuid
from collections import OrderedDict

from captcha.image import ImageCaptcha
from flask import Flask, abort, redirect, render_template_string, request, send_file, url_for

CARACTERES = string.ascii_uppercase + string.digits
TAMANHO_TEXTO = 5
MAX_PENDENTES = 1000  # CAPTCHAs gerados e ainda não respondidos guardados em memória

app = Flask(__name__)
gerador_imagem = ImageCaptcha(width=160, height=60)  # mesmo tamanho do Gerar_dataset.py
rng = random.Random()
trava = threading.Lock()
pendentes = OrderedDict()  # id -> (texto correto, PNG gerado uma única vez)
placar = {"tentativas": 0, "acertos": 0}

PAGINA = """<!doctype html>
<html lang="pt-br">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Login protegido por CAPTCHA</title>
<style>
  :root { --fundo:#f4f5f7; --cartao:#fff; --texto:#1d2330; --suave:#667085; --ok:#12805c; --erro:#c0362c; --borda:#d0d5dd; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, -apple-system, sans-serif; background:var(--fundo); color:var(--texto);
         min-height:100vh; display:flex; align-items:center; justify-content:center; padding:16px; }
  .cartao { background:var(--cartao); border:1px solid var(--borda); border-radius:12px; padding:28px; width:100%; max-width:380px; }
  h1 { font-size:20px; margin:0 0 4px; }
  p.sub { color:var(--suave); margin:0 0 20px; font-size:14px; }
  label { display:block; font-size:13px; font-weight:600; margin:14px 0 6px; }
  input[type=text] { width:100%; padding:10px 12px; font-size:16px; border:1px solid var(--borda); border-radius:8px;
                     letter-spacing:2px; text-transform:uppercase; }
  #captcha { display:block; width:160px; height:60px; outline:1px solid var(--borda); border-radius:6px; }
  button { margin-top:18px; width:100%; padding:11px; font-size:15px; font-weight:600; color:#fff; background:#2f5bea;
           border:0; border-radius:8px; cursor:pointer; }
  #resultado { margin:0 0 16px; padding:10px 12px; border-radius:8px; font-size:14px; font-weight:600; }
  #resultado.ok { background:#e7f6ef; color:var(--ok); }
  #resultado.erro { background:#fdecea; color:var(--erro); }
  .placar { margin-top:22px; padding-top:14px; border-top:1px solid var(--borda); font-size:13px; color:var(--suave);
            display:flex; justify-content:space-between; }
  .placar strong { color:var(--texto); font-variant-numeric: tabular-nums; }
</style>
</head>
<body>
<main class="cartao">
  <h1>Acesso restrito</h1>
  <p class="sub">Digite os caracteres da imagem para continuar.</p>
  {% if resultado %}
  <div id="resultado" class="{{ 'ok' if resultado.ok else 'erro' }}" data-ok="{{ 1 if resultado.ok else 0 }}"
       data-correto="{{ resultado.correto }}" data-digitado="{{ resultado.digitado }}">
    {% if resultado.ok %}Acesso liberado: CAPTCHA correto ({{ resultado.correto }}).
    {% else %}CAPTCHA incorreto: digitado {{ resultado.digitado or '—' }}, correto {{ resultado.correto }}.{% endif %}
  </div>
  {% endif %}
  <form method="post" action="{{ url_for('verificar') }}">
    <input type="hidden" name="id" value="{{ id }}">
    <label for="usuario">Usuário</label>
    <input id="usuario" name="usuario" type="text" value="demo" autocomplete="off" style="text-transform:none;letter-spacing:0">
    <label>CAPTCHA</label>
    <img id="captcha" src="{{ url_for('imagem', id_captcha=id) }}" width="160" height="60" alt="CAPTCHA">
    <label for="resposta">Resposta</label>
    <input id="resposta" name="resposta" type="text" maxlength="{{ tamanho }}" autocomplete="off" required>
    <button id="enviar" type="submit">Entrar</button>
  </form>
  <div class="placar">
    <span>Tentativas <strong id="tentativas">{{ placar.tentativas }}</strong></span>
    <span>Acertos <strong id="acertos">{{ placar.acertos }}</strong></span>
    <span>Taxa <strong id="taxa">{{ taxa }}</strong></span>
  </div>
</main>
</body>
</html>"""


def novo_captcha():
    with trava:
        texto = "".join(rng.choices(CARACTERES, k=TAMANHO_TEXTO))
        id_captcha = uuid.UUID(int=rng.getrandbits(128)).hex
        # ImageCaptcha usa o random global: gerar dentro da trava mantém a sequência reproduzível
        pendentes[id_captcha] = (texto, gerador_imagem.generate(texto).getvalue())
        while len(pendentes) > MAX_PENDENTES:
            pendentes.popitem(last=False)
    return id_captcha


def renderizar(resultado=None):
    t = placar["tentativas"]
    taxa = f"{100 * placar['acertos'] / t:.1f}%" if t else "—"
    return render_template_string(PAGINA, id=novo_captcha(), resultado=resultado, placar=placar,
                                  taxa=taxa, tamanho=TAMANHO_TEXTO)


@app.get("/")
def inicio():
    return renderizar()


@app.get("/captcha/<id_captcha>.png")
def imagem(id_captcha):
    item = pendentes.get(id_captcha)
    if item is None:
        abort(404)
    # Sempre os mesmos bytes para o mesmo id: quem baixa a imagem vê exatamente o que a página mostra
    return send_file(io.BytesIO(item[1]), mimetype="image/png", max_age=0)


@app.post("/verificar")
def verificar():
    with trava:
        item = pendentes.pop(request.form.get("id", ""), None)
    if item is None:  # CAPTCHA expirado ou reutilizado
        return redirect(url_for("inicio"))
    correto = item[0]
    digitado = request.form.get("resposta", "").strip().upper()
    ok = digitado == correto
    with trava:
        placar["tentativas"] += 1
        placar["acertos"] += int(ok)
    return renderizar({"ok": ok, "digitado": digitado, "correto": correto})


@app.post("/zerar")
def zerar():
    with trava:
        placar.update(tentativas=0, acertos=0)
    return ("", 204)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Servidor local de CAPTCHA para o ataque controlado")
    parser.add_argument("--porta", type=int, default=5050)  # a 5000 é do AirPlay no macOS
    parser.add_argument("--seed", type=int, default=None, help="Fixa a sequência de CAPTCHAs gerados")
    args = parser.parse_args()
    if args.seed is not None:
        rng.seed(args.seed)
        random.seed(args.seed)  # ruído/distorção do ImageCaptcha
    print(f"Servidor de CAPTCHA em http://127.0.0.1:{args.porta}")
    app.run(host="127.0.0.1", port=args.porta, threaded=True)
