# Ataque controlado a CAPTCHA (Flask + Playwright)

Demonstração ponta a ponta: um robô com Playwright abre uma página de login protegida por CAPTCHA,
lê a imagem, resolve com um modelo treinado no projeto, digita a resposta e envia.
Mede a **taxa de sucesso por CAPTCHA inteiro**, a métrica que importa num ataque real
(a acurácia do treino é por caractere).

> **Uso restrito:** o alvo é só o servidor local deste projeto (`servidor.py`, escuta em 127.0.0.1).
> O robô recusa qualquer outra URL. Não use contra sites de terceiros.

## Como rodar

Na raiz do repositório, com o `.venv` ativo (`pip install flask playwright` e `playwright install chromium`):

```bash
# Terminal 1: servidor (http://127.0.0.1:5050)
python -m ataque_local.servidor --seed 42

# Terminal 2: avaliação (navegador invisível)
python -m ataque_local.robo --modelo EfficientNetCaptcha \
    --ckpt resultados_bernardo/checkpoints_finais/EfficientNetCaptcha.ckpt --tentativas 500

# Demonstração ao vivo (abre o Chromium e desacelera os passos)
python -m ataque_local.robo --modelo AlexNetCaptcha \
    --ckpt resultados_bernardo/checkpoints_finais/AlexNetCaptcha.ckpt --tentativas 10 --visivel --lento 600
```

Os CAPTCHAs usam a mesma biblioteca e formato do dataset (`ImageCaptcha` 160×60, 5 caracteres A–Z/0–9).
Com `--seed`, a sequência de CAPTCHAs é a mesma a cada reinício, o que permite comparar modelos no mesmo conjunto.

## Resultados (500 tentativas por modelo, servidor com `--seed 42`, inferência em CPU num Mac)

| Modelo | CAPTCHAs passados | Acc. por caractere | Inferência (mediana) | Tentativa completa (mediana) |
|---|---|---|---|---|
| AlexNetCaptcha | **78,2%** (391/500) | 94,88% | 7,5 ms | 67 ms |
| EfficientNetCaptcha (B0) | **95,6%** (478/500) | 99,08% | 89,5 ms | 150 ms |

- A acurácia por caractere bate com a do conjunto de teste (94,76% e 98,99%), o que confirma que o robô
  usa o mesmo pré-processamento do treino.
- A diferença entre os modelos cresce de ~4 pontos (por caractere) para ~17 pontos (por CAPTCHA),
  porque basta errar um dos 5 caracteres para falhar.
- A confiança mínima entre os 5 caracteres separa bem acertos de erros (mediana AlexNet: 0,994 nos acertos
  e 0,714 nos erros; EfficientNet: 1,000 e 0,868). Um atacante poderia pedir um CAPTCHA novo quando a
  confiança é baixa, e a taxa efetiva subiria ainda mais.

## Saídas

`resultados/<modelo>/tentativas.csv` tem uma linha por tentativa: previsto, correto, acertou, confiança mínima
e tempos. `resultados/<modelo>/erros/` guarda as imagens que o modelo errou, nomeadas no formato do dataset
(`<texto>_<n>.png`), para gerar mapas de XAI dos casos de falha.
