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

Demonstração lado a lado (AlexNet à esquerda, EfficientNet à direita, mesma sequência de CAPTCHAs):

```bash
bash ataque_local/demo.sh            # 15 tentativas, 700 ms entre ações
bash ataque_local/demo.sh 30 1200    # tentativas, atraso em ms
```

Os CAPTCHAs usam a mesma biblioteca e formato do dataset (`ImageCaptcha` 160×60, 5 caracteres A–Z/0–9).
Com `--seed`, a sequência de CAPTCHAs (texto **e** imagem: fonte, distorção, cor e ruído) é a mesma a cada reinício,
então modelos diferentes enfrentam exatamente as mesmas imagens. A biblioteca `captcha` sorteia o visual com o módulo
`secrets`, que ignora seeds; o servidor o substitui por um gerador com seed quando `--seed` é passado.

## Resultados (500 tentativas por modelo, servidor com `--seed 42`, inferência em CPU num Mac)

| Modelo | CAPTCHAs passados | Acc. por caractere | Inferência (mediana) | Tentativa completa (mediana) |
|---|---|---|---|---|
| AlexNetCaptcha | **82,2%** (411/500) | 96,00% | 7,4 ms | 67 ms |
| EfficientNetCaptcha (B0) | **96,4%** (482/500) | 99,28% | 89,9 ms | 150 ms |

- A acurácia por caractere fica próxima da do conjunto de teste (94,76% e 98,99%), o que confirma que o robô
  usa o mesmo pré-processamento do treino.
- A diferença entre os modelos cresce de ~3 pontos (por caractere) para ~14 pontos (por CAPTCHA),
  porque basta errar um dos 5 caracteres para falhar.
- Nos mesmos 500 CAPTCHAs: 11 derrubam os dois modelos, 78 só a AlexNet e 7 só a EfficientNet.
- A confiança mínima entre os 5 caracteres separa bem acertos de erros (mediana AlexNet: 0,991 nos acertos
  e 0,737 nos erros; EfficientNet: 1,000 e 0,799). Um atacante poderia pedir um CAPTCHA novo quando a
  confiança é baixa, e a taxa efetiva subiria ainda mais.

## Há vazamento (leakage) do treino?

Não. O servidor gera cada CAPTCHA na hora, com texto e ruído novos; ele não lê o `dataset_captcha`.
Conferido nos 500 CAPTCHAs da tabela acima:

- 0 dos textos aparecem no conjunto de treino (nem no dataset inteiro, que tem 29.992 textos de 36⁵ ≈ 60 milhões possíveis);
- 0 das 107 imagens de erro são idênticas a algum arquivo do dataset (comparação por hash).

Ressalva: os CAPTCHAs vêm do **mesmo gerador** (`ImageCaptcha`, mesma fonte, tamanho e ruído). O ataque mede
desempenho em dados novos da mesma distribuição, não generalização para outros estilos de CAPTCHA.

## Saídas

`resultados/<modelo>/tentativas.csv` tem uma linha por tentativa: previsto, correto, acertou, confiança mínima
e tempos. `resultados/<modelo>/erros/` guarda as imagens que o modelo errou, nomeadas no formato do dataset
(`<texto>_<n>.png`), para gerar mapas de XAI dos casos de falha.
