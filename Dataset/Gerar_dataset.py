import os
import random
import string
from captcha.image import ImageCaptcha

def gerar_dataset_captcha(qtd_imagens, diretorio_saida, tamanho_texto=4):
    """
    Gera um dataset de imagens CAPTCHA.
    
    qtd_imagens: Quantidade total de imagens a serem geradas.
    diretorio_saida: Pasta onde as imagens serão salvas.
    tamanho_texto: Quantidade de caracteres no CAPTCHA.
    """
    # Cria o diretório de saída caso ele não exista
    if not os.path.exists(diretorio_saida):
        os.makedirs(diretorio_saida)
        print(f"Diretório '{diretorio_saida}' criado.")

    # Inicializa o gerador de imagem (você pode customizar width e height)
    gerador = ImageCaptcha(width=160, height=60)

    # Define o conjunto de caracteres (Letras maiúsculas e números)
    caracteres_permitidos = string.ascii_uppercase + string.digits

    print(f"Iniciando a geração de {qtd_imagens} imagens...")

    for i in range(qtd_imagens):
        # Gera o texto aleatório do CAPTCHA
        texto_captcha = ''.join(random.choices(caracteres_permitidos, k=tamanho_texto))
        
        # Cria o nome do arquivo. 
        # Adicionamos o índice '_i' para evitar que duas imagens com o mesmo texto se sobrescrevam.
        nome_arquivo = f"{texto_captcha}_{i}.png"
        caminho_arquivo = os.path.join(diretorio_saida, nome_arquivo)
        
        # Gera a imagem e salva no disco
        gerador.write(texto_captcha, caminho_arquivo)
        
        # Feedback visual no terminal a cada 1000 imagens geradas
        if (i + 1) % 1000 == 0:
            print(f"{i + 1} imagens geradas...")

    print(f"\nSucesso! {qtd_imagens} imagens foram salvas na pasta '{diretorio_saida}'.")

# Bloco principal de execução
if __name__ == "__main__":
    # Configurações do dataset
    NUMERO_DE_IMAGENS = 30000   # Ajuste para o tamanho desejado (ex: 50000 para um treino robusto)
    PASTA_DESTINO = "dataset_captcha"
    TAMANHO_CAPTCHA = 5         # Número de caracteres na imagem
    
    gerar_dataset_captcha(NUMERO_DE_IMAGENS, PASTA_DESTINO, TAMANHO_CAPTCHA)