import os
import string
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl

# Vocabulário padrão para CAPTCHAs alfanuméricos
VOCABULARIO = string.ascii_uppercase + string.digits
char2idx = {char: idx for idx, char in enumerate(VOCABULARIO)}
idx2char = {idx: char for idx, char in enumerate(VOCABULARIO)}

class CaptchaDataset(Dataset):
    """
    Dataset customizado para ler as imagens de CAPTCHA e seus rótulos.
    """
    def __init__(self, diretorio_dados, transform=None):
        self.diretorio_dados = diretorio_dados
        self.arquivos_imagem = [f for f in os.listdir(diretorio_dados) if f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.arquivos_imagem)

    def __getitem__(self, idx):
        nome_arquivo = self.arquivos_imagem[idx]
        caminho_imagem = os.path.join(self.diretorio_dados, nome_arquivo)
        
        # Extrai o texto do CAPTCHA do nome do arquivo (antes do '_')
        texto_captcha = nome_arquivo.split('_')[0]
        label_tensor = torch.tensor([char2idx[char] for char in texto_captcha], dtype=torch.long)
        
        imagem = Image.open(caminho_imagem).convert("RGB")
        
        if self.transform:
            imagem = self.transform(imagem)
            
        return imagem, label_tensor


class CaptchaDataModule(pl.LightningDataModule):
    """
    DataModule paramétrico para encapsular o Dataset e os DataLoaders.
    """
    def __init__(self, 
                 diretorio_dados="dataset_captcha", 
                 batch_size=32, 
                 num_workers=4,
                 # Hiperparâmetros Dinâmicos de Divisão (Split)
                 train_split=0.70,
                 val_split=0.15,
                 # O test_split será calculado automaticamente: 1.0 - train - val
                 random_seed=42,
                 # Hiperparâmetros de Transformação de Imagem
                 resize_dims=None, # Ex: (60, 160). Se None, não redimensiona.
                 normalize_mean=[0.485, 0.456, 0.406], # Padrão ImageNet
                 normalize_std=[0.229, 0.224, 0.225]):
        
        super().__init__()
        self.diretorio_dados = diretorio_dados
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.random_seed = random_seed
        
        # =====================================================================
        # CONSTRUTOR DINÂMICO DE TRANSFORMAÇÕES
        # =====================================================================
        transforms_list = []
        
        if resize_dims is not None:
            transforms_list.append(transforms.Resize(resize_dims))
            
        transforms_list.append(transforms.ToTensor())
        
        if normalize_mean is not None and normalize_std is not None:
            transforms_list.append(transforms.Normalize(mean=normalize_mean, std=normalize_std))
            
        self.transform = transforms.Compose(transforms_list)
        # =====================================================================

    def setup(self, stage=None):
        """
        Calcula as divisões baseado nas frações fornecidas.
        """
        dataset_completo = CaptchaDataset(self.diretorio_dados, transform=self.transform)
        tamanho_total = len(dataset_completo)
        
        tam_treino = int(self.train_split * tamanho_total)
        tam_val = int(self.val_split * tamanho_total)
        tam_teste = tamanho_total - tam_treino - tam_val # Garante que a soma dê o total exato
        
        # Generator com a seed passada por parâmetro para garantir reprodutibilidade
        gerador_seed = torch.Generator().manual_seed(self.random_seed)
        
        self.dataset_treino, self.dataset_val, self.dataset_teste = random_split(
            dataset_completo, 
            [tam_treino, tam_val, tam_teste],
            generator=gerador_seed
        )

    def train_dataloader(self):
        return DataLoader(self.dataset_treino, 
                          batch_size=self.batch_size, 
                          shuffle=True, 
                          num_workers=self.num_workers, 
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, 
                          batch_size=self.batch_size, 
                          shuffle=False, 
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset_teste, 
                          batch_size=self.batch_size, 
                          shuffle=False, 
                          num_workers=self.num_workers)