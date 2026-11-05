import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, expansion_ratio=4):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        self.pwconv1 = nn.Linear(dim, expansion_ratio * dim) 
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expansion_ratio * dim, dim)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (B, C, H, W) -> (B, H, W, C)

        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        x = x.permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)
        x = input + x
        return x


class ConvNeXtCaptcha(pl.LightningModule):
    def __init__(self, 
                 num_classes=36, 
                 num_chars=5, 
                 learning_rate=1e-3,
                 weight_decay=0.05,
                 # Hiperparâmetros Arquiteturais Dinâmicos
                 in_channels=3,
                 depths=[3, 3, 9, 3],
                 dims=[96, 192, 384, 768],
                 expansion_ratio=4,
                 dropout_rate=0.0,
                 pool_width=1,  # Preserva a ordem horizontal dos caracteres
                 cosine_t_max=20):

        super(ConvNeXtCaptcha, self).__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.num_chars = num_chars
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.cosine_t_max = cosine_t_max

        # =====================================================================
        # CONSTRUTOR DINÂMICO UNIFICADO DA CONVNEXT (self.features)
        # =====================================================================
        layers = []

        # 1. Stem: Primeira redução espacial (stride 4)
        layers.append(nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4))
        layers.append(nn.GroupNorm(1, dims[0], eps=1e-6))

        # 2. Estágios Principais + Camadas de Downsampling intermediárias
        for i in range(len(depths)):
            # Adiciona downsampling entre estágios (a partir do 2º estágio)
            if i > 0:
                layers.append(nn.GroupNorm(1, dims[i-1], eps=1e-6))
                layers.append(nn.Conv2d(dims[i-1], dims[i], kernel_size=2, stride=2))
            
            # Adiciona os blocos ConvNeXt do estágio atual
            for _ in range(depths[i]):
                layers.append(ConvNeXtBlock(dim=dims[i], expansion_ratio=expansion_ratio))

        self.features = nn.Sequential(*layers)
        # =====================================================================

        # Classificador e Pooling padronizados (idênticos à ResNet)
        self.pool = nn.AdaptiveAvgPool2d((1, pool_width))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(dims[-1] * pool_width, self.num_chars * self.num_classes),
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        logits = logits.view(-1, self.num_chars, self.num_classes)
        return logits

    def compute_loss_and_acc(self, logits, targets):
        loss = 0.0
        total = targets.size(0) * self.num_chars
        for i in range(self.num_chars):
            loss += self.criterion(logits[:, i, :], targets[:, i])

        predicoes = torch.argmax(logits, dim=2)
        corretos = (predicoes == targets).sum().item()
        acc = corretos / total
        return loss, acc

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss, acc = self.compute_loss_and_acc(logits, labels)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss, acc = self.compute_loss_and_acc(logits, labels)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss, acc = self.compute_loss_and_acc(logits, labels)
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cosine_t_max)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
