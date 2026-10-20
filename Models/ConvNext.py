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
                 # Hiperparâmetros Arquiteturais Dinâmicos
                 in_channels=3,
                 depths=[3, 3, 9, 3],
                 dims=[96, 192, 384, 768],
                 expansion_ratio=4,
                 weight_decay=0.05,
                 cosine_t_max=20):
                     
        super(ConvNeXtCaptcha, self).__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.num_chars = num_chars
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.cosine_t_max = cosine_t_max

        # =====================================================================
        # CONSTRUTOR DINÂMICO DA CONVNEXT
        # =====================================================================
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            nn.GroupNorm(1, dims[0], eps=1e-6) 
        )
        self.downsample_layers.append(stem)

        # Camadas de Redução Dinâmicas baseadas na lista 'dims'
        for i in range(len(dims) - 1):
            downsample_layer = nn.Sequential(
                nn.GroupNorm(1, dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # Estágios principais dinâmicos baseados nas listas 'depths' e 'dims'
        self.stages = nn.ModuleList()
        for i in range(len(depths)):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i], expansion_ratio=expansion_ratio) for _ in range(depths[i])]
            )
            self.stages.append(stage)
        # =====================================================================

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], self.num_chars * self.num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        for i in range(len(self.stages)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            
        x = x.mean([-2, -1]) # Global Average Pooling
        x = self.norm(x)
        logits = self.head(x)
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