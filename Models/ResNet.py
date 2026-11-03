import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim


class BasicBlock(nn.Module):
    """Bloco residual da ResNet-18/34: duas convoluções 3x3 + skip connection."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Projeção 1x1 no atalho quando a dimensão ou resolução muda
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ResNetCaptcha(pl.LightningModule):
    def __init__(self,
                 num_classes=36,
                 num_chars=5,
                 learning_rate=1e-3,
                 weight_decay=0.0,
                 # Hiperparâmetros Arquiteturais Dinâmicos
                 in_channels=3,
                 channels=[64, 128, 256, 512],
                 blocks_per_stage=[2, 2, 2, 2],  # [2,2,2,2] = ResNet-18, [3,4,6,3] = ResNet-34
                 dropout_rate=0.3,
                 # Largura do pooling final: preserva a ordem horizontal dos caracteres
                 pool_width=4):

        super(ResNetCaptcha, self).__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.num_chars = num_chars
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # =====================================================================
        # CONSTRUTOR DINÂMICO DA RESNET
        # =====================================================================
        # Stem sem stride/maxpool: as imagens de CAPTCHA (60x160) são pequenas
        layers = [
            nn.Conv2d(in_channels, channels[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        ]
        current_in_channels = channels[0]

        for stage_idx, out_channels in enumerate(channels):
            for block_idx in range(blocks_per_stage[stage_idx]):
                # Apenas o primeiro bloco de cada estágio (exceto o 1º) reduz a resolução
                stride = 2 if (block_idx == 0 and stage_idx > 0) else 1
                layers.append(BasicBlock(current_in_channels, out_channels, stride))
                current_in_channels = out_channels

        self.features = nn.Sequential(*layers)
        # =====================================================================

        self.pool = nn.AdaptiveAvgPool2d((1, pool_width))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(channels[-1] * pool_width, self.num_chars * self.num_classes),
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        features = self.pool(self.features(x))
        flattened = torch.flatten(features, 1)
        logits = self.classifier(flattened)
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
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
