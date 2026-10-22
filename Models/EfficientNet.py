import math

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim

basic_mb_params = [
    # expand_ratio(t), channels(c), repeats(n), stride(s), kernel_size(k)
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]

alpha, beta = 1.2, 1.1

scale_values = {
    # (phi, resolution, dropout)
    "b0": (0, 224, 0.2),
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super(ConvBlock, self).__init__()
        self.cnnblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        return self.cnnblock(x)

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Reescala cada canal pelo peso de atenção aprendido
        return x * self.se(x)

class MBBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 ratio, reduction=4, survival_prob=0.8):
        super(MBBlock, self).__init__()

        self.survival_prob = survival_prob
        self.use_residual = in_channels == out_channels and stride == 1

        hidden_dim = in_channels * ratio
        self.expand = in_channels != hidden_dim

        # O SE da EfficientNet reduz para 1/4 dos canais de ENTRADA do bloco
        reduced_dim = max(1, int(in_channels / reduction))

        if self.expand:
            # Expansão pointwise (1x1)
            self.expand_conv = ConvBlock(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0)

        self.conv = nn.Sequential(
            # Convolução depthwise
            ConvBlock(hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim),
            SqueezeExcitation(hidden_dim, reduced_dim),
            # Projeção pointwise (1x1) linear, sem ativação
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def stochastic_depth(self, x):
        # Durante o treino, descarta o bloco inteiro de alguns exemplos do lote
        if not self.training:
            return x
        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
        return torch.div(x, self.survival_prob) * binary_tensor

    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs
        x = self.conv(x)

        if self.use_residual:
            return self.stochastic_depth(x) + inputs
        return x


class EfficientNetCaptcha(pl.LightningModule):
    def __init__(self,
                 num_classes=36,
                 num_chars=5,
                 learning_rate=1e-3,
                 # Hiperparâmetros Arquiteturais Dinâmicos
                 in_channels=3,
                 version="b0",
                 survival_prob=0.8,
                 dropout_rate=None,
                 last_channels=1280):

        super(EfficientNetCaptcha, self).__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.num_chars = num_chars
        self.learning_rate = learning_rate

        phi, _, default_dropout = scale_values[version]
        # A resolução de scale_values não é usada: as imagens do CAPTCHA mantêm o tamanho do dataset
        depth_factor = alpha ** phi
        width_factor = beta ** phi
        if dropout_rate is None:
            dropout_rate = default_dropout

        # =====================================================================
        # CONSTRUTOR DINÂMICO DA EFFICIENTNET
        # =====================================================================
        stem_channels = int(32 * width_factor)
        layers = [ConvBlock(in_channels, stem_channels, kernel_size=3, stride=2, padding=1)]
        current_in_channels = stem_channels

        for expand_ratio, channels, repeats, stride, kernel_size in basic_mb_params:
            # Arredonda os canais para múltiplos de 4 após o escalonamento de largura
            out_channels = 4 * math.ceil(int(channels * width_factor) / 4)
            num_layers = math.ceil(repeats * depth_factor)

            for layer in range(num_layers):
                layers.append(
                    MBBlock(
                        current_in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        # Apenas o primeiro bloco de cada estágio reduz a resolução
                        stride=stride if layer == 0 else 1,
                        padding=kernel_size // 2,
                        ratio=expand_ratio,
                        survival_prob=survival_prob,
                    )
                )
                current_in_channels = out_channels

        final_channels = math.ceil(last_channels * width_factor)
        layers.append(ConvBlock(current_in_channels, final_channels, kernel_size=1, stride=1, padding=0))

        self.features = nn.Sequential(*layers)
        # =====================================================================

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(final_channels, self.num_chars * self.num_classes),
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
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
