import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim


class AlexNetCaptcha(pl.LightningModule):
    def __init__(self,
                 num_classes=36,
                 num_chars=5,
                 learning_rate=1e-4,
                 weight_decay=0.0,
                 # Hiperparâmetros Arquiteturais Dinâmicos
                 in_channels=3,
                 channels=[96, 256, 384, 384, 256],
                 fc_sizes=[4096, 4096],
                 dropout_rate=0.5,
                 # Stride da 1ª convolução. O artigo usa 4 em imagens 227x227; em 60x160
                 # isso reduz a altura a 2 antes do último pooling, que então falha
                 stem_stride=2,
                 # Largura do pooling final: preserva a ordem horizontal dos caracteres
                 pool_width=9):

        super(AlexNetCaptcha, self).__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.num_chars = num_chars
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # =====================================================================
        # EXTRATOR CONVOLUCIONAL (5 convoluções, como no artigo)
        # =====================================================================
        # Para 60x160: 30x80 -> pool 14x39 -> pool 6x19, mantida até a saída.
        # O último MaxPool fica em self.pool para o Grad-CAM usar o mapa 6x19.
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=11, stride=stem_stride, padding=5),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(channels[0], channels[1], kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[2], channels[3], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[3], channels[4], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # =====================================================================

        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d((1, pool_width)),
        )

        # Construtor Dinâmico da Camada Densa (Cabeça de Classificação)
        classifier_layers = []
        current_fc_in = channels[-1] * pool_width

        for fc_size in fc_sizes:
            classifier_layers.append(nn.Dropout(p=dropout_rate))
            classifier_layers.append(nn.Linear(current_fc_in, fc_size))
            classifier_layers.append(nn.ReLU(inplace=True))
            current_fc_in = fc_size

        classifier_layers.append(nn.Linear(current_fc_in, self.num_chars * self.num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

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
