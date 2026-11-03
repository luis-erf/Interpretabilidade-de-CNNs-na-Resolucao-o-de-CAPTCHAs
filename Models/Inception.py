import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim

# (out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, out_pool)
inception_cfg = [
    (64, 96, 128, 16, 32, 32),     # 3a -> 256
    (128, 128, 192, 32, 96, 64),   # 3b -> 480
    "pool",
    (192, 96, 208, 16, 48, 64),    # 4a -> 512
    (160, 112, 224, 24, 64, 64),   # 4b -> 512
    (128, 128, 256, 24, 64, 64),   # 4c -> 512
    (112, 144, 288, 32, 64, 64),   # 4d -> 528
    (256, 160, 320, 32, 128, 128), # 4e -> 832
]


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class InceptionModule(nn.Module):
    """Quatro ramos paralelos (1x1, 3x3, 5x5, pooling) concatenados no eixo dos canais."""
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_pool):
        super(InceptionModule, self).__init__()
        self.branch1 = ConvBlock(in_channels, out_1x1, 1)
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, red_3x3, 1),
            ConvBlock(red_3x3, out_3x3, 3, padding=1),
        )
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, red_5x5, 1),
            ConvBlock(red_5x5, out_5x5, 5, padding=2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, out_pool, 1),
        )
        self.out_channels = out_1x1 + out_3x3 + out_5x5 + out_pool

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


class InceptionCaptcha(pl.LightningModule):
    def __init__(self,
                 num_classes=36,
                 num_chars=5,
                 learning_rate=1e-3,
                 weight_decay=0.0,
                 # Hiperparâmetros Arquiteturais Dinâmicos
                 in_channels=3,
                 stem_channels=64,
                 num_modules=None,  # None usa todos os módulos de inception_cfg (GoogLeNet completa)
                 dropout_rate=0.4,
                 # Largura do pooling final: preserva a ordem horizontal dos caracteres
                 pool_width=4):

        super(InceptionCaptcha, self).__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.num_chars = num_chars
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # =====================================================================
        # CONSTRUTOR DINÂMICO DA INCEPTION (GoogLeNet adaptada a imagens pequenas)
        # =====================================================================
        # Stem leve (um único stride 2): as imagens de CAPTCHA (60x160) são pequenas
        layers = [
            ConvBlock(in_channels, stem_channels, 3, stride=1, padding=1),
            ConvBlock(stem_channels, stem_channels * 2, 3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        current_in_channels = stem_channels * 2

        cfg = inception_cfg
        if num_modules is not None:
            # Conta apenas os módulos Inception, mantendo os poolings intercalados
            kept, count = [], 0
            for item in inception_cfg:
                if item == "pool":
                    kept.append(item)
                elif count < num_modules:
                    kept.append(item)
                    count += 1
            cfg = kept

        for item in cfg:
            if item == "pool":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                module = InceptionModule(current_in_channels, *item)
                layers.append(module)
                current_in_channels = module.out_channels

        self.features = nn.Sequential(*layers)
        # =====================================================================

        self.pool = nn.AdaptiveAvgPool2d((1, pool_width))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(current_in_channels * pool_width, self.num_chars * self.num_classes),
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
