import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim

class VGGCaptcha(pl.LightningModule):
    def __init__(self, 
                 num_classes=36, 
                 num_chars=5, 
                 learning_rate=1e-3,
                 # Hiperparâmetros Arquiteturais Dinâmicos
                 in_channels=3,
                 channels=[64, 128, 256, 512],
                 conv_blocks_per_stage=[1, 1, 2, 2],
                 fc_sizes=[1024],
                 dropout_rate=0.5,
                 # O cálculo baseia-se na divisão sucessiva por 2 do MaxPooling
                 input_height=60, 
                 input_width=160):
        
        super(VGGCaptcha, self).__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.num_chars = num_chars
        self.learning_rate = learning_rate

        # =====================================================================
        # CONSTRUTOR DINÂMICO DOS BLOCOS VGG
        # =====================================================================
        layers = []
        current_in_channels = in_channels
        
        for stage_idx, out_channels in enumerate(channels):
            num_convs = conv_blocks_per_stage[stage_idx]
            
            for _ in range(num_convs):
                layers.append(nn.Conv2d(current_in_channels, out_channels, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))
                current_in_channels = out_channels
                
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            
        self.vgg_features = nn.Sequential(*layers)
        # =====================================================================

        # Cálculo dinâmico do tamanho do tensor achatado (Flatten)
        # Cada MaxPooling(stride=2) divide altura e largura por 2
        num_poolings = len(channels)
        final_h = input_height // (2 ** num_poolings)
        final_w = input_width // (2 ** num_poolings)
        feature_size = channels[-1] * final_h * final_w

        # Construtor Dinâmico da Camada Densa (Cabeça de Classificação)
        classifier_layers = []
        current_fc_in = feature_size
        
        for fc_size in fc_sizes:
            classifier_layers.append(nn.Linear(current_fc_in, fc_size))
            classifier_layers.append(nn.ReLU(inplace=True))
            classifier_layers.append(nn.Dropout(p=dropout_rate))
            current_fc_in = fc_size
            
        classifier_layers.append(nn.Linear(current_fc_in, self.num_chars * self.num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        features = self.vgg_features(x)
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