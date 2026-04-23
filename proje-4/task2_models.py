import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# MODEL 1: LeNet-5 Benzeri Temel CNN Sınıfı
# ---------------------------------------------------------
class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        # CIFAR-10 görüntüleri RGB (3 kanal) ve 32x32 pikseldir.
        # 1. Evrişim (Convolution) Katmanı
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5)
        # Havuzlama (Pooling) Katmanı
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 2. Evrişim Katmanı
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        
        # Düzleştirme sonrası Tam Bağlantılı (Fully Connected) Katmanlar
        # 32 kanal * 5x5 piksel boyutu = 800
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) # CIFAR-10'da 10 sınıf olduğu için çıkış 10

    def forward(self, x):
        # İleri yayılım (Forward pass) adımları
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = nn.Flatten()(x) # Özellik haritalarını düzleştir (Flatten)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # Son katmanda aktivasyon kullanmıyoruz (CrossEntropyLoss kendi içinde hallediyor)
        return x

# ---------------------------------------------------------
# MODEL 2: İyileştirilmiş CNN (Batch Norm ve Dropout Eklenmiş)
# ---------------------------------------------------------
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        # Hiperparametreleri Model 1 ile AYNI tutuyoruz (Ödevin şartı)
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.bn1 = nn.BatchNorm2d(16) # EKSTRA: Batch Normalization
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.bn2 = nn.BatchNorm2d(32) # EKSTRA: Batch Normalization
        
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.dropout = nn.Dropout(p=0.5) # EKSTRA: Dropout (Aşırı öğrenmeyi/ezberlemeyi önler)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = nn.Flatten()(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x) # Dropout'u Tam Bağlantılı katman arasına ekledik
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
