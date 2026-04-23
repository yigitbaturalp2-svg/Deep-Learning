import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# task2_models.py dosyasından modellerimizi içe aktarıyoruz
from task2_models import BasicCNN, ImprovedCNN

def train_and_evaluate(model, trainloader, testloader, device, model_name="Model", epochs=5):
    print(f"\n{'='*40}")
    print(f"Eğitim Başlıyor: {model_name}")
    print(f"{'='*40}")

    # 1. Kayıp Fonksiyonu ve Optimizer (Derste istenen gereksinimler)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.to(device)

    # 2. Eğitim Döngüsü
    for epoch in range(epochs):
        model.train() # Modeli eğitim moduna al
        running_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Gradyanları sıfırla
            optimizer.zero_grad()

            # İleri yayılım (Forward), Kayıp (Loss) hesaplama ve Geri yayılım (Backward)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"[{model_name}] Epoch {epoch + 1}/{epochs} tamamlandı. Ortalama Loss: {running_loss / len(trainloader):.4f}")

    # 3. Test (Değerlendirme) Döngüsü
    print(f"{model_name} için Test aşamasına geçiliyor...")
    model.eval() # Modeli test moduna al (Dropout, BatchNorm vb. için kritik!)
    correct = 0
    total = 0
    
    # Test aşamasında gradyan hesaplamaya gerek yok (hafıza tasarrufu)
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # En yüksek olasılığa sahip sınıfı seç
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"!!! {model_name} Test Seti Doğruluğu (Accuracy): %{accuracy:.2f} !!!")
    return accuracy

def main():
    # Ekran kartı (GPU) varsa onu kullan, yoksa işlemciye (CPU) geç
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan Cihaz: {device}")

    # Veri yükleme (Görev 1'deki işlemlerin aynısı)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    batch_size = 64
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Modelleri oluştur
    model1 = BasicCNN()
    model2 = ImprovedCNN()

    # Vaktimiz dar olduğu için hızlı sonuç görmek adına epoch sayısını 5 yaptık. 
    # Raporu hazırlarken daha yüksek accuracy için bunu 10 veya 15 yapabilirsin.
    epoch_count = 5

    # Modelleri sırayla eğit ve test et
    acc1 = train_and_evaluate(model1, trainloader, testloader, device, model_name="MODEL 1 (Temel CNN)", epochs=epoch_count)
    acc2 = train_and_evaluate(model2, trainloader, testloader, device, model_name="MODEL 2 (Geliştirilmiş CNN)", epochs=epoch_count)

    print("\n--- SONUÇ KARŞILAŞTIRMASI ---")
    print(f"Model 1 (Temel) Başarımı: %{acc1:.2f}")
    print(f"Model 2 (Geliştirilmiş) Başarımı: %{acc2:.2f}")

if __name__ == '__main__':
    main()
