import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os

def main():
    print(f"{'='*50}")
    print("GÖREV 5: Model 4 ile Karşılaştırma İçin Tam CNN Eğitimi")
    print(f"{'='*50}\n")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan Cihaz: {device}")

    # Model 4 ile aynı veri setini ve ön işlemeyi kullanıyoruz
    transform = transforms.Compose([
        transforms.Resize((64, 64)), # ResNet için biraz büyütüyoruz (32x32 de olur ama 64x64 daha iyi sonuç verir)
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    batch_size = 64
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    # ResNet18'i sıfırdan eğitiyoruz (pretrained=False)
    print("ResNet18 modeli oluşturuluyor...")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    history = {'loss': [], 'accuracy': []}

    print("Eğitim başlıyor...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(trainloader)
        history['loss'].append(avg_loss)

        # Accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        history['accuracy'].append(acc)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: %{acc:.2f}")

    # Sonuçları kaydet
    os.makedirs('results', exist_ok=True)
    
    # Loss Plot
    plt.figure()
    plt.plot(history['loss'])
    plt.title('Model 5 (Full ResNet18) Eğitim Kaybı')
    plt.savefig('results/Model_5_loss.png')
    
    # Accuracy Plot
    plt.figure()
    plt.plot(history['accuracy'])
    plt.title('Model 5 (Full ResNet18) Test Doğruluğu')
    plt.savefig('results/Model_5_accuracy.png')

    # CM
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title('Model 5 Karmaşıklık Matrisi')
    plt.savefig('results/Model_5_cm.png')
    
    print("\nModel 5 eğitimi tamamlandı ve görseller kaydedildi.")

if __name__ == '__main__':
    main()
