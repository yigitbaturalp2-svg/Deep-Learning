import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os
from task2_models import BasicCNN, ImprovedCNN, StandardCNN

def train_and_evaluate(model, trainloader, testloader, device, model_name, epochs=5):
    print(f"\n>>> {model_name} Eğitimi Başlıyor...")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = {'loss': [], 'accuracy': []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(trainloader)
        history['loss'].append(avg_loss)
        
        # Epoch sonu accuracy hesapla
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        history['accuracy'].append(acc)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: %{acc:.2f}")

    return model, history

def save_plots(history_dict, model_names, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Loss Plot
    plt.figure(figsize=(10, 5))
    for name in model_names:
        plt.plot(history_dict[name]['loss'], label=f'{name} Loss')
    plt.title('Modellerin Eğitim Kayıpları (Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/comparison_loss.png')
    plt.close()

    # Accuracy Plot
    plt.figure(figsize=(10, 5))
    for name in model_names:
        plt.plot(history_dict[name]['accuracy'], label=f'{name} Accuracy')
    plt.title('Modellerin Test Doğrulukları (Accuracy)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/comparison_accuracy.png')
    plt.close()

def save_confusion_matrix(model, testloader, device, model_name, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title(f'{model_name} Karmaşıklık Matrisi')
    plt.ylabel('Gerçek Sınıf')
    plt.xlabel('Tahmin Edilen Sınıf')
    plt.savefig(f'{output_dir}/{model_name.replace(" ", "_")}_cm.png')
    plt.close()

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan Cihaz: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    batch_size = 64
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    models_to_train = {
        "Model 1 (Temel)": BasicCNN(),
        "Model 2 (Geliştirilmiş)": ImprovedCNN(),
        "Model 3 (Standart)": StandardCNN()
    }

    all_histories = {}
    epoch_count = 5

    for name, model in models_to_train.items():
        trained_model, history = train_and_evaluate(model, trainloader, testloader, device, name, epochs=epoch_count)
        all_histories[name] = history
        save_confusion_matrix(trained_model, testloader, device, name)

    save_plots(all_histories, list(models_to_train.keys()))
    print("\nBütün modeller eğitildi ve görseller 'results' klasörüne kaydedildi.")

if __name__ == '__main__':
    main()
