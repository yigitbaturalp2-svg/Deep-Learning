import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

def extract_features(model, dataloader, device):
    features = []
    labels_list = []
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            # Özellikleri çıkar ve CPU'ya alıp numpy dizisine çevir
            outputs = model(inputs)
            features.append(outputs.cpu().numpy())
            labels_list.append(labels.numpy())
            
    # Listeleri tek bir numpy dizisinde birleştir
    features = np.vstack(features)
    labels_list = np.concatenate(labels_list)
    return features, labels_list

def main():
    print(f"{'='*50}")
    print("GÖREV 4: Hibrit Model (CNN Özellik Çıkarımı + SVM)")
    print(f"{'='*50}\n")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan Cihaz: {device}\n")

    # 1. Veri Yükleme (Öncekiyle aynı standart ön işleme)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    batch_size = 64
    # Note: download=False because we assume Task 1 finished download
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2) # Özellik çıkarırken shuffle=False daha güvenlidir
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 2. Önceden Eğitilmiş Modeli Yükleme (ResNet18)
    print("Önceden eğitilmiş ResNet18 modeli yükleniyor...")
    # weights='DEFAULT' ile ImageNet üzerinde eğitilmiş en iyi ağırlıkları alıyoruz
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Sınıflandırma katmanını (fully connected layer) iptal edip Identity (Etkisiz Eleman) yapıyoruz
    # Böylece model bize 1000 sınıflık tahmin yerine, 512 boyutlu özellik vektörünü verecek
    resnet.fc = torch.nn.Identity()
    resnet = resnet.to(device)

    # 3. Özellik Çıkarımı ve Kaydetme (.npy formatında)
    print("Eğitim setinden özellikler çıkarılıyor (Bu işlem birkaç dakika sürebilir)...")
    train_features, train_labels = extract_features(resnet, trainloader, device)
    
    print("Test setinden özellikler çıkarılıyor...")
    test_features, test_labels = extract_features(resnet, testloader, device)

    # Dosyaları kaydetme
    os.makedirs('features', exist_ok=True)
    np.save('features/train_features.npy', train_features)
    np.save('features/train_labels.npy', train_labels)
    np.save('features/test_features.npy', test_features)
    np.save('features/test_labels.npy', test_labels)

    # ÖDEV GEREKSİNİMİ: Boyutları ekrana yazdırmak
    print("\n--- Kaydedilen .npy Dosyalarının Boyutları ---")
    print(f"Eğitim Özellikleri (X_train): {train_features.shape}")
    print(f"Eğitim Etiketleri (y_train): {train_labels.shape}")
    print(f"Test Özellikleri (X_test): {test_features.shape}")
    print(f"Test Etiketleri (y_test): {test_labels.shape}\n")

    # 4. Kanonik Makine Öğrenmesi Modeli Eğitimi (SVM)
    print("Destek Vektör Makinesi (SVM) eğitiliyor...")
    # SVM eğitimi özellik seti büyük olduğunda yavaş olabilir, hızlı sonuç için max_iter koyulabilir ama standart bırakıyoruz
    svm_model = SVC(kernel='linear', random_state=42) 
    svm_model.fit(train_features, train_labels)

    # 5. Test ve Değerlendirme
    print("Test seti üzerinde tahminler yapılıyor...")
    predictions = svm_model.predict(test_features)
    
    accuracy = accuracy_score(test_labels, predictions) * 100
    print(f"\n!!! Hibrit Model (ResNet18 Özellikleri + SVM) Test Doğruluğu: %{accuracy:.2f} !!!")
    
    # Rapor için karmaşıklık matrisi (Confusion Matrix)
    print("\nKarmaşıklık Matrisi (Raporuna eklemen için):")
    print(confusion_matrix(test_labels, predictions))

if __name__ == '__main__':
    main()
