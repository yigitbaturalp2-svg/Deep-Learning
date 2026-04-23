import torch
import torchvision
import torchvision.transforms as transforms

def main():
    # 1. Ön İşleme (Preprocessing): Görüntüleri Tensor'a çevir ve normalize et
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Batch size: Modeli eğitirken veriyi kaçarlı gruplar halinde vereceğimiz
    batch_size = 64

    # 2. Eğitim veri setini indir ve yükle
    print("Eğitim veri seti indiriliyor/yükleniyor...")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    # 3. Test veri setini indir ve yükle
    print("Test veri seti indiriliyor/yükleniyor...")
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    # Sınıf isimleri
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print(f"Eğitim seti başarıyla yüklendi. Boyut: {len(trainset)}")
    print(f"Test seti başarıyla yüklendi. Boyut: {len(testset)}")

if __name__ == '__main__':
    main()
