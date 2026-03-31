# YZM304 Derin Öğrenme | I. Proje Modülü: Öğrenci Depresyon Klasifikasyonu

Bu proje, YZM304 Derin Öğrenme dersi I. Proje Modülü kapsamında "Öğrenci Depresyon Verileri" (Depression Student Dataset.csv) veri seti üzerinde geliştirilmiştir. Çalışma, laboratuvar dersinde öğretilen yapının Nesne Yönelimli (OOP) versiyonunu ve PyTorch ile Scikit-Learn kütüphanelerinin eşdeğer kıyaslamalarını içermektedir.

Tüm sistemi eşzamanlı çalıştırmak ve grafiksel kıyaslamaları doğrudan ekrana almak için **`main.py`** dosyasını çalıştırmanız yeterlidir.

---

## 1. Giriş

Projenin temel amacı, demografik veriler ve yaşam alışkanlıkları baz alınarak bireyin depresyon durumunu (`Depression`) tahmin edebilmektir. Elde edilecek hedef değişken **Yes/No** (2 Sınıf) olduğu için yapı ikili sınıflandırmaya (Binary Classification) dönüştürülüp, laboratuvardaki altyapı bozulmadan Softmax ve Cross-Entropy entegrasyonlarıyla çoklu modül gibi (2 çıkış nöronuyla) çalışacak şekilde dizayn edilmiştir.

Kurulan modellerin Overfitting (aşırı öğrenme) durumunu ölçebilmek amacıyla test setinin haricinde Modele özel `Doğrulama (Dev)` seti mimariye entegre edilmiştir.

---

## 2. Metot

### Veri Ön İşleme
Veri manipülasyonu işlemleri için nesne yönelimli `DataProcessor` modülü kodlanmıştır. Eksik değer manipülasyonlarından ve dinamik `One-Hot Encoding` mimarisinden geçen veriler `StandardScaler` ile ölçeklenmiştir. Veri seti tabakalı (stratify) teknik ile **%70 Eğitim, %15 Doğrulama ve %15 Test** seti olacak şekilde 3 parçaya koparılmıştır.

### Model Mimarileri ve Test Süreci
Bütün yapılar **1000 iterasyon** ve Stokastik Gradyan İniş (SGD) kullanılıp **0.05 Öğrenme Oranıyla** kurulmuştur. Ağ ağırlıkları (Weights) yönerge şartını ispatlamak adına her modelde Normal Dağılım `0.01` varyansıyla (Random Seed 42) başlatılmıştır:
1. **NumPy Temel Model:** Yönergedeki gibi 1 gizli katman kullanan temel ağ.
2. **NumPy Çok Katmanlı Model:** Alternatif optimizasyon analizleri için gizli katman sayısı artırılmış NumPy modülü.
3. **Scikit-Learn Modeli (MLPClassifier):** Mimariyle ayna nöron dizilimine sahip Sklearn kütüphane modülü.
4. **PyTorch Modeli:** Ağırlık başlatma değerleri otomasyon yerine manuel enjekte edilmiş eşdeğer ağ.

---

## 3. Bulgular

4 farklı modülün test verileri üzerinde çalıştırılmasıyla oluşan Confusion Matrix (Karmaşıklık Matrisi) ve Öğrenme Eğrileri ana dizine çıkartılmıştır:

### Modül 1 - Temel NumPy Modeli (1 Gizli Katman)
Laboratuvar kodu temel alınarak 1 gizli katman ile yürütülen çoklu sınıflandırıcının sonuç tablosu:
![Öğrenme Eğrisi](EkranGörüntüleri/Figure_1.png)
![Karmaşıklık Matrisi](EkranGörüntüleri/Figure_2.png)

### Modül 2 - Çok Katmanlı NumPy Modeli
Tahmin yeteneğinin değişimi adına katman sayısı artırılarak özelleştirilmiş ağ modeli:
![Öğrenme Eğrisi](EkranGörüntüleri/Figure_3.png)
![Karmaşıklık Matrisi](EkranGörüntüleri/Figure_4.png)

### Modül 3 - Scikit-Learn Modeli
![Öğrenme Eğrisi](EkranGörüntüleri/Figure_5.png)
![Karmaşıklık Matrisi](EkranGörüntüleri/Figure_6.png)

### Modül 4 - PyTorch Modeli
Gradient hesaplamalarını otomatik gerçekleştiren PyTorch yapısı, bizim yazdığımız Custom model ile aynı çizgide stabil şekilde birleşmiştir.
![Öğrenme Eğrisi](EkranGörüntüleri/Figure_7.png)
![Karmaşıklık Matrisi](EkranGörüntüleri/Figure_8.png)

---

## 4. Tartışma ve Sonuç

Bu projede NumPy üzerinden ileri/geriye yayılım ve çapraz entropi formülleri kusursuz çalıştırılmış olup; modern kütüphaneler (`PyTorch`, `Scikit-Learn`) ile eşdeğer kapasiteye sahip klasifikasyon algoritmaları tamamen sıfırdan üretilmiş ve Depresyon tespiti sistemine mükemmel uygulanabilmiştir. 

Yeni veri seti, az sayıda güçlü bağımsız değişkene sahip (Feature Boyutu optimize edilmiş) bir yapıda olduğu için eski çok-öznitelikli verilerdeki hantallık çözülmüştür. Bu sayede modeller donanım limitlerine takılmadan saniyeler içerisinde 1000 iterasyon barajını aşabilmekte ve hedef değişken (Binary) nispeten homojen yayıldığından Karmaşıklık Matrislerinde hiçbir manipülasyon olmaksızın en az hata (Underfitting / Bias yoksunu) üretilecek şekilde eğitilebilmektedir.
