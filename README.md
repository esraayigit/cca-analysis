# cca-analysis
Canonical Correlation Analysis (CCA) ve High-Dimensional CCA Uygulaması
Bu kod, Canonical Correlation Analysis (CCA) ve Yüksek Boyutlu Canonical Correlation Analysis (HD-CCA) algoritmalarını uygular ve elde edilen projeksiyonları görselleştirir.

# Kapsanan Adımlar:
# Veri Yükleme ve Görselleştirme:

utils.getdata() ile iki veri kümesi (X, Y) yüklenir.
utils.plotdata() kullanılarak bu veri kümeleri görselleştirilir.
# Canonical Correlation Analysis (CCA):

Veriler ortalamaları çıkarılarak merkezileştirilir.
Kovaryans matrisleri hesaplanır.
Eigenvalue ve eigenvector analiziyle en iyi eşleşen yönler (wx, wy) bulunur.
CCA filtreleriyle verilerin projeksiyonu görselleştirilir.
# High-Dimensional CCA (HD-CCA):

Yüksek boyutlu veri kümeleri utils.getHDdata() ile yüklenir.
Singular Value Decomposition (SVD) kullanılarak en iyi eşleşen yönler hesaplanır.
Projeksiyonlar görselleştirilir.
# Sonuçların Görselleştirilmesi:

Projeksiyonlar ve CCA filtreleri grafiklerle açıklanır.
İki boyutlu ve yüksek boyutlu veriler için ayrı ayrı görselleştirme sağlanır.
# Kullanıcı Notları:
utils.py dosyasının kod ile aynı klasörde bulunması gerekir.
CCA filtrelerinin ve projeksiyonların doğruluğu görsel olarak incelenebilir.
Yüksek boyutlu veri kümesi işlemlerinde SVD algoritmasının hesaplama yeteneği kullanılır.
Bu commit, hem düşük boyutlu hem de yüksek boyutlu veri kümelerinde CCA yöntemlerini denemek ve sonuçları anlamak için örnek bir uygulama sunar.
