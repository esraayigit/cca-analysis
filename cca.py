import numpy as np
import matplotlib.pyplot as plt
import utils  # utils.py dosyasını ödev klasörüne koyduğunuzdan emin olun

# Verileri yükle
X, Y = utils.getdata()

# Verileri görselleştir
p1, p2 = utils.plotdata(X, Y)
plt.show()

def CCA(X, Y):
    # Verilerin ortalamasını çıkararak merkezileştirme
    X_mean = X - X.mean(axis=1, keepdims=True)
    Y_mean = Y - Y.mean(axis=1, keepdims=True)

    # Kovaryans matrisleri
    C_xx = X_mean @ X_mean.T / X.shape[1]
    C_yy = Y_mean @ Y_mean.T / Y.shape[1]
    C_xy = X_mean @ Y_mean.T / X.shape[1]

    # Kovaryans matrislerinin eigenvalue ve eigenvector analizi
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(C_xx) @ C_xy @ np.linalg.inv(C_yy) @ C_xy.T)

    # En büyük eigenvalue'ya karşılık gelen eigenvector'lar
    idx = np.argmax(eig_vals)
    wx = eig_vecs[:, idx]
    wy = eig_vecs[:, idx]  # Burada wx ve wy aynı eigenvector ile sırasıyla elde edilir

    return wx.real, wy.real
    
# CCA'yı uygula
wx, wy = CCA(X, Y)

# Sonuçları görselleştir
p1, p2 = utils.plotdata(X, Y)
p1.arrow(0, 0, -wx[0], -wx[1], color='red', width=0.1, label='wx')
p2.arrow(0, 0, wy[0], wy[1], color='red', width=0.1, label='wy')
plt.show()

# Projeksiyonları çiz
plt.figure(figsize=(8, 4))
plt.plot(wx.dot(X), label="Projection X (wx)")
plt.plot(wy.dot(Y), label="Projection Y (wy)", color='orange')
plt.title("Projection of Data on CCA Filters")
plt.legend()

# Ekstra açıklamalar
plt.xlabel('Data points')
plt.ylabel('Projection values')
plt.grid(True)
plt.show()

def CCA_HD(X, Y):
    """
    Yüksek boyutlu veriler için Canonical Correlation Analysis (HD-CCA)
    """

    # Verileri merkezileştirme
    X_mean = X - X.mean(axis=0)
    Y_mean = Y - Y.mean(axis=0)

    # SVD kullanarak en iyi eşleşen yönlerin bulunması
    Ux, Sx, Vx = np.linalg.svd(X_mean, full_matrices=False)
    Uy, Sy, Vy = np.linalg.svd(Y_mean, full_matrices=False)

    # İlk bileşen (direksiyonlar)
    wx = Ux[:, 0]  # İlk bileşen
    wy = Uy[:, 0]

    return wx, wy

# Yüksek boyutlu verileri yükle
X, Y = utils.getHDdata()

# Görselleştirme
utils.plotHDdata(X[:, 0], Y[:, 0])  # Burada sadece ilk bileşenler kullanılmalı
plt.show()

# CCA'yı uygula
wx, wy = CCA_HD(X, Y)

# Yüksek boyutlu verilerle görselleştirme
utils.plotHDdata(wx, wy)  # Burada sadece ilk bileşenler kullanılmalı
plt.show()

# Projeksiyonları çiz
plt.figure(figsize=(6,2))
plt.plot(wx.dot(X))
plt.plot(wy.dot(Y))
plt.show()