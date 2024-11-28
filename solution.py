import numpy as np

def CCA(X, Y):
    """
    Canonical Correlation Analysis (CCA) hesaplama
    X ve Y, aynı sayıda örneğe sahip (NxM) veri matrisleridir.
    """
    # Verilerin ortalamasını çıkararak merkezileştirme
    X_mean = X - X.mean(axis=1, keepdims=True)
    Y_mean = Y - Y.mean(axis=1, keepdims=True)

    # Kovaryans matrisleri
    C_xx = X_mean @ X_mean.T
    C_yy = Y_mean @ Y_mean.T
    C_xy = X_mean @ Y_mean.T

    # Kovaryans matrislerinin eigenvalue ve eigenvector analizi
    eig_vals_x, eig_vecs_x = np.linalg.eig(np.linalg.inv(C_xx) @ C_xy @ np.linalg.inv(C_yy) @ C_xy.T)
    eig_vals_y, eig_vecs_y = np.linalg.eig(np.linalg.inv(C_yy) @ C_xy.T @ np.linalg.inv(C_xx) @ C_xy)

    # En büyük eigenvalue'ya karşılık gelen eigenvector'lar
    wx = eig_vecs_x[:, np.argmax(eig_vals_x)]
    wy = eig_vecs_y[:, np.argmax(eig_vals_y)]

    return wx.real, wy.real

def CCA_HD(X, Y):
    """
    Yüksek boyutlu veriler için Canonical Correlation Analysis (CCA)
    X ve Y yüksek boyutlu (çok sayıda özellik içeren) veri matrisleridir.
    """
    # Verilerin ortalamasını çıkararak merkezileştirme
    X_mean = X - X.mean(axis=0)
    Y_mean = Y - Y.mean(axis=0)

    # SVD kullanarak en iyi eşleşen yönlerin bulunması
    Ux, Sx, Vx = np.linalg.svd(X_mean, full_matrices=False)
    Uy, Sy, Vy = np.linalg.svd(Y_mean, full_matrices=False)

    # İlk bileşenleri al (bu, en yüksek korelasyonu yakalayacak bileşenlerdir)
    wx = Ux[:, 0]  # İlk bileşen
    wy = Uy[:, 0]

    return wx, wy