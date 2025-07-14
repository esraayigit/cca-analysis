# 🚀 ÜRÜN ÖNERİ SİSTEMİ - KULLANIM REHBERİ

## 📁 DOSYA AÇIKLAMALARI

### 1. `urun_oneri_sistemi.py` - **Standart Versiyon**
- Normal boyutlu veriler için (< 100MB)
- Basit kullanım
- Tüm özellikler dahil

### 2. `urun_oneri_sistemi_memory_optimized.py` - **RAM-Optimized Versiyon** ⭐ **ÖNERİLEN**
- Büyük veriler için optimize edilmiş
- RAM kullanımını minimize eder
- Chunk processing desteği
- Real-time memory monitoring
- Progress tracking
- Debug mode ile detaylı takip

## ⚡ HANGİSİNİ KULLANACAĞINIZ?

### RAM Problemi Yaşadıysanız → **Memory-Optimized Version**
```bash
python urun_oneri_sistemi_memory_optimized.py
```

### Normal Kullanım → Standart Version
```bash
python urun_oneri_sistemi.py
```

## 🔧 KURULUM

1. **Gereksinimler:**
```bash
pip install -r requirements.txt
```

2. **Dosya Yolu Güncelleme:**
Her iki dosyada da bu satırları kendi bilgisayarınıza göre güncelleyin:
```python
file_path = "/Users/esranuryigit/Desktop/BirliktelikAnalizi_son.xlsx"
output_path = "/Users/esranuryigit/Desktop/urun_onerileri_sonuc.xlsx"
```

## 🎯 MEMORY-OPTIMIZED VERSİYON ÖZELLİKLERİ

### ✅ Memory Optimizations:
- Segment-by-segment processing
- Automatic garbage collection
- Data type optimization (category, int16, int8)
- Sparse matrix usage
- Chunk reading for large files
- Memory usage monitoring

### ✅ User Experience:
- Progress tracking (1/3, 2/3, 3/3)
- Real-time memory usage display
- Better error handling
- Debug mode for troubleshooting

### ✅ Performance Features:
- Low-memory Apriori algorithm
- Efficient transaction encoding
- Automatic file size detection
- Smart chunk processing

## 📊 ÇIKTI DOSYALARI

Her iki versiyon da şunları üretir:

### Excel Dosyası (2 sayfa):
1. **"Öneriler" Sayfası:**
   - Bayi kodu
   - Segment
   - Satılan ürün
   - Önerilen ürün  
   - Confidence, lift, support değerleri

2. **"Kurallar" Sayfası:**
   - Tüm association rules
   - Detaylı istatistikler

## 🐛 DEBUG MODE

Memory-optimized versiyonda debug mode aktif:
```python
main_memory_optimized(debug=True)  # RAM kullanımını gösterir
main_memory_optimized(debug=False) # Normal çalışma
```

## ⚠️ SORUN GİDERME

### RAM Yetersizliği:
1. Memory-optimized versiyon kullanın
2. Debug mode'u açarak memory takip edin
3. Gerekirse min_support değerini artırın (0.03 → 0.05)

### Yavaş Çalışma:
1. Debug mode'u kapatın
2. Chunk size'ı küçültün (büyük dosyalar için)
3. SSD kullanıyorsanız daha iyi performans alırsınız

### Hiç Kural Bulunmuyor:
1. Min_support değerini düşürün (0.03 → 0.01)
2. Min_confidence değerini düşürün (0.5 → 0.3)
3. Veri kalitesini kontrol edin

## 📈 PERFORMANS TAVSİYELERİ

### Küçük Veri (< 50MB):
- Standart versiyon kullanın
- Hızlı sonuç alırsınız

### Orta Veri (50-200MB):
- Memory-optimized kullanın
- Debug mode kapalı

### Büyük Veri (> 200MB):
- Memory-optimized kullanın
- Debug mode açık (ilk çalıştırmada)
- Chunk processing otomatik aktif olur

## 🎉 SONUÇ

**Memory-optimized versiyon** çoğu durumda daha iyi performans ve güvenilirlik sağlar. RAM problemi yaşadığınızda kesinlikle bu versiyonu kullanın!

Sorularınız olursa debug mode açık çalıştırıp log'ları inceleyin. 🚀