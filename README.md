# Ürün Öneri Sistemi - Apriori Algoritması

Bu Python scripti, bayi satış verilerine dayalı olarak Apriori algoritmasını kullanarak ürün önerileri geliştiren bir sistemdir.

## Özellikler

- **Segment Bazlı Analiz**: Her segment için ayrı ayrı Apriori algoritması çalıştırır
- **Ürün Grubu Filtreleme**: Aynı ürün grubu içindeki kuralları dışarıda bırakır
- **Öneri Sistemi**: Son ay antecedent ürünü satıp consequent ürünü satmayan bayileri tespit eder
- **Satış Birimi Analizi**: Her ürün için detaylı satış istatistikleri
- **Excel Çıktısı**: Sonuçları patrona sunmak üzere Excel formatında kaydeder

## Gereksinimler

Scripti çalıştırmadan önce gerekli Python paketlerini yükleyin:

```bash
pip install -r requirements.txt
```

## Kullanım

1. **Veri Dosyası Hazırlığı**: Excel dosyanızın aşağıdaki sütunları içerdiğinden emin olun:
   - cairkod (Bayi kodu)
   - yil (Yıl)
   - aylik (Ay)
   - urokod_aylik (Ürün kodu)
   - deger (Satış değeri)
   - StockAd (Ürün adı)
   - segment (Segment)
   - StockId (Stok ID)
   - stockkod (Stok kodu)
   - Ürün Grubu (Ürün kategorisi)

2. **Dosya Yolu Güncelleme**: `urun_oneri_sistemi.py` dosyasındaki dosya yollarını kendi sisteminize göre güncelleyin:
   ```python
   file_path = "/Users/esranuryigit/Desktop/BirliktelikAnalizi_son.xlsx"
   output_path = "/Users/esranuryigit/Desktop/urun_onerileri_sonuc.xlsx"
   ```

3. **Scripti Çalıştırma**:
   ```bash
   python urun_oneri_sistemi.py
   ```

## Algoritma Parametreleri

- **Minimum Support**: %3 (0.03)
- **Minimum Confidence**: %50 (0.5)
- **Segment Bazlı**: Her segment için ayrı analiz
- **Ürün Grubu Filtreleme**: Farklı ürün grupları arasındaki ilişkiler

## Çıktı Dosyaları

Script iki adet Excel sayfası içeren bir dosya oluşturur:

### 1. Ürün Önerileri Sayfası
- `cairkod`: Bayi kodu
- `segment`: Segment bilgisi
- `antecedents`: Satılan ürün(ler)
- `consequents`: Önerilen ürün(ler)
- `confidence`: Güven düzeyi
- `lift`: Kaldıraç değeri
- `support`: Destek değeri
- `onerilen_urun`: Spesifik önerilen ürün
- `toplam_satis`: Ürünün toplam satışı
- `ortalama_satis`: Ürünün ortalama satışı
- `satis_adedi`: Satış adedi
- `urun_grubu`: Ürün kategorisi

### 2. Association Rules Sayfası
- Tüm bulunan kuralların detaylı listesi
- Segment bazında filtrelenmiş kurallar

## Özet Rapor

Script çalıştıktan sonra konsoldaşağıdakileri görüntüler:
- Toplam öneri sayısı
- Öneri alan bayi sayısı
- Segment bazında özet istatistikler
- En yüksek confidence'a sahip öneriler

## Teknik Detaylar

### Veri İşleme
1. Excel dosyası pandas ile okunur
2. Her bayi-ay kombinasyonu için transaction matrix oluşturulur
3. Ürün grupları belirlenir

### Apriori Algoritması
1. Her segment için ayrı frequent itemsets bulunur
2. Association rules oluşturulur
3. Aynı ürün grubu içindeki kurallar filtrelenir

### Öneri Sistemi
1. Son ay verileri analiz edilir
2. Antecedent ürünü satan ancak consequent ürünü satmayan bayiler tespit edilir
3. Satış birim bilgileri eklenir

## Hata Ayıklama

Eğer script hata verirse:
1. Excel dosya yolunu kontrol edin
2. Sütun isimlerinin doğru olduğundan emin olun
3. Veri formatının uygun olduğunu kontrol edin
4. Gerekli Python paketlerinin yüklü olduğunu doğrulayın

## Özelleştirme

Parametreleri değiştirmek için `main()` fonksiyonundaki değerleri güncelleyin:
```python
rules_df = run_apriori_by_segment(transactions, product_groups, 
                                 min_support=0.03,      # Minimum support
                                 min_confidence=0.5)    # Minimum confidence
```
