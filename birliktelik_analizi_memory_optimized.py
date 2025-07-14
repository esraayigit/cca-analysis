import pandas as pd
import numpy as np
from itertools import combinations
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
import gc
import psutil
import os
from typing import List, Dict, Optional, Tuple
warnings.filterwarnings('ignore')

# Türkçe karakter desteği için
plt.rcParams['font.family'] = 'DejaVu Sans'

class MemoryOptimizedBirliktelikAnalizi:
    def __init__(self, dosya_yolu: str, debug: bool = False):
        """
        Memory-optimized birliktelik analizi sınıfı
        
        Args:
            dosya_yolu (str): Excel dosyasının yolu
            debug (bool): Debug mode (memory tracking)
        """
        self.dosya_yolu = dosya_yolu
        self.debug = debug
        self.df = None
        self.transactions = []
        self.frequent_itemsets = None
        self.rules = None
        self.musteri_sutun = None
        self.urun_sutun = None
        
        if self.debug:
            print(f"🔧 Debug mode aktif - {self.get_memory_usage()}")
    
    def get_memory_usage(self) -> str:
        """Mevcut memory kullanımını döndür"""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return f"RAM: {memory_info.rss / 1024 / 1024:.2f} MB"
        except:
            return "RAM: N/A"
    
    def log_memory(self, step: str):
        """Memory kullanımını logla"""
        if self.debug:
            print(f"📊 {step}: {self.get_memory_usage()}")
    
    def veri_yukle_optimized(self, chunk_size: int = 10000) -> bool:
        """
        Excel dosyasını memory-efficient şekilde yükle
        
        Args:
            chunk_size (int): Chunk boyutu (büyük dosyalar için)
        """
        try:
            print("📂 Veri yükleniyor...")
            self.log_memory("Yükleme öncesi")
            
            # Dosya boyutunu kontrol et
            file_size = os.path.getsize(self.dosya_yolu) / (1024 * 1024)  # MB
            print(f"📏 Dosya boyutu: {file_size:.2f} MB")
            
            if file_size > 50:  # 50MB'dan büyükse chunk processing
                print("🔄 Büyük dosya - chunk processing aktif")
                chunks = []
                try:
                    for i, chunk in enumerate(pd.read_excel(self.dosya_yolu, chunksize=chunk_size)):
                        chunks.append(chunk)
                        if self.debug and i % 5 == 0:
                            print(f"   Chunk {i+1} yüklendi - {self.get_memory_usage()}")
                    
                    self.df = pd.concat(chunks, ignore_index=True)
                    del chunks
                    gc.collect()
                    
                except Exception as e:
                    print(f"⚠️ Chunk processing başarısız, normal yükleme deneniyor: {e}")
                    self.df = pd.read_excel(self.dosya_yolu)
            else:
                self.df = pd.read_excel(self.dosya_yolu)
            
            print("✅ Veri başarıyla yüklendi!")
            print(f"📊 Veri boyutu: {self.df.shape}")
            print(f"📋 Sütunlar: {list(self.df.columns)}")
            
            # Memory temizliği
            gc.collect()
            self.log_memory("Yükleme sonrası")
            
            if not self.debug:  # Debug mode değilse sadece kısa özet
                print(f"🔍 İlk 3 satır:\n{self.df.head(3)}")
            else:
                print(f"🔍 İlk 5 satır:\n{self.df.head()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Veri yükleme hatası: {e}")
            return False
    
    def veri_temizle_optimized(self, musteri_sutun: str = 'CariKod', urun_sutun: str = 'StockAd'):
        """
        Veriyi memory-efficient şekilde temizle
        
        Args:
            musteri_sutun (str): Müşteri ID sütunu
            urun_sutun (str): Ürün adı sütunu
        """
        print(f"\n🧹 Veri temizleniyor...")
        self.log_memory("Temizleme öncesi")
        
        print(f"👤 Müşteri sütunu: {musteri_sutun}")
        print(f"🛍️ Ürün sütunu: {urun_sutun}")
        
        # Sadece gerekli sütunları tut
        if musteri_sutun not in self.df.columns:
            print(f"❌ '{musteri_sutun}' sütunu bulunamadı!")
            print(f"📋 Mevcut sütunlar: {list(self.df.columns)}")
            return False
            
        if urun_sutun not in self.df.columns:
            print(f"❌ '{urun_sutun}' sütunu bulunamadı!")
            print(f"📋 Mevcut sütunlar: {list(self.df.columns)}")
            return False
        
        # Sadece gerekli sütunları seç
        original_shape = self.df.shape
        self.df = self.df[[musteri_sutun, urun_sutun]].copy()
        
        # Veri tiplerini optimize et
        self.df[musteri_sutun] = self.df[musteri_sutun].astype('category')
        self.df[urun_sutun] = self.df[urun_sutun].astype('category')
        
        # Boş değerleri temizle
        self.df = self.df.dropna(subset=[musteri_sutun, urun_sutun])
        
        # Boş string'leri temizle
        self.df = self.df[self.df[musteri_sutun] != '']
        self.df = self.df[self.df[urun_sutun] != '']
        
        print(f"✅ Veri boyutu: {original_shape} → {self.df.shape}")
        print(f"👥 Benzersiz müşteri: {self.df[musteri_sutun].nunique():,}")
        print(f"🛍️ Benzersiz ürün: {self.df[urun_sutun].nunique():,}")
        
        # Sütun isimlerini kaydet
        self.musteri_sutun = musteri_sutun
        self.urun_sutun = urun_sutun
        
        # Memory temizliği
        gc.collect()
        self.log_memory("Temizleme sonrası")
        
        return True
    
    def transaction_olustur_optimized(self, max_sepet_boyutu: int = 100) -> List[List[str]]:
        """
        Memory-efficient transaction oluşturma
        
        Args:
            max_sepet_boyutu (int): Maksimum sepet boyutu (aşırı büyük sepetleri filtrele)
        """
        print(f"\n🛒 Transaction'lar oluşturuluyor...")
        self.log_memory("Transaction öncesi")
        
        # Müşteri başına ürün listeleri oluştur (memory efficient)
        transactions_dict = defaultdict(set)
        
        # Chunk halinde işle
        chunk_size = 50000
        total_rows = len(self.df)
        
        for i in range(0, total_rows, chunk_size):
            chunk = self.df.iloc[i:i+chunk_size]
            
            for _, row in chunk.iterrows():
                musteri = row[self.musteri_sutun]
                urun = row[self.urun_sutun]
                transactions_dict[musteri].add(urun)
            
            # Progress
            if self.debug and i % (chunk_size * 5) == 0:
                print(f"   İşlenen satır: {i:,}/{total_rows:,} - {self.get_memory_usage()}")
            
            # Memory temizliği
            if i % (chunk_size * 10) == 0:
                gc.collect()
        
        # Dict'i listeye çevir
        self.transactions = []
        buyuk_sepet_sayisi = 0
        
        for musteri, urunler in transactions_dict.items():
            urun_listesi = list(urunler)
            
            # Çok büyük sepetleri filtrele (memory optimization)
            if len(urun_listesi) <= max_sepet_boyutu:
                self.transactions.append(urun_listesi)
            else:
                buyuk_sepet_sayisi += 1
        
        # Memory temizliği
        del transactions_dict
        gc.collect()
        
        # İstatistikler
        sepet_boyutlari = [len(t) for t in self.transactions]
        
        print(f"✅ {len(self.transactions):,} transaction oluşturuldu")
        print(f"📊 Ortalama sepet boyutu: {np.mean(sepet_boyutlari):.2f}")
        print(f"🔥 En büyük sepet: {max(sepet_boyutlari)} ürün")
        print(f"📈 Medyan sepet boyutu: {np.median(sepet_boyutlari):.1f}")
        
        if buyuk_sepet_sayisi > 0:
            print(f"⚠️ {buyuk_sepet_sayisi} adet çok büyük sepet filtrelendi (>{max_sepet_boyutu} ürün)")
        
        self.log_memory("Transaction sonrası")
        
        return self.transactions
    
    def apriori_analiz_optimized(self, min_support: float = 0.05, min_confidence: float = 0.5, 
                               min_lift: float = 1.0, max_len: int = 3) -> Optional[pd.DataFrame]:
        """
        Memory-optimized Apriori analizi
        
        Args:
            min_support (float): Minimum destek değeri
            min_confidence (float): Minimum güven değeri  
            min_lift (float): Minimum lift değeri
            max_len (int): Maksimum itemset uzunluğu
        """
        print(f"\n🔍 Memory-optimized Apriori analizi başlatılıyor...")
        print(f"📏 Min Support: {min_support:.2%}")
        print(f"📏 Min Confidence: {min_confidence:.2%}")
        print(f"📏 Min Lift: {min_lift}")
        print(f"📏 Max Itemset Length: {max_len}")
        
        self.log_memory("Apriori öncesi")
        
        if not self.transactions:
            print("❌ Transaction'lar bulunamadı! Önce transaction_olustur_optimized() çalıştırın.")
            return None
        
        try:
            # Transaction'ları binary matrix'e çevir (memory efficient)
            print("🔄 Binary matrix oluşturuluyor...")
            
            te = TransactionEncoder()
            te_ary = te.fit(self.transactions).transform(self.transactions)
            
            # Sadece gerekli sütunları tut (minimum support check)
            df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
            
            # Memory temizliği
            del te_ary
            gc.collect()
            
            print(f"✅ Binary matrix: {df_encoded.shape}")
            self.log_memory("Binary matrix sonrası")
            
            # Apriori algoritması (memory optimized)
            print("⚙️ Apriori algoritması çalışıyor...")
            
            self.frequent_itemsets = apriori(df_encoded, 
                                           min_support=min_support, 
                                           use_colnames=True,
                                           low_memory=True,
                                           max_len=max_len)
            
            # Memory temizliği
            del df_encoded
            gc.collect()
            
            if len(self.frequent_itemsets) == 0:
                print("❌ Belirlenen kriterlere uygun sık ürün seti bulunamadı!")
                print("💡 min_support değerini düşürmeyi deneyin.")
                return None
            
            print(f"✅ {len(self.frequent_itemsets)} sık ürün seti bulundu")
            self.log_memory("Frequent itemsets sonrası")
            
            # Birliktelik kurallarını oluştur
            print("🔗 Birliktelik kuralları oluşturuluyor...")
            
            self.rules = association_rules(self.frequent_itemsets, 
                                         metric="confidence", 
                                         min_threshold=min_confidence)
            
            # Lift filtresini uygula
            initial_rule_count = len(self.rules)
            self.rules = self.rules[self.rules['lift'] >= min_lift]
            
            print(f"✅ {len(self.rules)} birliktelik kuralı oluşturuldu")
            print(f"📊 Lift filtresi: {initial_rule_count} → {len(self.rules)}")
            
            # Memory temizliği
            gc.collect()
            self.log_memory("Kurallar sonrası")
            
            return self.rules
            
        except Exception as e:
            print(f"❌ Apriori analizi hatası: {e}")
            print("💡 min_support değerini artırmayı veya max_len'i azaltmayı deneyin.")
            return None
    
    def sonuclari_goster_optimized(self, top_n: int = 10):
        """Optimized sonuç gösterimi"""
        if self.rules is None or len(self.rules) == 0:
            print("❌ Gösterilecek kural bulunamadı!")
            return
        
        print(f"\n📊 EN İYİ {top_n} BİRLİKTELİK KURALI")
        print("=" * 80)
        
        # Kuralları lift'e göre sırala
        top_rules = self.rules.nlargest(top_n, 'lift')
        
        for i, (idx, rule) in enumerate(top_rules.iterrows(), 1):
            antecedents = ', '.join(list(rule['antecedents']))
            consequents = ', '.join(list(rule['consequents']))
            
            print(f"\n🔸 Kural {i}:")
            print(f"   📝 {antecedents} → {consequents}")
            print(f"   📊 Destek: {rule['support']:.2%}")
            print(f"   🎯 Güven: {rule['confidence']:.2%}")
            print(f"   📈 Lift: {rule['lift']:.2f}")
            
            # Sadece debug mode'da detaylı yorum
            if self.debug:
                print(f"   💡 Yorum: '{antecedents}' alan müşterilerin {rule['confidence']:.0%}'i '{consequents}' de alıyor")
    
    def sik_urunler_goster_optimized(self, top_n: int = 15):
        """Memory-efficient sık ürün gösterimi"""
        if self.frequent_itemsets is None:
            print("❌ Önce analiz yapın!")
            return
        
        # Tek ürün setlerini filtrele
        single_items = self.frequent_itemsets[
            self.frequent_itemsets['itemsets'].apply(len) == 1
        ].copy()
        
        single_items['urun'] = single_items['itemsets'].apply(lambda x: list(x)[0])
        single_items = single_items.sort_values('support', ascending=False)
        
        print(f"\n🔥 EN SIK SATILAN {top_n} ÜRÜN")
        print("=" * 70)
        
        for i, (idx, row) in enumerate(single_items.head(top_n).iterrows(), 1):
            urun_adi = row['urun'][:50]  # Uzun isimleri kısalt
            if len(row['urun']) > 50:
                urun_adi += "..."
            print(f"{i:2d}. {urun_adi:<53} ({row['support']:.2%})")
    
    def mini_grafik_olustur(self):
        """Memory-efficient mini görselleştirme"""
        if self.rules is None or len(self.rules) == 0:
            print("❌ Görselleştirme için kural bulunamadı!")
            return
        
        try:
            print("\n📊 Grafik oluşturuluyor...")
            self.log_memory("Grafik öncesi")
            
            # Sadece en önemli grafikler (memory efficient)
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # 1. Support vs Confidence (sample alarak)
            sample_size = min(1000, len(self.rules))
            rules_sample = self.rules.sample(sample_size) if len(self.rules) > sample_size else self.rules
            
            scatter = axes[0, 0].scatter(rules_sample['support'], rules_sample['confidence'], 
                                       c=rules_sample['lift'], cmap='viridis', alpha=0.6)
            axes[0, 0].set_xlabel('Support')
            axes[0, 0].set_ylabel('Confidence')
            axes[0, 0].set_title('Support vs Confidence')
            plt.colorbar(scatter, ax=axes[0, 0])
            
            # 2. Lift dağılımı
            axes[0, 1].hist(self.rules['lift'], bins=20, alpha=0.7, color='skyblue')
            axes[0, 1].set_xlabel('Lift')
            axes[0, 1].set_ylabel('Frekans')
            axes[0, 1].set_title('Lift Dağılımı')
            
            # 3. Top 10 rules (kısa isimler)
            top_rules = self.rules.nlargest(10, 'lift')
            rule_labels = []
            for _, rule in top_rules.iterrows():
                ant = list(rule['antecedents'])[0][:12]
                con = list(rule['consequents'])[0][:12]
                rule_labels.append(f"{ant}→{con}")
            
            y_pos = np.arange(len(rule_labels))
            axes[1, 0].barh(y_pos, top_rules['lift'])
            axes[1, 0].set_yticks(y_pos)
            axes[1, 0].set_yticklabels(rule_labels, fontsize=8)
            axes[1, 0].set_xlabel('Lift')
            axes[1, 0].set_title('Top 10 Rules')
            
            # 4. Confidence vs Lift (sample)
            axes[1, 1].scatter(rules_sample['confidence'], rules_sample['lift'], 
                             alpha=0.6, color='coral')
            axes[1, 1].set_xlabel('Confidence')
            axes[1, 1].set_ylabel('Lift')
            axes[1, 1].set_title('Confidence vs Lift')
            
            plt.tight_layout()
            plt.show()
            
            # Memory temizliği
            plt.close(fig)
            gc.collect()
            self.log_memory("Grafik sonrası")
            
        except Exception as e:
            print(f"❌ Grafik oluşturma hatası: {e}")
    
    def rapor_kaydet_optimized(self, dosya_adi: str = "birliktelik_raporu_optimized.xlsx"):
        """Memory-efficient rapor kaydetme"""
        if self.rules is None:
            print("❌ Kaydedilecek kural bulunamadı!")
            return
        
        try:
            print(f"\n💾 Rapor kaydediliyor: {dosya_adi}")
            self.log_memory("Kaydetme öncesi")
            
            with pd.ExcelWriter(dosya_adi, engine='openpyxl') as writer:
                # Kuralları hazırla
                rules_df = self.rules.copy()
                rules_df['antecedents'] = rules_df['antecedents'].apply(lambda x: ', '.join(list(x)))
                rules_df['consequents'] = rules_df['consequents'].apply(lambda x: ', '.join(list(x)))
                
                # Gereksiz sütunları kaldır
                columns_to_save = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
                rules_df = rules_df[columns_to_save]
                
                rules_df.to_excel(writer, sheet_name='Birliktelik_Kurallari', index=False)
                
                # Sık ürünler
                if self.frequent_itemsets is not None:
                    single_items = self.frequent_itemsets[
                        self.frequent_itemsets['itemsets'].apply(len) == 1
                    ].copy()
                    
                    single_items['urun'] = single_items['itemsets'].apply(lambda x: list(x)[0])
                    single_items = single_items[['urun', 'support']].sort_values('support', ascending=False)
                    single_items.to_excel(writer, sheet_name='Sik_Urunler', index=False)
                
                # Özet istatistikler
                ozet_data = {
                    'Metrik': ['Toplam Kural Sayısı', 'Ortalama Confidence', 'Ortalama Lift', 
                              'En Yüksek Lift', 'Transaction Sayısı'],
                    'Değer': [len(self.rules), 
                             self.rules['confidence'].mean(),
                             self.rules['lift'].mean(),
                             self.rules['lift'].max(),
                             len(self.transactions)]
                }
                
                ozet_df = pd.DataFrame(ozet_data)
                ozet_df.to_excel(writer, sheet_name='Ozet_Istatistikler', index=False)
            
            print(f"✅ Rapor '{dosya_adi}' dosyasına kaydedildi!")
            self.log_memory("Kaydetme sonrası")
            
        except Exception as e:
            print(f"❌ Rapor kaydetme hatası: {e}")

def main_optimized():
    """Memory-optimized ana fonksiyon"""
    print("🚀 MEMORY-OPTIMIZED BİRLİKTELİK ANALİZİ")
    print("=" * 60)
    
    # Analiz nesnesini oluştur
    analiz = MemoryOptimizedBirliktelikAnalizi(
        dosya_yolu="/Users/esranuryigit/Desktop/alpata satış tahminleme projesi/birliktelikAnalizi/birliktelikAnaliz.xlsx",
        debug=True  # Memory tracking için
    )
    
    # Veriyi yükle
    if not analiz.veri_yukle_optimized():
        print("❌ Veri yükleme başarısız!")
        return
    
    # Veriyi temizle
    if not analiz.veri_temizle_optimized(musteri_sutun='CariKod', urun_sutun='StockAd'):
        print("❌ Veri temizleme başarısız!")
        return
    
    # Transaction'ları oluştur
    transactions = analiz.transaction_olustur_optimized(max_sepet_boyutu=50)  # Memory limit
    
    if not transactions:
        print("❌ Transaction oluşturulamadı!")
        return
    
    # Apriori analizi (conservative parameters)
    rules = analiz.apriori_analiz_optimized(
        min_support=0.05,      # %5 destek
        min_confidence=0.3,    # %30 güven  
        min_lift=1.0,          # 1.0 lift
        max_len=2              # Maksimum 2 itemset (memory optimization)
    )
    
    if rules is not None and len(rules) > 0:
        print(f"\n🎉 ANALİZ BAŞARILI!")
        print(f"✅ {len(rules)} kural bulundu")
        
        # Sonuçları göster
        analiz.sonuclari_goster_optimized(top_n=10)
        analiz.sik_urunler_goster_optimized(top_n=15)
        
        # Mini grafik (memory efficient)
        analiz.mini_grafik_olustur()
        
        # Raporu kaydet
        analiz.rapor_kaydet_optimized("birliktelik_analizi_optimized.xlsx")
        
        print(f"\n📊 FINAL MEMORY: {analiz.get_memory_usage()}")
        
    else:
        print("❌ Hiç kural bulunamadı!")
        print("💡 Parametreleri düşürmeyi deneyin:")
        print("   - min_support=0.01 (daha düşük)")
        print("   - min_confidence=0.2 (daha düşük)")
        print("   - Veri kalitesini kontrol edin")
    
    print("\n🎯 Analiz tamamlandı!")

if __name__ == "__main__":
    main_optimized()