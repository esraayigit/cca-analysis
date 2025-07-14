import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
import gc
import psutil
import os
warnings.filterwarnings('ignore')

class UrunOneriSistemi:
    def __init__(self, dosya_yolu, debug=False):
        """
        Ürün öneri sistemi - Birliktelik analizi tabanlı
        
        Args:
            dosya_yolu (str): Excel dosyasının yolu
            debug (bool): Debug mode
        """
        self.dosya_yolu = dosya_yolu
        self.debug = debug
        self.df = None
        self.transactions = []
        self.rules = None
        self.urun_gruplari = {}
        self.oneriler = []
        
    def get_memory_usage(self):
        """Memory kullanımını döndür"""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return f"RAM: {memory_info.rss / 1024 / 1024:.2f} MB"
        except:
            return "RAM: N/A"
    
    def log_memory(self, step):
        """Memory kullanımını logla"""
        if self.debug:
            print(f"📊 {step}: {self.get_memory_usage()}")
    
    def veri_yukle(self):
        """Veriyi yükle"""
        try:
            print("📂 Veri yükleniyor...")
            self.log_memory("Yükleme öncesi")
            
            # Dosya boyutunu kontrol et
            file_size = os.path.getsize(self.dosya_yolu) / (1024 * 1024)
            print(f"📏 Dosya boyutu: {file_size:.2f} MB")
            
            if file_size > 50:
                print("🔄 Büyük dosya - chunk processing")
                chunks = []
                for chunk in pd.read_excel(self.dosya_yolu, chunksize=10000):
                    chunks.append(chunk)
                self.df = pd.concat(chunks, ignore_index=True)
                del chunks
                gc.collect()
            else:
                self.df = pd.read_excel(self.dosya_yolu)
            
            print("✅ Veri yüklendi!")
            print(f"📊 Veri boyutu: {self.df.shape}")
            print(f"📋 Sütunlar: {list(self.df.columns)}")
            
            # İlk birkaç satır göster
            print("\n🔍 İlk 3 satır:")
            print(self.df.head(3))
            
            self.log_memory("Yükleme sonrası")
            return True
            
        except Exception as e:
            print(f"❌ Veri yükleme hatası: {e}")
            return False
    
    def veri_temizle(self, musteri_sutun='CariKod', urun_sutun='StockAd', 
                     urun_grubu_sutun='Ürün Grubu', yil_sutun='yil', ay_sutun='aylik'):
        """
        Veriyi temizle ve gerekli sütunları belirle
        
        Args:
            musteri_sutun (str): Müşteri/Bayi sütunu
            urun_sutun (str): Ürün adı sütunu
            urun_grubu_sutun (str): Ürün grubu sütunu
            yil_sutun (str): Yıl sütunu
            ay_sutun (str): Ay sütunu
        """
        print(f"\n🧹 Veri temizleniyor...")
        self.log_memory("Temizleme öncesi")
        
        # Gerekli sütunları kontrol et
        required_columns = [musteri_sutun, urun_sutun, urun_grubu_sutun, yil_sutun, ay_sutun]
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            print(f"❌ Eksik sütunlar: {missing_columns}")
            print(f"📋 Mevcut sütunlar: {list(self.df.columns)}")
            return False
        
        # Sadece gerekli sütunları al
        self.df = self.df[required_columns].copy()
        
        # Sütun isimlerini standartlaştır
        self.df.columns = ['musteri', 'urun', 'urun_grubu', 'yil', 'ay']
        
        # Veri tiplerini optimize et
        self.df['musteri'] = self.df['musteri'].astype('category')
        self.df['urun'] = self.df['urun'].astype('category') 
        self.df['urun_grubu'] = self.df['urun_grubu'].astype('category')
        self.df['yil'] = self.df['yil'].astype('int16')
        self.df['ay'] = self.df['ay'].astype('int8')
        
        # Boş değerleri temizle
        original_shape = self.df.shape
        self.df = self.df.dropna()
        
        print(f"✅ Veri boyutu: {original_shape} → {self.df.shape}")
        print(f"👥 Benzersiz müşteri: {self.df['musteri'].nunique():,}")
        print(f"🛍️ Benzersiz ürün: {self.df['urun'].nunique():,}")
        print(f"📦 Benzersiz ürün grubu: {self.df['urun_grubu'].nunique()}")
        
        # Ürün gruplarını kaydet
        self.urun_gruplari = self.df.groupby('urun')['urun_grubu'].first().to_dict()
        
        print(f"\n📦 Ürün grupları:")
        for grup, count in self.df['urun_grubu'].value_counts().head(10).items():
            print(f"   {grup}: {count} ürün")
        
        self.log_memory("Temizleme sonrası")
        return True
    
    def transaction_olustur(self, max_sepet_boyutu=100):
        """Transaction'ları oluştur"""
        print(f"\n🛒 Transaction'lar oluşturuluyor...")
        self.log_memory("Transaction öncesi")
        
        # Müşteri başına ürün listeleri
        transactions_dict = defaultdict(set)
        
        # Chunk halinde işle
        chunk_size = 50000
        total_rows = len(self.df)
        
        for i in range(0, total_rows, chunk_size):
            chunk = self.df.iloc[i:i+chunk_size]
            
            for _, row in chunk.iterrows():
                transactions_dict[row['musteri']].add(row['urun'])
            
            if self.debug and i % (chunk_size * 5) == 0:
                print(f"   İşlenen: {i:,}/{total_rows:,} - {self.get_memory_usage()}")
        
        # Dict'i listeye çevir
        self.transactions = []
        filtrelenen_sepet = 0
        
        for musteri, urunler in transactions_dict.items():
            urun_listesi = list(urunler)
            if len(urun_listesi) <= max_sepet_boyutu:
                self.transactions.append(urun_listesi)
            else:
                filtrelenen_sepet += 1
        
        del transactions_dict
        gc.collect()
        
        # İstatistikler
        sepet_boyutlari = [len(t) for t in self.transactions]
        print(f"✅ {len(self.transactions):,} transaction oluşturuldu")
        print(f"📊 Ortalama sepet boyutu: {np.mean(sepet_boyutlari):.2f}")
        print(f"🔥 En büyük sepet: {max(sepet_boyutlari)}")
        
        if filtrelenen_sepet > 0:
            print(f"⚠️ {filtrelenen_sepet} büyük sepet filtrelendi")
        
        self.log_memory("Transaction sonrası")
        return True
    
    def birliktelik_analizi(self, min_support=0.05, min_confidence=0.5, min_lift=1.0):
        """
        Birliktelik analizi yap ve aynı ürün grubu kurallarını filtrele
        
        Args:
            min_support (float): Minimum destek
            min_confidence (float): Minimum güven
            min_lift (float): Minimum lift
        """
        print(f"\n🔍 Birliktelik analizi başlatılıyor...")
        print(f"📏 Min Support: {min_support:.2%}")
        print(f"📏 Min Confidence: {min_confidence:.2%}")
        print(f"📏 Min Lift: {min_lift}")
        
        self.log_memory("Analiz öncesi")
        
        try:
            # Binary matrix oluştur
            print("🔄 Binary matrix oluşturuluyor...")
            te = TransactionEncoder()
            te_ary = te.fit(self.transactions).transform(self.transactions)
            df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
            
            del te_ary
            gc.collect()
            
            print(f"✅ Binary matrix: {df_encoded.shape}")
            self.log_memory("Binary matrix sonrası")
            
            # Apriori algoritması
            print("⚙️ Apriori algoritması çalışıyor...")
            frequent_itemsets = apriori(df_encoded, min_support=min_support, 
                                      use_colnames=True, low_memory=True, max_len=2)
            
            del df_encoded
            gc.collect()
            
            if len(frequent_itemsets) == 0:
                print("❌ Sık ürün seti bulunamadı!")
                return False
            
            print(f"✅ {len(frequent_itemsets)} sık ürün seti bulundu")
            self.log_memory("Frequent itemsets sonrası")
            
            # Association rules
            print("🔗 Birliktelik kuralları oluşturuluyor...")
            rules = association_rules(frequent_itemsets, metric="confidence", 
                                    min_threshold=min_confidence)
            
            del frequent_itemsets
            gc.collect()
            
            if len(rules) == 0:
                print("❌ Kural bulunamadı!")
                return False
            
            print(f"🔗 {len(rules)} kural oluşturuldu")
            
            # Lift filtresi
            rules = rules[rules['lift'] >= min_lift]
            print(f"📈 Lift filtresi sonrası: {len(rules)} kural")
            
            # AYNI ÜRÜN GRUBU KURALLARINI FİLTRELE
            print("🚫 Aynı ürün grubu kuralları filtreleniyor...")
            
            filtered_rules = []
            for _, rule in rules.iterrows():
                antecedents = list(rule['antecedents'])
                consequents = list(rule['consequents'])
                
                # Antecedent ürünlerin grupları
                ant_groups = set()
                for urun in antecedents:
                    if urun in self.urun_gruplari:
                        ant_groups.add(self.urun_gruplari[urun])
                
                # Consequent ürünlerin grupları
                con_groups = set()
                for urun in consequents:
                    if urun in self.urun_gruplari:
                        con_groups.add(self.urun_gruplari[urun])
                
                # Farklı ürün gruplarından ise kuralı al
                if not ant_groups.intersection(con_groups):
                    filtered_rules.append({
                        'antecedents': antecedents,
                        'consequents': consequents,
                        'antecedent_support': rule['antecedent support'],
                        'consequent_support': rule['consequent support'], 
                        'support': rule['support'],
                        'confidence': rule['confidence'],
                        'lift': rule['lift'],
                        'antecedent_groups': list(ant_groups),
                        'consequent_groups': list(con_groups)
                    })
            
            self.rules = pd.DataFrame(filtered_rules)
            
            print(f"✅ Farklı ürün grubu kuralları: {len(self.rules)}")
            self.log_memory("Filtreleme sonrası")
            
            if len(self.rules) == 0:
                print("❌ Farklı ürün grubu kuralı bulunamadı!")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Analiz hatası: {e}")
            return False
    
    def oneri_olustur(self):
        """
        Son ay antecedent ürünü satıp consequent ürünü satmayan müşterilere öneri oluştur
        """
        print(f"\n🎯 Öneri sistemi çalışıyor...")
        self.log_memory("Öneri öncesi")
        
        if self.rules is None or len(self.rules) == 0:
            print("❌ Önce birliktelik analizi yapın!")
            return False
        
        # Son ay verilerini al
        son_yil = self.df['yil'].max()
        son_ay = self.df[self.df['yil'] == son_yil]['ay'].max()
        
        print(f"📅 Son ay: {son_yil}/{son_ay}")
        
        son_ay_verisi = self.df[(self.df['yil'] == son_yil) & (self.df['ay'] == son_ay)].copy()
        
        # Müşteri-ürün matrisini oluştur
        musteri_urunler = son_ay_verisi.groupby('musteri')['urun'].apply(set).to_dict()
        
        print(f"👥 Son ay {len(musteri_urunler)} müşteri alışveriş yaptı")
        
        # Her kural için öneri kontrolü
        oneriler = []
        
        for _, rule in self.rules.iterrows():
            antecedents = set(rule['antecedents'])
            consequents = set(rule['consequents'])
            
            # Her müşteri için kontrol
            for musteri, urunler in musteri_urunler.items():
                # Antecedent ürünleri satın almış mı?
                antecedent_alindi = antecedents.issubset(urunler)
                
                # Consequent ürünleri satın almış mı?
                consequent_alindi = bool(consequents.intersection(urunler))
                
                # Antecedent var, consequent yok -> ÖNERİ FIRSATI!
                if antecedent_alindi and not consequent_alindi:
                    for onerilen_urun in consequents:
                        if onerilen_urun not in urunler:  # Zaten almamış
                            oneriler.append({
                                'musteri': musteri,
                                'antecedent_urunler': ', '.join(antecedents),
                                'onerilen_urun': onerilen_urun,
                                'onerilen_urun_grubu': self.urun_gruplari.get(onerilen_urun, 'Bilinmeyen'),
                                'confidence': rule['confidence'],
                                'lift': rule['lift'],
                                'support': rule['support'],
                                'antecedent_groups': ', '.join(rule['antecedent_groups']),
                                'consequent_groups': ', '.join(rule['consequent_groups'])
                            })
        
        self.oneriler = pd.DataFrame(oneriler)
        
        print(f"🎉 {len(self.oneriler)} öneri oluşturuldu!")
        
        if len(self.oneriler) > 0:
            print(f"👥 {self.oneriler['musteri'].nunique()} müşteriye öneri var")
            
            # En sık önerilen ürünler
            print(f"\n🔥 En sık önerilen ürünler:")
            top_oneriler = self.oneriler['onerilen_urun'].value_counts().head(10)
            for urun, count in top_oneriler.items():
                print(f"   {urun[:50]}: {count} öneri")
        
        self.log_memory("Öneri sonrası")
        return True
    
    def sonuclari_goster(self, top_n=10):
        """Sonuçları göster"""
        if self.oneriler is None or len(self.oneriler) == 0:
            print("❌ Öneri bulunamadı!")
            return
        
        print(f"\n📊 ÖNERİ SİSTEMİ SONUÇLARI")
        print("=" * 60)
        
        # Genel istatistikler
        print(f"📈 Toplam öneri: {len(self.oneriler):,}")
        print(f"👥 Öneri alan müşteri: {self.oneriler['musteri'].nunique():,}")
        print(f"🛍️ Önerilen ürün çeşidi: {self.oneriler['onerilen_urun'].nunique():,}")
        
        # En yüksek confidence'lı öneriler
        print(f"\n🏆 EN YÜKSEK CONFIDENCE'LI {top_n} ÖNERİ:")
        print("-" * 80)
        
        top_oneriler = self.oneriler.nlargest(top_n, 'confidence')
        
        for i, (_, oneri) in enumerate(top_oneriler.iterrows(), 1):
            print(f"\n{i}. Müşteri: {oneri['musteri']}")
            print(f"   📝 Aldığı ürünler: {oneri['antecedent_urunler']}")
            print(f"   🎯 Önerilen ürün: {oneri['onerilen_urun']}")
            print(f"   📦 Ürün grubu: {oneri['onerilen_urun_grubu']}")
            print(f"   📊 Confidence: {oneri['confidence']:.1%}")
            print(f"   📈 Lift: {oneri['lift']:.2f}")
        
        # Ürün grubu bazında öneriler
        print(f"\n📦 ÜRÜN GRUBU BAZINDA ÖNERİLER:")
        print("-" * 50)
        
        grup_ozetle = self.oneriler.groupby('onerilen_urun_grubu').agg({
            'musteri': 'count',
            'confidence': 'mean',
            'lift': 'mean'
        }).round(2).sort_values('musteri', ascending=False)
        
        for grup, stats in grup_ozetle.head(10).iterrows():
            print(f"   {grup}: {stats['musteri']} öneri (Avg Conf: {stats['confidence']:.1%})")
    
    def excel_kaydet(self, dosya_adi="urun_onerileri.xlsx"):
        """Sonuçları Excel'e kaydet"""
        if self.oneriler is None or len(self.oneriler) == 0:
            print("❌ Kaydedilecek öneri yok!")
            return
        
        try:
            print(f"\n💾 Excel dosyası kaydediliyor: {dosya_adi}")
            
            with pd.ExcelWriter(dosya_adi, engine='openpyxl') as writer:
                # Öneriler
                self.oneriler.to_excel(writer, sheet_name='Ürün_Onerileri', index=False)
                
                # Kurallar
                if self.rules is not None and len(self.rules) > 0:
                    rules_export = self.rules.copy()
                    rules_export['antecedents'] = rules_export['antecedents'].apply(lambda x: ', '.join(x))
                    rules_export['consequents'] = rules_export['consequents'].apply(lambda x: ', '.join(x))
                    rules_export['antecedent_groups'] = rules_export['antecedent_groups'].apply(lambda x: ', '.join(x))
                    rules_export['consequent_groups'] = rules_export['consequent_groups'].apply(lambda x: ', '.join(x))
                    
                    rules_export.to_excel(writer, sheet_name='Birliktelik_Kurallari', index=False)
                
                # Özet istatistikler
                ozet_data = {
                    'Metrik': [
                        'Toplam Öneri Sayısı',
                        'Öneri Alan Müşteri Sayısı', 
                        'Önerilen Ürün Çeşidi',
                        'Ortalama Confidence',
                        'Ortalama Lift',
                        'En Yüksek Confidence',
                        'En Yüksek Lift'
                    ],
                    'Değer': [
                        len(self.oneriler),
                        self.oneriler['musteri'].nunique(),
                        self.oneriler['onerilen_urun'].nunique(),
                        self.oneriler['confidence'].mean(),
                        self.oneriler['lift'].mean(),
                        self.oneriler['confidence'].max(),
                        self.oneriler['lift'].max()
                    ]
                }
                
                pd.DataFrame(ozet_data).to_excel(writer, sheet_name='Ozet_Istatistikler', index=False)
            
            print(f"✅ Excel dosyası kaydedildi: {dosya_adi}")
            
        except Exception as e:
            print(f"❌ Excel kaydetme hatası: {e}")

def main():
    """Ana fonksiyon"""
    print("🚀 ÜRÜN ÖNERİ SİSTEMİ - BİRLİKTELİK ANALİZİ")
    print("=" * 60)
    
    # Sistem oluştur
    sistem = UrunOneriSistemi(
        dosya_yolu="/Users/esranuryigit/Desktop/BirliktelikAnalizi_son.xlsx",
        debug=True
    )
    
    # Veri yükleme
    if not sistem.veri_yukle():
        print("❌ Veri yükleme başarısız!")
        return
    
    # Veri temizleme (sütun isimlerini verinize göre ayarlayın)
    if not sistem.veri_temizle(
        musteri_sutun='cairkod',          # Bayi kodu
        urun_sutun='StockAd',            # Ürün adı
        urun_grubu_sutun='Ürün Grubu',   # Ürün grubu
        yil_sutun='yil',                 # Yıl
        ay_sutun='aylik'                 # Ay
    ):
        print("❌ Veri temizleme başarısız!")
        return
    
    # Transaction oluşturma
    if not sistem.transaction_olustur(max_sepet_boyutu=50):
        print("❌ Transaction oluşturulamadı!")
        return
    
    # Birliktelik analizi
    if not sistem.birliktelik_analizi(
        min_support=0.03,      # %3 destek
        min_confidence=0.5,    # %50 güven
        min_lift=1.0          # 1.0 lift
    ):
        print("❌ Birliktelik analizi başarısız!")
        return
    
    # Öneri oluşturma
    if not sistem.oneri_olustur():
        print("❌ Öneri oluşturulamadı!")
        return
    
    # Sonuçları göster
    sistem.sonuclari_goster(top_n=15)
    
    # Excel'e kaydet
    sistem.excel_kaydet("urun_onerileri_birliktelik.xlsx")
    
    print(f"\n🎯 Sistem tamamlandı! {sistem.get_memory_usage()}")

if __name__ == "__main__":
    main()