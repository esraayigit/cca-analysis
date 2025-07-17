import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
import gc
import psutil
import os
import time
from datetime import datetime
from functools import wraps
warnings.filterwarnings('ignore')

def timing_decorator(func):
    """
    Fonksiyon çalışma süresini ölçen decorator
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"⏱️ {func.__name__} çalışma süresi: {end_time - start_time:.2f} saniye")
        return result
    return wrapper

def get_memory_usage():
    """
    Mevcut memory kullanımını gösterir
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return f"RAM: {memory_info.rss / 1024 / 1024:.2f} MB"

def log_performance(message, start_time=None):
    """
    Performance logging fonksiyonu
    """
    current_time = time.time()
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    if start_time:
        elapsed = current_time - start_time
        print(f"[{timestamp}] {message} - Geçen süre: {elapsed:.2f}s - {get_memory_usage()}")
    else:
        print(f"[{timestamp}] {message} - {get_memory_usage()}")
    
    return current_time

@timing_decorator
def load_and_prepare_data_optimized(file_path, debug=False):
    """
    Excel dosyasını memory-efficient şekilde yükler ve VERİ TİPLERİNİ DÜZELTIR
    """
    start_time = log_performance("📂 Veri yükleme başladı")
    
    try:
        # Dosya boyutu kontrolü
        file_size = os.path.getsize(file_path) / (1024 * 1024)
        log_performance(f"📊 Dosya boyutu: {file_size:.2f} MB", start_time)
        
        # Veri yükleme
        load_start = time.time()
        if file_size > 50:
            log_performance("⚠️ Büyük dosya - chunk processing başladı")
            chunks = []
            for i, chunk in enumerate(pd.read_excel(file_path, chunksize=10000)):
                chunks.append(chunk)
                if i % 5 == 0:  # Her 5 chunk'da log
                    log_performance(f"📦 {i+1}. chunk yüklendi")
            df = pd.concat(chunks, ignore_index=True)
            del chunks
            gc.collect()
        else:
            df = pd.read_excel(file_path)
        
        log_performance(f"✅ Veri yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun", load_start)
        
        # ORIJINAL VERI ANALIZI
        if debug:
            print(f"\n🔍 ORIJINAL VERİ ANALİZİ:")
            print(f"Sütunlar: {list(df.columns)}")
            if len(df.columns) > 9:  # Ürün Grubu sütunu varsa
                orig_col = df.columns[9]  # Son sütun genellikle Ürün Grubu
                print(f"Son sütun (Ürün Grubu?): '{orig_col}'")
                print(f"Son sütun tipi: {df[orig_col].dtype}")
                print(f"Son sütun benzersiz değer sayısı: {df[orig_col].nunique()}")
                print(f"Son sütun örnek değerleri: {df[orig_col].dropna().unique()[:10].tolist()}")
        
        # Sütun standardizasyonu
        prep_start = time.time()
        expected_columns = ['carikod', 'yil', 'aylik', 'cirokod_aylik', 'ciro_degeri', 'StockAd','segment', 'StockId','StockKod','Ürün Grubu']
        
        if len(df.columns) >= len(expected_columns):
            df.columns = expected_columns[:len(df.columns)]
        
        columns_to_keep = ['carikod', 'yil', 'aylik', 'StockAd','segment', 'StockId','StockKod','Ürün Grubu']
        df = df[columns_to_keep].copy()
        
        # ÜRÜN GRUBU ANALİZİ - CLEANING ÖNCESI
        if debug:
            print(f"\n🔍 ÜRÜN GRUBU ANALİZİ (CLEANING ÖNCESI):")
            print(f"'Ürün Grubu' sütunu tipi: {df['Ürün Grubu'].dtype}")
            print(f"'Ürün Grubu' benzersiz değer sayısı: {df['Ürün Grubu'].nunique()}")
            print(f"'Ürün Grubu' NaN sayısı: {df['Ürün Grubu'].isnull().sum()}")
            
            # Sayısal mı kontrol et
            numeric_groups = pd.to_numeric(df['Ürün Grubu'], errors='coerce').notna().sum()
            print(f"Sayısal görünen değer sayısı: {numeric_groups}")
            
            # Unique değerlerin örnekleri
            unique_groups = df['Ürün Grubu'].dropna().unique()
            print(f"İlk 20 benzersiz ürün grubu:")
            for i, group in enumerate(unique_groups[:20], 1):
                print(f"  {i:2d}. '{group}' (tip: {type(group).__name__})")
            
            # Value counts
            print(f"\nEn çok kullanılan ürün grupları:")
            top_groups = df['Ürün Grubu'].value_counts().head(10)
            for group, count in top_groups.items():
                print(f"  '{group}': {count:,} adet")
        
        # KRİTİK: ÜRÜN ADLARINI STRING'E ÇEVİR VE TEMİZLE
        log_performance("🔧 Ürün adları string'e çevriliyor...")
        
        # 1. NaN değerleri temizle
        df = df.dropna(subset=['StockAd', 'segment', 'Ürün Grubu'])
        
        # 2. Tüm kritik sütunları string'e çevir
        df['StockAd'] = df['StockAd'].astype(str).str.strip()
        df['segment'] = df['segment'].astype(str).str.strip()
        df['Ürün Grubu'] = df['Ürün Grubu'].astype(str).str.strip()
        
        # 3. 'nan' string'lerini temizle
        df = df[
            (df['StockAd'] != 'nan') & 
            (df['StockAd'] != 'None') & 
            (df['StockAd'] != '') &
            (df['segment'] != 'nan') & 
            (df['segment'] != 'None') & 
            (df['segment'] != '') &
            (df['Ürün Grubu'] != 'nan') & 
            (df['Ürün Grubu'] != 'None') & 
            (df['Ürün Grubu'] != '')
        ]
        
        # 4. Boş veya sadece whitespace olan değerleri temizle
        df = df[
            (df['StockAd'].str.len() > 0) &
            (df['segment'].str.len() > 0) &
            (df['Ürün Grubu'].str.len() > 0)
        ]
        
        # 5. Veri tipi optimizasyonu - ÇOK ÖNEM
        df['carikod'] = df['carikod'].astype(str)  # Bu da string olsun
        df['yil'] = pd.to_numeric(df['yil'], errors='coerce').astype('int16')
        df['aylik'] = pd.to_numeric(df['aylik'], errors='coerce').astype('int8')
        
        # 6. Son kontrol - numeric conversion hatalarını temizle
        df = df.dropna(subset=['yil', 'aylik'])
        
        # ÜRÜN GRUBU ANALİZİ - CLEANING SONRASI
        if debug:
            print(f"\n🔍 ÜRÜN GRUBU ANALİZİ (CLEANING SONRASI):")
            print(f"'Ürün Grubu' benzersiz değer sayısı: {df['Ürün Grubu'].nunique()}")
            
            # Segment 6 özel analizi
            segment6_df = df[df['segment'] == 'Segment 6']
            print(f"\n🎯 SEGMENT 6 ÖZEL ANALİZİ:")
            print(f"Segment 6 satır sayısı: {len(segment6_df):,}")
            print(f"Segment 6 benzersiz ürün sayısı: {segment6_df['StockAd'].nunique()}")
            print(f"Segment 6 benzersiz ürün grubu sayısı: {segment6_df['Ürün Grubu'].nunique()}")
            
            # Segment 6 ürün grupları
            seg6_groups = segment6_df['Ürün Grubu'].value_counts()
            print(f"Segment 6 tüm ürün grupları: {dict(seg6_groups)}")
            
            # Ürün grubu-ürün ilişkisi örnekleri
            print(f"\n🔗 ÜRÜN GRUBU - ÜRÜN İLİŞKİSİ ÖRNEKLERİ (Segment 6):")
            sample_products = segment6_df[['StockAd', 'Ürün Grubu']].drop_duplicates().head(10)
            for _, row in sample_products.iterrows():
                ürün_adı = row['StockAd'][:50] + "..." if len(row['StockAd']) > 50 else row['StockAd']
                print(f"  Ürün: '{ürün_adı}' -> Grup: '{row['Ürün Grubu']}'")
        
        # Temizleme sonrası kontrol
        initial_rows = len(df)
        
        log_performance(f"🧹 Veri temizlendi ve string'e çevrildi: {df.shape[0]} satır", prep_start)
        
        # Debug: Veri tiplerini kontrol et
        if debug:
            print(f"\n🔍 VERİ TİPİ KONTROLÜ:")
            print(f"StockAd tipi: {df['StockAd'].dtype}")
            print(f"segment tipi: {df['segment'].dtype}")
            print(f"Ürün Grubu tipi: {df['Ürün Grubu'].dtype}")
            print(f"StockAd örnekleri: {df['StockAd'].head().tolist()}")
            print(f"Unique StockAd sayısı: {df['StockAd'].nunique()}")
            
            # NaN kontrolü
            nan_check = df[['StockAd', 'segment', 'Ürün Grubu']].isnull().sum()
            print(f"NaN sayıları: {nan_check.to_dict()}")
        
        gc.collect()
        log_performance("✅ Veri hazırlama tamamlandı", start_time)
        
        return df
        
    except Exception as e:
        log_performance(f"❌ Veri yükleme hatası: {e}")
        return None

def create_transactions_vectorized(segment_df, debug=False):
    """
    Vectorized transaction oluşturma - ÜRÜN ADLARI İLE (Doğru Yaklaşım)
    """
    start_time = log_performance("🔄 Transaction oluşturma başladı")
    
    # Veri zaten load_and_prepare_data_optimized'da temizlendi
    if debug:
        print(f"🔍 Segment veri kontrolü:")
        print(f"  - StockAd tipi: {segment_df['StockAd'].dtype}")
        print(f"  - Ürün Grubu tipi: {segment_df['Ürün Grubu'].dtype}")
        print(f"  - StockAd örnekleri: {segment_df['StockAd'].head().tolist()}")
        print(f"  - Ürün Grubu örnekleri: {segment_df['Ürün Grubu'].head().tolist()}")
        print(f"  - NaN kontrolü: {segment_df[['StockAd', 'Ürün Grubu']].isnull().sum().to_dict()}")
        print(f"  - Benzersiz ürün sayısı: {segment_df['StockAd'].nunique()}")
        print(f"  - Benzersiz ürün grubu sayısı: {segment_df['Ürün Grubu'].nunique()}")
    
    # Son güvenlik kontrolü
    segment_df = segment_df[
        (segment_df['StockAd'].notna()) & 
        (segment_df['Ürün Grubu'].notna()) &
        (segment_df['StockAd'] != '') &
        (segment_df['Ürün Grubu'] != '')
    ].copy()
    
    if len(segment_df) == 0:
        log_performance("⚠️ Temizleme sonrası veri kalmadı")
        return [], {}
    
    # ÜRÜN ADLARI İLE TRANSACTION OLUŞTUR (Orijinal yaklaşım)
    grouped = segment_df.groupby(['carikod', 'yil', 'aylik'])['StockAd'].apply(
        lambda x: x.tolist()  # Ürün adları listesi
    ).reset_index()
    
    # Boş transaction'ları filtrele
    transactions_list = [
        trans for trans in grouped['StockAd'].tolist()
        if isinstance(trans, list) and len(trans) > 0
    ]
    
    # ÜRÜN ADI -> ÜRÜN GRUBU mapping (Filtreleme için)
    product_groups = dict(zip(segment_df['StockAd'], segment_df['Ürün Grubu']))
    
    if debug:
        print(f"🔍 Transaction oluşturma sonucu (ÜRÜN ADLARI İLE):")
        print(f"  - Transaction sayısı: {len(transactions_list)}")
        print(f"  - Benzersiz ürün sayısı: {len(product_groups)}")
        if len(transactions_list) > 0:
            print(f"  - İlk transaction: {transactions_list[0][:5]}")  # İlk 5 ürün
        if len(product_groups) > 0:
            first_items = list(product_groups.items())[:5]
            print(f"  - İlk product groups: {first_items}")
        
        # Transaction boyutları analizi
        transaction_sizes = [len(trans) for trans in transactions_list]
        if transaction_sizes:
            print(f"  - Ortalama transaction boyutu: {sum(transaction_sizes)/len(transaction_sizes):.1f}")
            print(f"  - En büyük transaction boyutu: {max(transaction_sizes)}")
            print(f"  - En küçük transaction boyutu: {min(transaction_sizes)}")
    
    log_performance(f"✅ {len(transactions_list)} geçerli transaction oluşturuldu", start_time)
    
    return transactions_list, product_groups

def filter_rules_vectorized(rules, product_groups, debug=False):
    """
    Vectorized rule filtreleme - AYNI GRUP KURALLARINI FİLTRELE (Doğru Yaklaşım)
    """
    start_time = log_performance("🔍 Rule filtreleme başladı")
    
    if debug:
        print(f"🔍 Rules tipi: {type(rules)}")
        print(f"🔍 Rules boyutu: {len(rules)}")
        if len(rules) > 0:
            first_rule = rules.iloc[0]
            print(f"🔍 İlk rule antecedents tipi: {type(first_rule['antecedents'])}")
            print(f"🔍 İlk rule antecedents içeriği: {first_rule['antecedents']}")
            print(f"🔍 Product groups sample: {list(product_groups.items())[:5]}")
    
    rules_list = []
    
    for idx, (_, rule) in enumerate(rules.iterrows()):
        try:
            # frozenset'leri liste'ye çevir
            antecedents = list(rule['antecedents'])
            consequents = list(rule['consequents'])
            
            if debug and idx < 3:
                print(f"🔍 Rule {idx}: antecedents={antecedents}, consequents={consequents}")
            
            # Boş listeler kontrolü
            if not antecedents or not consequents:
                if debug and idx < 5:
                    print(f"⚠️ Rule {idx}: Boş antecedents veya consequents")
                continue
            
            # ÜRÜN GRUPLARINI BUL
            ant_groups = {product_groups.get(item, 'Unknown') for item in antecedents}
            con_groups = {product_groups.get(item, 'Unknown') for item in consequents}
            
            # 'Unknown' grupları filtrele
            ant_groups = {g for g in ant_groups if g != 'Unknown'}
            con_groups = {g for g in con_groups if g != 'Unknown'}
            
            # FARKLI GRUP KONTROLÜ - AYNI GRUPTAN OLANLAR FİLTRELENSİN
            if ant_groups and con_groups and not ant_groups.intersection(con_groups):
                rules_list.append({
                    'antecedents': ', '.join(antecedents),
                    'consequents': ', '.join(consequents),
                    'antecedent_support': rule['antecedent support'],
                    'consequent_support': rule['consequent support'],
                    'support': rule['support'],
                    'confidence': rule['confidence'],
                    'lift': rule['lift'],
                    'antecedent_groups': ', '.join(sorted(ant_groups)),
                    'consequent_groups': ', '.join(sorted(con_groups))
                })
                
                if debug and len(rules_list) <= 3:
                    print(f"✅ Geçerli kural {len(rules_list)}: {antecedents} -> {consequents}")
                    print(f"   Gruplar: {ant_groups} -> {con_groups}")
            else:
                if debug and idx < 5:
                    print(f"❌ Filtrelendi {idx}: Aynı grup - {ant_groups} ∩ {con_groups}")
        
        except Exception as e:
            if debug:
                print(f"⚠️ Rule {idx} işlenirken hata: {e}")
                print(f"  - Antecedents tipi: {type(rule['antecedents'])}")
                print(f"  - Consequents tipi: {type(rule['consequents'])}")
            continue
    
    log_performance(f"✅ {len(rules_list)} geçerli kural bulundu", start_time)
    
    if debug:
        print(f"🔍 Toplam işlenen rule: {len(rules)}")
        print(f"🔍 Geçerli kural sayısı: {len(rules_list)}")
        print(f"🔍 Filtrelenen oran: {((len(rules) - len(rules_list))/len(rules)*100):.1f}%")
        if len(rules_list) > 0:
            print(f"🔍 İlk geçerli kural: {rules_list[0]}")
    
    return rules_list

@timing_decorator
def process_segment_optimized(df, segment, min_support=0.03, min_confidence=0.5, debug=False):
    """
    Optimized segment processing - TÜM SEGMENTLER DESTEKLİ
    """
    log_performance(f"🎯 {segment} işlemi başladı")
    segment_start = time.time()
    
    # Segment verilerini filtrele - TÜM_SEGMENTLER ise filtreleme yok
    filter_start = time.time()
    if segment == "TÜM_SEGMENTLER":
        segment_df = df.copy()
        log_performance(f"📊 Tüm segmentler kullanılıyor: {len(segment_df)} satır", filter_start)
    else:
        segment_df = df[df['segment'] == segment].copy()
        log_performance(f"📊 Segment filtrelendi: {len(segment_df)} satır", filter_start)
    
    if len(segment_df) < 50:
        log_performance(f"⚠️ {segment} için yeterli veri yok ({len(segment_df)} satır)")
        return []
    
    # Transaction oluşturma (optimized)
    transactions_list, product_groups = create_transactions_vectorized(segment_df, debug)
    
    del segment_df
    gc.collect()
    
    if len(transactions_list) < 10:
        log_performance(f"⚠️ {segment} için yeterli transaction yok")
        return []
    
    try:
        # Transaction encoding
        encoding_start = time.time()
        te = TransactionEncoder()
        
        # Boş transaction kontrolü
        if not transactions_list or all(len(trans) == 0 for trans in transactions_list):
            log_performance("⚠️ Boş transaction listesi - atlanıyor")
            return []
        
        te_ary = te.fit(transactions_list).transform(transactions_list)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        
        log_performance(f"🔄 Transaction encoding tamamlandı: {df_encoded.shape}", encoding_start)
        
        del transactions_list, te_ary
        gc.collect()
        
        # FP-Growth
        fpgrowth_start = time.time()
        frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)
        log_performance(f"⚡ FP-Growth tamamlandı: {len(frequent_itemsets)} itemset", fpgrowth_start)
        
        del df_encoded
        gc.collect()
        
        if len(frequent_itemsets) == 0:
            log_performance(f"⚠️ {segment} için frequent itemsets bulunamadı")
            return []
        
        # Association rules
        rules_start = time.time()
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        log_performance(f"📋 Association rules oluşturuldu: {len(rules)} kural", rules_start)
        
        del frequent_itemsets
        gc.collect()
        
        if len(rules) == 0:
            log_performance(f"⚠️ {segment} için rules bulunamadı")
            return []
        
        # Rule filtreleme (optimized)
        filtered_rules = filter_rules_vectorized(rules, product_groups, debug)
        
        # Segment bilgisini ekle
        for rule in filtered_rules:
            rule['segment'] = segment
        
        del rules, product_groups
        gc.collect()
        
        log_performance(f"✅ {segment} işlemi tamamlandı", segment_start)
        
        return filtered_rules
        
    except Exception as e:
        log_performance(f"❌ {segment} işlem hatası: {e}")
        gc.collect()
        return []

def create_recommendations_vectorized(rules_df, df, debug=False):
    """
    Vectorized recommendation generation - TÜM SEGMENTLER DESTEKLİ
    """
    start_time = log_performance("💡 Öneriler hesaplanıyor")
    
    # Son ay verilerini al
    latest_month = df['aylik'].max()
    latest_year = df['yil'].max()
    
    last_month_data = df[(df['aylik'] == latest_month) & (df['yil'] == latest_year)].copy()
    
    # Vectorized grouping - FOR döngüsü yerine
    prep_start = time.time()
    sales_by_distributor = last_month_data.groupby('carikod')['StockAd'].apply(set).to_dict()
    
    # TÜM_SEGMENTLER ise segment filtrelemesi yapma
    if 'TÜM_SEGMENTLER' in rules_df['segment'].values:
        # Tüm bayiler kullanılacak
        all_distributors = set(df['carikod'].unique())
        segment_distributors = {'TÜM_SEGMENTLER': all_distributors}
    else:
        # Normal segment bazlı çalışma
        segment_distributors = df.groupby('segment')['carikod'].apply(set).to_dict()
    
    log_performance(f"📊 Öneri hazırlıkları tamamlandı", prep_start)
    
    recommendations = []
    
    # Batch processing - büyük döngüleri böl
    batch_size = 100
    total_rules = len(rules_df)
    
    for batch_start in range(0, total_rules, batch_size):
        batch_end = min(batch_start + batch_size, total_rules)
        batch_rules = rules_df.iloc[batch_start:batch_end]
        
        for _, rule in batch_rules.iterrows():
            segment = rule['segment']
            antecedents = set(rule['antecedents'].split(', '))
            consequents = set(rule['consequents'].split(', '))
            
            # Bu segment için bayiler
            distributors = segment_distributors.get(segment, set())
            
            # Vectorized processing
            for distributor in distributors:
                distributor_sales = sales_by_distributor.get(distributor, set())
                
                # Set operations (hızlı)
                has_antecedent = bool(distributor_sales & antecedents)
                has_consequent = bool(distributor_sales & consequents)
                
                if has_antecedent and not has_consequent:
                    for consequent in consequents:
                        recommendations.append({
                            'carikod': distributor,
                            'segment': segment,
                            'satilan_urun': rule['antecedents'],
                            'onerilen_urun': consequent,
                            'confidence': rule['confidence'],
                            'lift': rule['lift'],
                            'support': rule['support']
                        })
        
        # Progress log
        if batch_start % (batch_size * 10) == 0:
            log_performance(f"📈 İşlenen rules: {batch_end}/{total_rules}")
    
    log_performance(f"✅ Öneriler tamamlandı: {len(recommendations)} öneri", start_time)
    
    return recommendations

@timing_decorator
def main_memory_optimized(debug=True):  # Debug default olarak True yap
    """
    Memory-optimized ana fonksiyon - TÜM SEGMENTLER BİRLİKTE AYLIK ANALİZ
    """
    file_path = "/Users/esranuryigit/Desktop/BirliktelikAnalizi_Filtresiz.xlsx"
    output_path = "/Users/esranuryigit/Desktop/urun_onerileri_sonuc_optimized.xlsx"
    
    print("=== PERFORMANCE-OPTIMIZED ÜRÜN ÖNERİ SİSTEMİ ===")
    print("=== TÜM SEGMENTLER BİRLİKTE AYLIK ANALİZ ===")
    print("=" * 60)
    
    main_start = time.time()
    log_performance("🚀 Ana süreç başladı")
    
    # Veriyi yükle - artık veri tipleri düzgün
    df = load_and_prepare_data_optimized(file_path, debug)
    if df is None:
        return
    
    # Veri özeti
    summary_start = time.time()
    print(f"\n📊 VERİ ÖZETİ:")
    print(f"Toplam satış kaydı: {len(df):,}")
    print(f"Bayi sayısı: {df['carikod'].nunique():,}")
    print(f"Ürün sayısı: {df['StockAd'].nunique():,}")
    print(f"Segment sayısı: {df['segment'].nunique()}")
    print(f"Ürün grubu sayısı: {df['Ürün Grubu'].nunique()}")
    
    segments = df['segment'].unique()
    log_performance(f"📋 Segmentler: {list(segments)}", summary_start)
    
    # TÜM SEGMENTLER BİRLİKTE AYLIK ANALİZ
    print(f"\n🎯 TÜM SEGMENTLER BİRLİKTE İŞLENİYOR...")
    
    # İlk test için makul parametreler
    test_support = 0.01  # %1 - daha mantıklı
    test_confidence = 0.3  # %30 - daha mantıklı
    
    print(f"\n🔧 PARAMETRE AYARLARI:")
    print(f"Min Support: {test_support} ({test_support*100:.1f}%)")
    print(f"Min Confidence: {test_confidence} ({test_confidence*100:.1f}%)")
    print(f"Analiz tipi: TÜM SEGMENTLER - AYLIK TRANSACTION")
    
    # TÜM VERİ İLE SEGMENT-BAĞIMSIZ ANALİZ
    segment_rules = process_segment_optimized(df, segment="TÜM_SEGMENTLER", 
                                            min_support=test_support, 
                                            min_confidence=test_confidence,
                                            debug=debug)
    
    all_rules = segment_rules
    log_performance(f"✅ Tüm segmentler tamamlandı. Toplam kural: {len(all_rules)}")
    
    # Sonuçları DataFrame'e çevir
    df_creation_start = time.time()
    
    if len(all_rules) == 0:
        log_performance("⚠️ Hiç kural bulunamadı - boş DataFrame oluşturuluyor")
        rules_df = pd.DataFrame()
        
        # Neden kural bulunamadığına dair analiz
        print(f"\n🔍 KURAL BULUNAMAMA ANALİZİ:")
        print(f"Toplam satır sayısı: {len(df):,}")
        print(f"Unique ürün sayısı: {df['StockAd'].nunique()}")
        print(f"Unique bayi sayısı: {df['carikod'].nunique()}")
        
        # Transaction sayısı tahmini
        transactions = df.groupby(['carikod', 'yil', 'aylik']).size()
        print(f"Tahmini transaction sayısı: {len(transactions)}")
        print(f"Ortalama ürün/transaction: {len(df) / max(len(transactions), 1):.1f}")
        
        # En çok satılan ürünler
        top_products = df['StockAd'].value_counts().head(10)
        print(f"En çok satılan ürünler:")
        for product, count in top_products.items():
            print(f"  {product[:50]}...: {count:,} adet")
    else:
        rules_df = pd.DataFrame(all_rules)
        rules_df = rules_df.sort_values(by='support', ascending=False).head(5000)
        log_performance(f"📊 Rules DataFrame oluşturuldu: {len(rules_df)} kural", df_creation_start)
    
    del all_rules
    gc.collect()
    
    # Önerileri bul
    if not rules_df.empty:
        recommendations = create_recommendations_vectorized(rules_df, df, debug)
        recommendations_df = pd.DataFrame(recommendations)
        
        # Excel'e kaydet
        save_start = time.time()
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                if not recommendations_df.empty:
                    recommendations_df.to_excel(writer, sheet_name='Öneriler', index=False)
                    log_performance(f"💾 Öneriler kaydedildi: {len(recommendations_df)} satır")
                else:
                    # Boş öneri DataFrame'i kaydet
                    pd.DataFrame({'Mesaj': ['Hiç öneri bulunamadı']}).to_excel(writer, sheet_name='Öneriler', index=False)
                    log_performance("⚠️ Boş öneri listesi kaydedildi")
                
                if not rules_df.empty:
                    rules_df.to_excel(writer, sheet_name='Kurallar', index=False)
                    log_performance(f"💾 Kurallar kaydedildi: {len(rules_df)} satır")
                else:
                    # Boş kural DataFrame'i kaydet
                    pd.DataFrame({'Mesaj': ['Hiç kural bulunamadı']}).to_excel(writer, sheet_name='Kurallar', index=False)
                    log_performance("⚠️ Boş kural listesi kaydedildi")
            
            log_performance(f"💾 Excel dosyası kaydedildi: {output_path}", save_start)
            
        except Exception as e:
            log_performance(f"❌ Excel kaydetme hatası: {e}")
        
        # Özet rapor
        if not recommendations_df.empty:
            print(f"\n📊 ÖZET RAPOR:")
            print(f"Toplam öneri: {len(recommendations_df):,}")
            print(f"Öneri alan bayi sayısı: {recommendations_df['carikod'].nunique():,}")
            
            # En iyi öneriler
            top_recommendations = recommendations_df.nlargest(10, 'confidence')
            print(f"\n🏆 En yüksek confidence'lı öneriler:")
            for _, rec in top_recommendations.iterrows():
                print(f"  Bayi: {rec['carikod']}, Ürün: {rec['onerilen_urun']}, Confidence: {rec['confidence']:.2%}")
        else:
            print(f"\n⚠️ ÖZET RAPOR: Hiç öneri bulunamadı")
    
    else:
        log_performance("❌ Hiç kural bulunamadı - öneri oluşturulamıyor")
        
        # Yine de boş Excel dosyası kaydet
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                pd.DataFrame({'Mesaj': ['Hiç kural bulunamadı - parametreleri kontrol edin']}).to_excel(writer, sheet_name='Sonuç', index=False)
            log_performance(f"💾 Boş sonuç dosyası kaydedildi: {output_path}")
        except Exception as e:
            log_performance(f"❌ Boş dosya kaydetme hatası: {e}")
    
    log_performance(f"🎯 TÜM İŞLEM TAMAMLANDI!", main_start)

if __name__ == "__main__":
    main_memory_optimized(debug=True)