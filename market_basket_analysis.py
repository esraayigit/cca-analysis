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
    Excel dosyasını memory-efficient şekilde yükler
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
        
        # Sütun standardizasyonu
        prep_start = time.time()
        expected_columns = ['carikod', 'yil', 'aylik', 'cirokod_aylik', 'ciro_degeri', 'StockAd','segment', 'StockId','StockKod','Ürün Grubu']
        
        if len(df.columns) >= len(expected_columns):
            df.columns = expected_columns[:len(df.columns)]
        
        columns_to_keep = ['carikod', 'yil', 'aylik', 'StockAd','segment', 'StockId','StockKod','Ürün Grubu']
        df = df[columns_to_keep].copy()
        
        # Veri tipi optimizasyonu
        df['carikod'] = df['carikod'].astype('category')
        df['segment'] = df['segment'].astype('category')
        df['Ürün Grubu'] = df['Ürün Grubu'].astype('category')
        df['yil'] = df['yil'].astype('int16')
        df['aylik'] = df['aylik'].astype('int8')
        
        # Temizleme
        initial_rows = len(df)
        df = df.dropna(subset=['StockAd', 'segment', 'Ürün Grubu'])
        
        log_performance(f"🧹 Veri temizlendi: {initial_rows} -> {df.shape[0]} satır", prep_start)
        
        gc.collect()
        log_performance("✅ Veri hazırlama tamamlandı", start_time)
        
        return df
        
    except Exception as e:
        log_performance(f"❌ Veri yükleme hatası: {e}")
        return None

def create_transactions_vectorized(segment_df, debug=False):
    """
    Vectorized transaction oluşturma (FOR döngüsü yerine) - NaN handling eklendi
    """
    start_time = log_performance("🔄 Transaction oluşturma başladı")
    
    # NaN değerleri temizle ve string'e çevir
    segment_df = segment_df.dropna(subset=['StockAd', 'Ürün Grubu']).copy()
    
    # StockAd ve Ürün Grubu'nu string'e çevir
    segment_df['StockAd'] = segment_df['StockAd'].astype(str)
    segment_df['Ürün Grubu'] = segment_df['Ürün Grubu'].astype(str)
    
    # 'nan' string'lerini temizle
    segment_df = segment_df[
        (segment_df['StockAd'] != 'nan') & 
        (segment_df['Ürün Grubu'] != 'nan')
    ]
    
    # Vectorized grouping - FOR döngüsü yerine
    grouped = segment_df.groupby(['carikod', 'yil', 'aylik'])['StockAd'].apply(
        lambda x: [str(item) for item in x.tolist() if str(item) != 'nan']
    ).reset_index()
    
    # Boş transaction'ları filtrele
    transactions_list = [
        trans for trans in grouped['StockAd'].tolist()
        if isinstance(trans, list) and len(trans) > 0
    ]
    
    # Ürün gruplarını dictionary comprehension ile oluştur (NaN kontrolü ile)
    product_groups = {}
    for stock, group in zip(segment_df['StockAd'], segment_df['Ürün Grubu']):
        stock_str = str(stock)
        group_str = str(group)
        if stock_str != 'nan' and group_str != 'nan':
            product_groups[stock_str] = group_str
    
    log_performance(f"✅ {len(transactions_list)} geçerli transaction oluşturuldu", start_time)
    
    return transactions_list, product_groups

def filter_rules_vectorized(rules, product_groups, debug=False):
    """
    Vectorized rule filtreleme (FOR döngüsü yerine) - String handling düzeltildi
    """
    start_time = log_performance("🔍 Rule filtreleme başladı")
    
    if debug:
        print(f"🔍 Rules tipi: {type(rules)}")
        print(f"🔍 Rules boyutu: {len(rules)}")
        if len(rules) > 0:
            first_rule = rules.iloc[0]
            print(f"🔍 İlk rule antecedents tipi: {type(first_rule['antecedents'])}")
            print(f"🔍 İlk rule antecedents içeriği: {first_rule['antecedents']}")
    
    # Vectorized approach - pandas operations kullan
    rules_list = []
    
    # Rules'ı liste olarak hazırla - frozenset'i string'e çevir
    antecedents_list = []
    consequents_list = []
    
    for idx, (_, rule) in enumerate(rules.iterrows()):
        try:
            # frozenset'leri string listesine çevir - güvenli şekilde
            if hasattr(rule['antecedents'], '__iter__'):
                # frozenset veya set ise
                ant_items = []
                for item in rule['antecedents']:
                    if pd.notna(item):  # NaN kontrolü
                        str_item = str(item).strip()
                        if str_item and str_item != 'nan':
                            ant_items.append(str_item)
            else:
                # Tek item ise
                if pd.notna(rule['antecedents']):
                    str_item = str(rule['antecedents']).strip()
                    ant_items = [str_item] if str_item and str_item != 'nan' else []
                else:
                    ant_items = []
            
            if hasattr(rule['consequents'], '__iter__'):
                # frozenset veya set ise
                con_items = []
                for item in rule['consequents']:
                    if pd.notna(item):  # NaN kontrolü
                        str_item = str(item).strip()
                        if str_item and str_item != 'nan':
                            con_items.append(str_item)
            else:
                # Tek item ise
                if pd.notna(rule['consequents']):
                    str_item = str(rule['consequents']).strip()
                    con_items = [str_item] if str_item and str_item != 'nan' else []
                else:
                    con_items = []
            
            antecedents_list.append(ant_items)
            consequents_list.append(con_items)
            
            if debug and idx < 3:  # İlk 3 rule için debug
                print(f"🔍 Rule {idx}: ant_items={ant_items}, con_items={con_items}")
                
        except Exception as e:
            if debug:
                print(f"⚠️ Rule {idx} işlenirken hata: {e}")
            antecedents_list.append([])
            consequents_list.append([])
    
    # Vectorized group checking
    for i, (_, rule) in enumerate(rules.iterrows()):
        antecedents = antecedents_list[i]
        consequents = consequents_list[i]
        
        # Boş listeler kontrolü
        if not antecedents or not consequents:
            if debug and i < 5:
                print(f"⚠️ Rule {i}: Boş antecedents ({len(antecedents)}) veya consequents ({len(consequents)})")
            continue
        
        try:
            # Set operations kullan (hızlı)
            ant_groups = {product_groups.get(str(item), 'Unknown') for item in antecedents}
            con_groups = {product_groups.get(str(item), 'Unknown') for item in consequents}
            
            # 'Unknown' grupları filtrele
            ant_groups = {g for g in ant_groups if g != 'Unknown'}
            con_groups = {g for g in con_groups if g != 'Unknown'}
            
            # Farklı ürün gruplarından olanları al
            if ant_groups and con_groups and not ant_groups.intersection(con_groups):
                rules_list.append({
                    'antecedents': ', '.join(antecedents),
                    'consequents': ', '.join(consequents),
                    'antecedent_support': rule['antecedent support'],
                    'consequent_support': rule['consequent support'],
                    'support': rule['support'],
                    'confidence': rule['confidence'],
                    'lift': rule['lift'],
                    'antecedent_groups': ', '.join(ant_groups),
                    'consequent_groups': ', '.join(con_groups)
                })
                
                if debug and len(rules_list) <= 3:
                    print(f"✅ Geçerli kural {len(rules_list)}: {antecedents} -> {consequents}")
        
        except Exception as e:
            if debug:
                print(f"⚠️ Rule {i} grup kontrolünde hata: {e}")
            continue
    
    log_performance(f"✅ {len(rules_list)} geçerli kural bulundu", start_time)
    
    if debug:
        print(f"🔍 Toplam işlenen rule: {len(rules)}")
        print(f"🔍 Geçerli kural sayısı: {len(rules_list)}")
        if len(rules_list) > 0:
            print(f"🔍 İlk geçerli kural: {rules_list[0]}")
    
    return rules_list

@timing_decorator
def process_segment_optimized(df, segment, min_support=0.03, min_confidence=0.5, debug=False):
    """
    Optimized segment processing
    """
    log_performance(f"🎯 {segment} segmenti başladı")
    segment_start = time.time()
    
    # Segment verilerini filtrele
    filter_start = time.time()
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
        
        log_performance(f"✅ {segment} segmenti tamamlandı", segment_start)
        
        return filtered_rules
        
    except Exception as e:
        log_performance(f"❌ {segment} segmenti hatası: {e}")
        gc.collect()
        return []

def create_recommendations_vectorized(rules_df, df, debug=False):
    """
    Vectorized recommendation generation
    """
    start_time = log_performance("💡 Öneriler hesaplanıyor")
    
    # Son ay verilerini al
    latest_month = df['aylik'].max()
    latest_year = df['yil'].max()
    
    last_month_data = df[(df['aylik'] == latest_month) & (df['yil'] == latest_year)].copy()
    
    # Vectorized grouping - FOR döngüsü yerine
    prep_start = time.time()
    sales_by_distributor = last_month_data.groupby('carikod')['StockAd'].apply(set).to_dict()
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
def main_memory_optimized(debug=False):
    """
    Memory-optimized ana fonksiyon
    """
    file_path = "/Users/esranuryigit/Desktop/BirliktelikAnalizi_Filtresiz.xlsx"
    output_path = "/Users/esranuryigit/Desktop/urun_onerileri_sonuc_optimized.xlsx"
    
    print("=== PERFORMANCE-OPTIMIZED ÜRÜN ÖNERİ SİSTEMİ ===")
    print("=" * 60)
    
    main_start = time.time()
    log_performance("🚀 Ana süreç başladı")
    
    # Veriyi yükle
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
    
    # Her segment için işlem - DAHA DÜŞÜK PARAMETRELER
    all_rules = []
    
    # Test için daha düşük parametreler
    test_support = 0.005  # %0.5 (çok düşük)
    test_confidence = 0.1  # %10 (çok düşük)
    
    print(f"\n🔧 TEST PARAMETRELERİ:")
    print(f"Min Support: {test_support} ({test_support*100:.1f}%)")
    print(f"Min Confidence: {test_confidence} ({test_confidence*100:.1f}%)")
    
    for i, segment in enumerate(segments, 1):
        print(f"\n{'='*20} SEGMENT {i}/{len(segments)}: {segment} {'='*20}")
        
        segment_rules = process_segment_optimized(df, segment, 
                                                min_support=test_support, 
                                                min_confidence=test_confidence,
                                                debug=debug)
        
        all_rules.extend(segment_rules)
        log_performance(f"✅ {segment} tamamlandı. Toplam kural: {len(all_rules)}")
        
        # İlk segment'te kural bulunursa diğerlerine devam et
        if i == 1 and len(segment_rules) > 0:
            print(f"🎉 İLK SEGMENT'TE {len(segment_rules)} KURAL BULUNDU!")
            print("Diğer segmentlere devam ediliyor...")
        elif i == 1 and len(segment_rules) == 0:
            print("⚠️ İlk segment'te kural bulunamadı, parametreler çok yüksek olabilir")
        
        gc.collect()
    
    # Sonuçları DataFrame'e çevir
    df_creation_start = time.time()
    
    if len(all_rules) == 0:
        log_performance("⚠️ Hiç kural bulunamadı - boş DataFrame oluşturuluyor")
        rules_df = pd.DataFrame()
        
        # Neden kural bulunamadığına dair analiz
        print(f"\n🔍 KURAL BULUNAMAMA ANALİZİ:")
        
        # Her segment için temel istatistikler
        for segment in segments[:3]:  # İlk 3 segment
            seg_df = df[df['segment'] == segment]
            print(f"\n📊 {segment} Analizi:")
            print(f"  - Satır sayısı: {len(seg_df):,}")
            print(f"  - Unique ürün sayısı: {seg_df['StockAd'].nunique()}")
            print(f"  - Unique bayi sayısı: {seg_df['carikod'].nunique()}")
            
            # Transaction sayısı tahmini
            transactions = seg_df.groupby(['carikod', 'yil', 'aylik']).size()
            print(f"  - Tahmini transaction sayısı: {len(transactions)}")
            print(f"  - Ortalama ürün/transaction: {len(seg_df) / max(len(transactions), 1):.1f}")
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