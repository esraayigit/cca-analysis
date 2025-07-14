import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
import gc  # Garbage collection için
import psutil  # Memory monitoring için
import os
warnings.filterwarnings('ignore')

def get_memory_usage():
    """
    Mevcut memory kullanımını gösterir
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return f"RAM Kullanımı: {memory_info.rss / 1024 / 1024:.2f} MB"

def load_and_prepare_data_optimized(file_path, debug=False):
    """
    Excel dosyasını memory-efficient şekilde yükler
    """
    try:
        print("Excel dosyası yükleniyor...")
        if debug:
            print(get_memory_usage())
        
        # Chunk size ile okuma (çok büyük dosyalar için)
        try:
            # Önce dosya boyutunu kontrol et
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"Dosya boyutu: {file_size:.2f} MB")
            
            if file_size > 50:  # 50MB'dan büyükse chunk ile oku
                print("Büyük dosya tespit edildi, chunk processing aktif")
                chunks = []
                for chunk in pd.read_excel(file_path, chunksize=10000):
                    chunks.append(chunk)
                    if debug:
                        print(f"Chunk yüklendi, {get_memory_usage()}")
                df = pd.concat(chunks, ignore_index=True)
                del chunks
                gc.collect()
            else:
                df = pd.read_excel(file_path)
                
        except:
            # Chunk reading başarısız olursa normal okuma
            df = pd.read_excel(file_path)
        
        print(f"Veri yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")
        if debug:
            print(get_memory_usage())
        
        # Sütun isimlerini standartlaştır
        expected_columns = ['cairkod', 'yil', 'aylik', 'urokod_aylik', 'deger', 'StockAd', 'segment', 'StockId', 'stockkod', 'Ürün Grubu']
        
        if len(df.columns) >= len(expected_columns):
            df.columns = expected_columns[:len(df.columns)]
        
        # Sadece gerekli sütunları tut
        columns_to_keep = ['cairkod', 'yil', 'aylik', 'deger', 'StockAd', 'segment', 'Ürün Grubu']
        df = df[columns_to_keep].copy()
        
        # Veri tiplerini optimize et
        df['cairkod'] = df['cairkod'].astype('category')
        df['segment'] = df['segment'].astype('category')
        df['Ürün Grubu'] = df['Ürün Grubu'].astype('category')
        df['yil'] = df['yil'].astype('int16')
        df['aylik'] = df['aylik'].astype('int8')
        
        # Null değerleri temizle
        initial_rows = len(df)
        df = df.dropna(subset=['StockAd', 'segment', 'Ürün Grubu'])
        
        print(f"Veri temizlendi: {initial_rows} -> {df.shape[0]} satır ({initial_rows - df.shape[0]} satır silindi)")
        
        # Memory temizliği
        gc.collect()
        
        if debug:
            print(get_memory_usage())
            print(f"Veri tipleri: {df.dtypes}")
        
        return df
        
    except Exception as e:
        print(f"Veri yükleme hatası: {e}")
        return None

def process_segment_with_memory_limit(df, segment, min_support=0.03, min_confidence=0.5, debug=False):
    """
    Tek segment için memory-limited işlem
    """
    print(f"\n{segment} segmenti işleniyor...")
    
    if debug:
        print(f"İşlem öncesi: {get_memory_usage()}")
    
    # Segment verilerini filtrele
    segment_df = df[df['segment'] == segment].copy()
    
    if len(segment_df) < 50:  # Minimum veri kontrolü
        print(f"{segment} segmenti için yeterli veri yok ({len(segment_df)} satır)")
        del segment_df
        gc.collect()
        return []
    
    # Transaction matrix oluştur (memory efficient)
    transactions_list = []
    product_groups = {}
    
    # Gruplama işlemi
    grouped = segment_df.groupby(['cairkod', 'yil', 'aylik'])
    
    for name, group in grouped:
        products = group['StockAd'].tolist()
        if len(products) > 0:  # Boş transaction'ları ekle
            transactions_list.append(products)
        
        # Ürün gruplarını kaydet (memory efficient)
        for _, row in group.iterrows():
            if row['StockAd'] not in product_groups:
                product_groups[row['StockAd']] = row['Ürün Grubu']
    
    del segment_df, grouped
    gc.collect()
    
    if debug:
        print(f"Transaction oluşturma sonrası: {get_memory_usage()}")
    
    print(f"{segment} için {len(transactions_list)} transaction oluşturuldu")
    
    if len(transactions_list) < 10:
        print(f"{segment} segmenti için yeterli transaction yok")
        return []
    
    try:
        # Transaction encoder (memory limited)
        te = TransactionEncoder()
        te_ary = te.fit(transactions_list).transform(transactions_list)
        
        # Sparse matrix kullan (memory efficient)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        
        del transactions_list, te_ary
        gc.collect()
        
        if debug:
            print(f"Encoding sonrası: {get_memory_usage()}")
        
        # Apriori with memory optimization
        frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True, low_memory=True)
        
        del df_encoded
        gc.collect()
        
        if len(frequent_itemsets) == 0:
            print(f"{segment} için frequent itemsets bulunamadı")
            return []
        
        if debug:
            print(f"Apriori sonrası: {get_memory_usage()}")
        
        # Association rules
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        
        del frequent_itemsets
        gc.collect()
        
        if len(rules) == 0:
            print(f"{segment} için rules bulunamadı")
            return []
        
        # Kuralları filtrele (farklı ürün grupları)
        filtered_rules = []
        rules_processed = 0
        
        for _, rule in rules.iterrows():
            rules_processed += 1
            
            # Her 100 rule'da bir memory check
            if rules_processed % 100 == 0 and debug:
                print(f"İşlenen rules: {rules_processed}/{len(rules)}, {get_memory_usage()}")
            
            antecedents = list(rule['antecedents'])
            consequents = list(rule['consequents'])
            
            # Ürün gruplarını kontrol et
            ant_groups = set([product_groups.get(item, 'Unknown') for item in antecedents])
            con_groups = set([product_groups.get(item, 'Unknown') for item in consequents])
            
            # Farklı ürün gruplarından olanları al
            if not ant_groups.intersection(con_groups):
                filtered_rules.append({
                    'segment': segment,
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
        
        print(f"{segment} için {len(filtered_rules)} geçerli kural bulundu")
        
        # Memory temizliği
        del rules, product_groups
        gc.collect()
        
        if debug:
            print(f"Segment tamamlandı: {get_memory_usage()}")
        
        return filtered_rules
        
    except Exception as e:
        print(f"{segment} segmenti için hata: {e}")
        gc.collect()
        return []

def main_memory_optimized(debug=False):
    """
    Memory-optimized ana fonksiyon
    """
    # Dosya yolu
    file_path = "/Users/esranuryigit/Desktop/BirliktelikAnalizi_son.xlsx"
    output_path = "/Users/esranuryigit/Desktop/urun_onerileri_sonuc_optimized.xlsx"
    
    print("=== MEMORY-OPTIMIZED ÜRÜN ÖNERİ SİSTEMİ ===")
    print("=" * 50)
    
    if debug:
        print(f"Başlangıç: {get_memory_usage()}")
    
    # Veriyi yükle
    df = load_and_prepare_data_optimized(file_path, debug)
    if df is None:
        return
    
    print(f"\nVeri özeti:")
    print(f"Toplam satış kaydı: {len(df):,}")
    print(f"Bayi sayısı: {df['cairkod'].nunique():,}")
    print(f"Ürün sayısı: {df['StockAd'].nunique():,}")
    print(f"Segment sayısı: {df['segment'].nunique()}")
    print(f"Ürün grubu sayısı: {df['Ürün Grubu'].nunique()}")
    
    if debug:
        print(f"Veri yükleme sonrası: {get_memory_usage()}")
    
    # Segmentleri al
    segments = df['segment'].unique()
    print(f"\nSegmentler: {list(segments)}")
    
    # Her segment için işlem
    all_rules = []
    
    for i, segment in enumerate(segments, 1):
        print(f"\n{'='*20} SEGMENT {i}/{len(segments)} {'='*20}")
        
        segment_rules = process_segment_with_memory_limit(df, segment, 
                                                        min_support=0.03, 
                                                        min_confidence=0.5,
                                                        debug=debug)
        
        all_rules.extend(segment_rules)
        
        # Progress
        print(f"✓ {segment} tamamlandı. Toplam kural sayısı: {len(all_rules)}")
        
        if debug:
            print(f"Segment {i} sonrası: {get_memory_usage()}")
        
        # Memory temizliği
        gc.collect()
    
    # Sonuçları DataFrame'e çevir
    rules_df = pd.DataFrame(all_rules)
    del all_rules
    gc.collect()
    
    print(f"\n🎉 TÜM SEGMENTLER TAMAMLANDI!")
    print(f"Toplam {len(rules_df)} kural bulundu")
    
    if debug:
        print(f"Rules DataFrame oluşturma sonrası: {get_memory_usage()}")
    
    # Önerileri bul (basitleştirilmiş versiyon - memory efficient)
    if not rules_df.empty:
        print("\nÖneriler hesaplanıyor...")
        
        # Son ay verilerini al (memory efficient)
        latest_month = df['aylik'].max()
        latest_year = df['yil'].max()
        
        last_month_data = df[(df['aylik'] == latest_month) & (df['yil'] == latest_year)].copy()
        
        recommendations = []
        
        for _, rule in rules_df.iterrows():
            segment = rule['segment']
            antecedents = rule['antecedents'].split(', ')
            consequents = rule['consequents'].split(', ')
            
            # Bu segment için bayileri al
            segment_distributors = df[df['segment'] == segment]['cairkod'].unique()
            
            for distributor in segment_distributors:
                # Bu bayinin son ay satışları
                distributor_sales = last_month_data[last_month_data['cairkod'] == distributor]['StockAd'].tolist()
                
                # Antecedent var, consequent yok mu?
                has_antecedent = any(ant in distributor_sales for ant in antecedents)
                has_consequent = any(con in distributor_sales for con in consequents)
                
                if has_antecedent and not has_consequent:
                    for consequent in consequents:
                        recommendations.append({
                            'cairkod': distributor,
                            'segment': segment,
                            'satilan_urun': rule['antecedents'],
                            'onerilen_urun': consequent,
                            'confidence': rule['confidence'],
                            'lift': rule['lift'],
                            'support': rule['support']
                        })
        
        recommendations_df = pd.DataFrame(recommendations)
        del last_month_data, recommendations
        gc.collect()
        
        print(f"Toplam {len(recommendations_df)} öneri bulundu")
        
        # Excel'e kaydet
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                if not recommendations_df.empty:
                    recommendations_df.to_excel(writer, sheet_name='Öneriler', index=False)
                if not rules_df.empty:
                    rules_df.to_excel(writer, sheet_name='Kurallar', index=False)
            
            print(f"✅ Sonuçlar kaydedildi: {output_path}")
            
        except Exception as e:
            print(f"❌ Excel kaydetme hatası: {e}")
        
        # Özet
        if not recommendations_df.empty:
            print(f"\n📊 ÖZET RAPOR:")
            print(f"Toplam öneri: {len(recommendations_df):,}")
            print(f"Öneri alan bayi sayısı: {recommendations_df['cairkod'].nunique():,}")
            
            # En iyi öneriler
            top_recommendations = recommendations_df.nlargest(10, 'confidence')
            print(f"\n🏆 En yüksek confidence'lı öneriler:")
            for _, rec in top_recommendations.iterrows():
                print(f"  Bayi: {rec['cairkod']}, Ürün: {rec['onerilen_urun']}, Confidence: {rec['confidence']:.2%}")
    
    else:
        print("❌ Hiç kural bulunamadı. Parametreleri kontrol edin.")
    
    if debug:
        print(f"\nSon memory durumu: {get_memory_usage()}")
    
    print("\n🎯 İşlem tamamlandı!")

if __name__ == "__main__":
    # Debug mode'u açmak için True yapın
    main_memory_optimized(debug=True)