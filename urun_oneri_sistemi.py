import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
import gc  # Garbage collection için
warnings.filterwarnings('ignore')

def load_and_prepare_data(file_path):
    """
    Excel dosyasını yükler ve veriyi hazırlar - Memory efficient
    """
    try:
        print("Excel dosyası yükleniyor...")
        
        # Sadece gerekli sütunları yükle
        usecols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # İlk 10 sütun
        
        df = pd.read_excel(file_path, usecols=usecols)
        print(f"Veri yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")
        print("Sütunlar:", df.columns.tolist())
        
        # Sütun isimlerini standartlaştır
        expected_columns = ['cairkod', 'yil', 'aylik', 'urokod_aylik', 'deger', 'StockAd', 'segment', 'StockId', 'stockkod', 'Ürün Grubu']
        
        if len(df.columns) >= len(expected_columns):
            df.columns = expected_columns[:len(df.columns)]
        
        # Gereksiz sütunları kaldır (memory tasarrufu için)
        columns_to_keep = ['cairkod', 'yil', 'aylik', 'deger', 'StockAd', 'segment', 'Ürün Grubu']
        df = df[columns_to_keep].copy()
        
        # Null değerleri temizle
        df = df.dropna(subset=['StockAd', 'segment', 'Ürün Grubu'])
        
        # Memory temizliği
        gc.collect()
        
        print(f"Veri temizlendi: {df.shape[0]} satır kaldı")
        return df
        
    except Exception as e:
        print(f"Veri yükleme hatası: {e}")
        return None

def create_transaction_matrix_by_segment(df, segment):
    """
    Belirli bir segment için transaction matrix oluşturur - Memory efficient
    """
    # Sadece ilgili segmenti filtrele
    segment_df = df[df['segment'] == segment].copy()
    
    # Her bayi-ay kombinasyonu için ürün satışlarını al
    transactions = segment_df.groupby(['cairkod', 'yil', 'aylik'])['StockAd'].apply(list).reset_index()
    transactions['segment'] = segment
    
    # Ürün gruplarını da ekle (sadece bu segment için)
    product_groups = segment_df.groupby('StockAd')['Ürün Grubu'].first().to_dict()
    
    # Memory temizliği
    del segment_df
    gc.collect()
    
    return transactions, product_groups

def run_apriori_for_single_segment(transactions, product_groups, segment, min_support=0.03, min_confidence=0.5):
    """
    Tek bir segment için Apriori algoritmasını çalıştırır - Memory efficient
    """
    print(f"\n{segment} segmenti için Apriori algoritması çalıştırılıyor...")
    
    if len(transactions) < 10:  # Minimum transaction sayısı
        print(f"{segment} segmenti için yeterli veri yok (min 10 transaction gerekli)")
        return []
    
    # Transaction encoder
    te = TransactionEncoder()
    product_lists = transactions['StockAd'].tolist()
    
    try:
        te_ary = te.fit(product_lists).transform(product_lists)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Memory temizliği
        del product_lists
        gc.collect()
        
        # Apriori algoritması
        frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True, low_memory=True)
        
        # Memory temizliği
        del df_encoded
        gc.collect()
        
        if len(frequent_itemsets) == 0:
            print(f"{segment} segmenti için frequent itemsets bulunamadı")
            return []
        
        # Association rules
        rules = association_rules(frequent_itemsets, 
                                metric="confidence", 
                                min_threshold=min_confidence)
        
        # Memory temizliği
        del frequent_itemsets
        gc.collect()
        
        if len(rules) == 0:
            print(f"{segment} segmenti için rules bulunamadı")
            return []
        
        # Aynı ürün grubu içindeki kuralları filtrele
        filtered_rules = []
        for _, rule in rules.iterrows():
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
        
        print(f"{segment} segmenti için {len(filtered_rules)} kural bulundu")
        
        # Memory temizliği
        del rules
        gc.collect()
        
        return filtered_rules
        
    except Exception as e:
        print(f"{segment} segmenti için hata: {e}")
        return []

def find_recommendation_opportunities(df, rules_df):
    """
    Son ay antecedent ürünü satıp consequent ürünü satmayan bayileri tespit eder
    """
    if rules_df.empty:
        return pd.DataFrame()
    
    # Son ay verilerini al
    latest_month = df['aylik'].max()
    latest_year = df['yil'].max()
    
    last_month_sales = df[(df['aylik'] == latest_month) & (df['yil'] == latest_year)]
    
    recommendations = []
    
    for _, rule in rules_df.iterrows():
        segment = rule['segment']
        antecedents = rule['antecedents'].split(', ')
        consequents = rule['consequents'].split(', ')
        
        # Bu segment için bayileri al
        segment_distributors = df[df['segment'] == segment]['cairkod'].unique()
        
        for distributor in segment_distributors:
            # Bu bayinin son ay satışları
            distributor_sales = last_month_sales[last_month_sales['cairkod'] == distributor]['StockAd'].tolist()
            
            # Antecedent ürünü satmış mı?
            has_antecedent = any(ant in distributor_sales for ant in antecedents)
            
            # Consequent ürünü satmış mı?
            has_consequent = any(con in distributor_sales for con in consequents)
            
            # Antecedent var, consequent yok -> öneri fırsatı
            if has_antecedent and not has_consequent:
                recommendations.append({
                    'cairkod': distributor,
                    'segment': segment,
                    'antecedents': rule['antecedents'],
                    'consequents': rule['consequents'],
                    'confidence': rule['confidence'],
                    'lift': rule['lift'],
                    'support': rule['support'],
                    'antecedent_groups': rule['antecedent_groups'],
                    'consequent_groups': rule['consequent_groups']
                })
    
    return pd.DataFrame(recommendations)

def add_sales_units(recommendations_df, df):
    """
    Önerilen ürünler için satış birimlerini ekler
    """
    if recommendations_df.empty:
        return recommendations_df
    
    # Ürün bilgilerini al
    product_info = df.groupby('StockAd').agg({
        'deger': ['sum', 'mean', 'count'],
        'Ürün Grubu': 'first'
    }).reset_index()
    
    # Sütun isimlerini düzenle
    product_info.columns = ['StockAd', 'total_sales', 'avg_sales', 'sales_count', 'product_group']
    
    # Öneriler için satış bilgilerini ekle
    recommendations_with_units = []
    
    for _, rec in recommendations_df.iterrows():
        consequents = rec['consequents'].split(', ')
        
        for consequent in consequents:
            product_sales = product_info[product_info['StockAd'] == consequent]
            
            if not product_sales.empty:
                unit_info = product_sales.iloc[0]
                
                rec_copy = rec.copy()
                rec_copy['onerilen_urun'] = consequent
                rec_copy['toplam_satis'] = unit_info['total_sales']
                rec_copy['ortalama_satis'] = unit_info['avg_sales']
                rec_copy['satis_adedi'] = unit_info['sales_count']
                rec_copy['urun_grubu'] = unit_info['product_group']
                
                recommendations_with_units.append(rec_copy)
    
    return pd.DataFrame(recommendations_with_units)

def save_results_to_excel(recommendations_df, rules_df, output_path):
    """
    Sonuçları Excel dosyasına kaydeder
    """
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Öneriler
            if not recommendations_df.empty:
                recommendations_df.to_excel(writer, sheet_name='Ürün Önerileri', index=False)
            
            # Kurallar
            if not rules_df.empty:
                rules_df.to_excel(writer, sheet_name='Association Rules', index=False)
            
        print(f"Sonuçlar {output_path} dosyasına kaydedildi")
        return True
    except Exception as e:
        print(f"Excel kaydetme hatası: {e}")
        return False

def main():
    """
    Ana fonksiyon
    """
    # Dosya yolu
    file_path = "/Users/esranuryigit/Desktop/BirliktelikAnalizi_son.xlsx"
    output_path = "/Users/esranuryigit/Desktop/urun_onerileri_sonuc.xlsx"
    
    print("Ürün Öneri Sistemi - Apriori Algoritması")
    print("=" * 50)
    
    # Veriyi yükle
    df = load_and_prepare_data(file_path)
    if df is None:
        return
    
    # Veri özeti
    print(f"\nVeri özeti:")
    print(f"Toplam satış kaydı: {len(df)}")
    print(f"Bayi sayısı: {df['cairkod'].nunique()}")
    print(f"Ürün sayısı: {df['StockAd'].nunique()}")
    print(f"Segment sayısı: {df['segment'].nunique()}")
    print(f"Ürün grubu sayısı: {df['Ürün Grubu'].nunique()}")
    
    # Segment listesini al
    segments = df['segment'].unique()
    print(f"\nSegmentler: {segments}")
    
    # Her segment için Apriori algoritmasını çalıştır - Memory efficient
    print("\nApriori algoritması segment segment çalıştırılıyor...")
    all_rules = []
    
    for i, segment in enumerate(segments, 1):
        print(f"\n--- {segment} segmenti işleniyor ({i}/{len(segments)}) ---")
        
        # Bu segment için transaction matrix oluştur
        transactions, product_groups = create_transaction_matrix_by_segment(df, segment)
        print(f"{segment} segmenti için {len(transactions)} transaction hazırlandı")
        
        # Bu segment için Apriori çalıştır
        segment_rules = run_apriori_for_single_segment(transactions, product_groups, segment,
                                                      min_support=0.03, min_confidence=0.5)
        
        all_rules.extend(segment_rules)
        
        # Memory temizliği
        del transactions, product_groups, segment_rules
        gc.collect()
        
        print(f"{segment} segmenti tamamlandı. Memory temizlendi. ({i}/{len(segments)} segment işlendi)")
        print(f"Şimdiye kadar toplam {len(all_rules)} kural bulundu.")
    
    rules_df = pd.DataFrame(all_rules)
    
    print(f"\nToplam {len(rules_df)} kural bulundu")
    
    # Öneri fırsatlarını bul
    print("\nÖneri fırsatları araştırılıyor...")
    if not rules_df.empty:
        recommendations_df = find_recommendation_opportunities(df, rules_df)
        print(f"Toplam {len(recommendations_df)} öneri fırsatı bulundu")
        
        # Satış birimlerini ekle
        if not recommendations_df.empty:
            print("\nSatış birimleri ekleniyor...")
            recommendations_df = add_sales_units(recommendations_df, df)
        
        # Memory temizliği
        gc.collect()
    else:
        print("Hiç kural bulunamadı, öneri analizi yapılamayacak.")
        recommendations_df = pd.DataFrame()
    
    # Sonuçları Excel'e kaydet
    print("\nSonuçlar Excel dosyasına kaydediliyor...")
    save_results_to_excel(recommendations_df, rules_df, output_path)
    
    # Özet rapor
    print("\n" + "=" * 50)
    print("ÖZET RAPOR")
    print("=" * 50)
    
    if not recommendations_df.empty:
        print(f"Toplam öneri: {len(recommendations_df)}")
        print(f"Öneri alan bayi sayısı: {recommendations_df['cairkod'].nunique()}")
        
        # Segment bazında özetler
        segment_summary = recommendations_df.groupby('segment').agg({
            'cairkod': 'count',
            'confidence': 'mean',
            'lift': 'mean'
        }).round(2)
        
        print("\nSegment bazında öneriler:")
        print(segment_summary)
        
        # En yüksek confidence'a sahip öneriler
        top_recommendations = recommendations_df.nlargest(10, 'confidence')[
            ['cairkod', 'segment', 'onerilen_urun', 'confidence', 'lift']
        ]
        
        print("\nEn yüksek confidence'a sahip 10 öneri:")
        print(top_recommendations)
    
    else:
        print("Hiç öneri bulunamadı. Parametreleri kontrol edin.")
    
    print(f"\nDetaylı sonuçlar {output_path} dosyasında bulunabilir.")

if __name__ == "__main__":
    main()