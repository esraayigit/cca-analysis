import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(file_path):
    """
    Excel dosyasını yükler ve veriyi hazırlar
    """
    try:
        df = pd.read_excel(file_path)
        print(f"Veri yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")
        print("Sütunlar:", df.columns.tolist())
        
        # Sütun isimlerini standartlaştır
        expected_columns = ['cairkod', 'yil', 'aylik', 'urokod_aylik', 'deger', 'StockAd', 'segment', 'StockId', 'stockkod', 'Ürün Grubu']
        
        if len(df.columns) >= len(expected_columns):
            df.columns = expected_columns[:len(df.columns)]
        
        return df
    except Exception as e:
        print(f"Veri yükleme hatası: {e}")
        return None

def create_transaction_matrix(df):
    """
    Her bayi için transaction matrix oluşturur
    """
    # Her bayi-ay kombinasyonu için ürün satışlarını al
    transactions = df.groupby(['cairkod', 'segment', 'yil', 'aylik'])['StockAd'].apply(list).reset_index()
    
    # Ürün gruplarını da ekle
    product_groups = df.groupby('StockAd')['Ürün Grubu'].first().to_dict()
    
    return transactions, product_groups

def run_apriori_by_segment(transactions, product_groups, min_support=0.03, min_confidence=0.5):
    """
    Her segment için Apriori algoritmasını çalıştırır
    """
    results = []
    
    segments = transactions['segment'].unique()
    
    for segment in segments:
        print(f"\n{segment} segmenti için Apriori algoritması çalıştırılıyor...")
        
        # Segment verilerini filtrele
        segment_data = transactions[transactions['segment'] == segment]
        
        if len(segment_data) < 10:  # Minimum transaction sayısı
            print(f"{segment} segmenti için yeterli veri yok (min 10 transaction gerekli)")
            continue
        
        # Transaction encoder
        te = TransactionEncoder()
        te_ary = te.fit(segment_data['StockAd'].tolist()).transform(segment_data['StockAd'].tolist())
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Apriori algoritması
        try:
            frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
            
            if len(frequent_itemsets) == 0:
                print(f"{segment} segmenti için frequent itemsets bulunamadı")
                continue
            
            # Association rules
            rules = association_rules(frequent_itemsets, 
                                    metric="confidence", 
                                    min_threshold=min_confidence)
            
            if len(rules) == 0:
                print(f"{segment} segmenti için rules bulunamadı")
                continue
            
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
            
            results.extend(filtered_rules)
            print(f"{segment} segmenti için {len(filtered_rules)} kural bulundu")
            
        except Exception as e:
            print(f"{segment} segmenti için hata: {e}")
            continue
    
    return pd.DataFrame(results)

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
    
    # Transaction matrix oluştur
    transactions, product_groups = create_transaction_matrix(df)
    print(f"\nTransaction matrix hazırlandı: {len(transactions)} transaction")
    
    # Apriori algoritmasını çalıştır
    print("\nApriori algoritması çalıştırılıyor...")
    rules_df = run_apriori_by_segment(transactions, product_groups, 
                                     min_support=0.03, min_confidence=0.5)
    
    print(f"\nToplam {len(rules_df)} kural bulundu")
    
    # Öneri fırsatlarını bul
    print("\nÖneri fırsatları araştırılıyor...")
    recommendations_df = find_recommendation_opportunities(df, rules_df)
    
    print(f"Toplam {len(recommendations_df)} öneri fırsatı bulundu")
    
    # Satış birimlerini ekle
    if not recommendations_df.empty:
        print("\nSatış birimleri ekleniyor...")
        recommendations_df = add_sales_units(recommendations_df, df)
    
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