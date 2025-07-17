import pandas as pd
import numpy as np

def analyze_product_groups(file_path):
    """
    Ürün gruplarını analiz eder
    """
    print("🔍 ÜRÜN GRUBU ANALİZİ")
    print("=" * 50)
    
    # Veriyi yükle
    df = pd.read_excel(file_path)
    print(f"📊 Toplam satır: {len(df):,}")
    print(f"📊 Sütunlar: {list(df.columns)}")
    
    # Sütun isimlerini standartlaştır
    expected_columns = ['carikod', 'yil', 'aylik', 'cirokod_aylik', 'ciro_degeri', 'StockAd','segment', 'StockId','StockKod','Ürün Grubu']
    
    if len(df.columns) >= len(expected_columns):
        df.columns = expected_columns[:len(df.columns)]
    
    print(f"\n📋 ÜRÜN GRUBU SÜTUNU ANALİZİ:")
    print(f"Sütun adı: 'Ürün Grubu'")
    print(f"Veri tipi: {df['Ürün Grubu'].dtype}")
    print(f"Benzersiz değer sayısı: {df['Ürün Grubu'].nunique()}")
    print(f"NaN sayısı: {df['Ürün Grubu'].isnull().sum()}")
    
    # Örnek değerler
    print(f"\n📝 ÖRNEK DEĞERLER:")
    unique_groups = df['Ürün Grubu'].dropna().unique()[:20]
    for i, group in enumerate(unique_groups, 1):
        print(f"{i:2d}. '{group}' (tip: {type(group)})")
    
    # Value counts
    print(f"\n📊 EN ÇOK KULLANILAN ÜRÜN GRUPLARI:")
    top_groups = df['Ürün Grubu'].value_counts().head(10)
    for group, count in top_groups.items():
        print(f"'{group}': {count:,} adet")
    
    # Segment 6 özel analizi
    print(f"\n🎯 SEGMENT 6 ÖZEL ANALİZİ:")
    segment6_df = df[df['segment'] == 'Segment 6']
    print(f"Segment 6 satır sayısı: {len(segment6_df):,}")
    print(f"Segment 6 benzersiz ürün sayısı: {segment6_df['StockAd'].nunique()}")
    print(f"Segment 6 benzersiz ürün grubu sayısı: {segment6_df['Ürün Grubu'].nunique()}")
    
    # Segment 6 ürün grupları
    seg6_groups = segment6_df['Ürün Grubu'].value_counts().head(10)
    print(f"\nSegment 6 ürün grupları:")
    for group, count in seg6_groups.items():
        print(f"'{group}': {count:,} adet")
    
    # Ürün grubu-ürün ilişkisi
    print(f"\n🔗 ÜRÜN GRUBU - ÜRÜN İLİŞKİSİ (Segment 6):")
    sample_products = segment6_df[['StockAd', 'Ürün Grubu']].drop_duplicates().head(10)
    for _, row in sample_products.iterrows():
        print(f"Ürün: '{row['StockAd'][:50]}...' -> Grup: '{row['Ürün Grubu']}'")

if __name__ == "__main__":
    file_path = "/Users/esranuryigit/Desktop/BirliktelikAnalizi_Filtresiz.xlsx"
    analyze_product_groups(file_path)