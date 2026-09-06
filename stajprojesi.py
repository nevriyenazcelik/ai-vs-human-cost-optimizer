# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np

# Kod her çalıştığında aynı rastgele sayıları üretsin diye bir seed sabitleyelim
#aynı düzende çalışması için bir anahtar kelime veya şifre gibi düşün.
np.random.seed(42)

# 1. Simüle edilecek şirket görevi sayısı (Örn: 200 farklı iş kalemi)
n_samples = 200

# 2. Görev kimlikleri ve temel özellikler
task_ids = [f"TASK_{i:03d}" for i in range(1, n_samples + 1)]

# Görevin karmaşıklığı (1: Çok basit veri girişi, 10: Ağır analiz/raporlama)
complexity = np.random.randint(1, 11, size=n_samples)

# İşlenecek veya üretilecek tahmini kelime/token sayısı (Yapısal veri büyüklüğü)
token_count = np.random.randint(200, 30000, size=n_samples)

# 3. İNSAN MALİYET MODELİ
# İnsanın işi bitirme süresi (saat): Karmaşıklık ve token sayısına bağlı bir matematiksel fonksiyon
# Araya biraz da gerçekçi insani sapmalar (rastgele gürültü) ekleyelim
human_hours = (complexity * 1.8) + (token_count / 1500) + np.random.normal(0, 1, size=n_samples)
human_hours = np.clip(human_hours, 0.5, 50)  # Süreleri mantıklı sınırlarda (30 dk ile 50 saat arası) tutalım

# Şirketin çalışanına ödediği ortalama saatlik ücret masrafı (Örn: saatlik 30$)
hourly_wage = 30
human_cost = human_hours * hourly_wage

# 4. YAPAY ZEKA (AI) ENERJİ VE MALİYET MODELİ
# Büyük dil modellerinin harcadığı enerji donanıma göre değişir.
# Token sayısı ve karmaşıklık arttıkça GPU'nun harcadığı elektrik (kWh) artar.
ai_energy_kwh = (token_count * 0.00004) * (1 + (complexity / 8))

# Elektrik birim fiyatı (Örn: kWh başına 0.15$) + Bulut sunucu (Azure/AWS GPU) kullanım/API maliyeti
electricity_price_kwh = 0.15
gpu_api_cost_per_token = 0.000015  # Büyük modellerin token başı maliyet simülasyonu

ai_cost = (ai_energy_kwh * electricity_price_kwh) + (token_count * gpu_api_cost_per_token)

# 5. Tüm bu verileri bir araya getirip Pandas Veri Çerçevesine (DataFrame) dönüştürelim
df = pd.DataFrame({
    'Task_ID': task_ids,
    'Complexity': complexity,
    'Token_Count': token_count,
    'Human_Hours': np.round(human_hours, 2),
    'Human_Cost_USD': np.round(human_cost, 2),
    'AI_Energy_kWh': np.round(ai_energy_kwh, 4),
    'AI_Cost_USD': np.round(ai_cost, 4)
})

# Oluşan veri setinin ilk 10 satırını ekranda görelim
print("--- GreenAI-Optimizer Projesi İlk Veri Seti (İlk 10 Görev) ---")
print(df.head(10))

import matplotlib.pyplot as plt

# Grafiğin boyutunu ayarlayalım
plt.figure(figsize=(12, 6))

# İnsan maliyetini mavi noktalarla çizdirelim
plt.scatter(df['Complexity'], df['Human_Cost_USD'], color='blue', label='İnsan Emeği Maliyeti ($)', alpha=0.6)

# Yapay Zeka maliyetini yeşil noktalarla çizdirelim
plt.scatter(df['Complexity'], df['AI_Cost_USD'], color='green', label='Yapay Zeka Maliyeti ($)', alpha=0.6)

# Grafiği isimlendirip süsleyelim
plt.title('Görev Karmaşıklığına Göre Maliyet Karşılaştırması (İnsan vs. Yapay Zeka)', fontsize=14)
plt.xlabel('İşin Karmaşıklık Seviyesi (1 - 10)', fontsize=12)
plt.ylabel('Tahmini Maliyet (USD)', fontsize=12)

# Lejant (açıklama kutusu) ve ızgara ekleyelim
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Grafiği ekranda gösterelim
plt.show()

# İnsanın maliyeti ile AI maliyeti arasındaki farkı bulup yeni bir sütun yapalım
df['Profit_USD'] = df['Human_Cost_USD'] - df['AI_Cost_USD']

# Şirkete en çok kâr ettiren (farkın en büyük olduğu) ilk 5 görevi listeleyelim
top_5_ai_tasks = df.sort_values(by='Profit_USD', ascending=False).head(5)

print("\n--- Yapay Zekaya Yaptırılması En Mantıklı Olan İlk 5 Görev ---")
print(top_5_ai_tasks[['Task_ID', 'Complexity', 'Token_Count', 'Profit_USD']])

# 1. Görevleri yönlendirecek akıllı bir fonksiyon tanımlayalım
def akilli_yonlendirici(row):
    # Kural 1: İşin karmaşıklığı 9 veya daha üzerindeyse, risk almayıp İNSANA verelim
    if row['Complexity'] >= 9:
        return 'İnsan (Kritik Görev)'
    
    # Kural 2: Diğer durumlarda maliyet kontrolü yapalım
    elif row['AI_Cost_USD'] < row['Human_Cost_USD']:
        return 'Yapay Zeka'
    
    else:
        return 'İnsan (Ekonomik)'

# 2. Yazdığımız bu fonksiyonu tablodaki (DataFrame) tüm satırlara tek tek uygulayalım
df['Sistem_Karari'] = df.apply(akilli_yonlendirici, axis=1)

# 3. Toplam 200 işten kaç tanesi nereye yönlendirilmiş, özetini görelim
print("\n--- Hibrit Sistemin Görev Dağılım Özeti ---")
print(df['Sistem_Karari'].value_counts())

# 4. Sistemin karar verdiği yeni tablodan ilk 5 satıra göz atalım
print("\n--- Yeni Karar Sütunlu Tablo Önizlemesi ---")
print(df[['Task_ID', 'Complexity', 'Human_Cost_USD', 'AI_Cost_USD', 'Sistem_Karari']].head(5))


import numpy as np

# 1. Finansal Tasarruf Analizi
toplam_sadece_insan = df['Human_Cost_USD'].sum()

# Hibrit Maliyet: Yapay Zeka'ya verilenler AI maliyeti, İnsana verilenler İnsan maliyeti çıkarır
df['Nihai_Maliyet'] = np.where(df['Sistem_Karari'] == 'Yapay Zeka', df['AI_Cost_USD'], df['Human_Cost_USD'])
toplam_hibrit_maliyet = df['Nihai_Maliyet'].sum()

tasarruf = toplam_sadece_insan - toplam_hibrit_maliyet
tasarruf_yuzdesi = (tasarruf / toplam_sadece_insan) * 100

# 2. Konsola Finansal Rapor Yazdıralım
print("\n================ FINANSAL ETKI RAPORU ================")
print(f"Geleneksel Sistem (Sadece İnsan) Maliyeti : ${toplam_sadece_insan:,.2f}")
print(f"Yeni Hibrit Sistem Maliyeti                : ${toplam_hibrit_maliyet:,.2f}")
print(f"Şirketin Net Tasarrufu                    : ${tasarruf:,.2f}")
print(f"Toplam Tasarruf Oranı                     : %{tasarruf_yuzdesi:.1f}")
print("======================================================")

# 3. Karar Dağılımının Pasta Grafiğini Çizdirelim
plt.figure(figsize=(7, 7))
karar_sayilari = df['Sistem_Karari'].value_counts()
plt.pie(karar_sayilari, labels=karar_sayilari.index, autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], startangle=90)
plt.title('Görevlerin Yönlendirme Dağılımı (% Hissesi)', fontsize=13)
plt.show()


# ============================================================
# 5. AŞAMA: MAKİNE ÖĞRENMESİ (Pure NumPy Classifier)
# ============================================================

# 1. Veriyi %80 Eğitim (Train) ve %20 Test olarak ayıralım
np.random.seed(42)
karistirilmis_indeksler = np.random.permutation(len(df))
test_boyutu = int(len(df) * 0.2)

test_indeksleri = karistirilmis_indeksler[:test_boyutu]
train_indeksleri = karistirilmis_indeksler[test_boyutu:]

train_df = df.iloc[train_indeksleri].copy()
test_df = df.iloc[test_indeksleri].copy()

# 2. En Yakın Komşu (KNN) Tabanlı Tahmin Fonksiyonu
def ml_model_tahmin_et(test_satiri, egitim_verisi):
    # Karmaşıklık ve Token farklarını ölçekleyip öklid uzaklığı hesaplayalım
    z_fark = (egitim_verisi['Complexity'] - test_satiri['Complexity']) ** 2
    t_fark = ((egitim_verisi['Token_Count'] - test_satiri['Token_Count']) / 30000) ** 2
    
    uzakliklar = np.sqrt(z_fark + t_fark)
    en_yakin_indeks = uzakliklar.idxmin()
    return egitim_verisi.loc[en_yakin_indeks, 'Sistem_Karari']

# 3. Test verisi üzerindeki görevleri modele tahmin ettirelim
tahminler = [ml_model_tahmin_et(row, train_df) for _, row in test_df.iterrows()]
gercek_degerler = test_df['Sistem_Karari'].values

# 4. Başarı Oranını (Accuracy) Hesaplayalım
dogru_sayisi = sum(1 for t, g in zip(tahminler, gercek_degerler) if t == g)
basari_orani = (dogru_sayisi / len(test_df)) * 100

print("\n================ MAKİNE ÖĞRENMESİ SONUÇLARI ================")
print(f"Yapay Zeka Modelinin Karar Verme Başarısı: %{basari_orani:.1f}")
print("============================================================")

# ============================================================
# 6. AŞAMA: CANLI İŞ TAHMİNCİSİ & CSV KAYDI
# ============================================================

# 1. canlı göstermek için yeni, bilinmeyen bir görev tanımlayalım
# Örn: Zorluğu 8 olan ve 18.000 kelimelik bir iş geldi
yeni_gorev = {'Complexity': 8, 'Token_Count': 18000}

# 2. Sıfırdan yazdığımız ML modeline bu yeni görevi yönlendirtelim
tahmin_karari = ml_model_tahmin_et(yeni_gorev, train_df)

print("\n================ CANLI İŞ TAHMİN BÖLÜMÜ ================")
print(f"Sisteme Gelen Yeni Görev -> Zorluk: {yeni_gorev['Complexity']}/10 | Kelime Sayısı: {yeni_gorev['Token_Count']}")
print(f"Yapay Zeka Modelinin Anlık Kararı: >>> {tahmin_karari} <<<")
print("==========================================================")


df.to_csv("yonlendirilmis_staj_verisi.csv", index=False)
print("\n[BİLGİ] Bütün tablo 'yonlendirilmis_staj_verisi.csv' adıyla klasörüne kaydedildi!")