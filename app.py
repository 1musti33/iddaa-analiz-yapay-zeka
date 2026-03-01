import streamlit as st
import requests
import pandas as pd
import math
from sklearn.neighbors import NearestNeighbors

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Pro İddaa Analiz", page_icon="⚽", layout="wide")
st.title("⚽ Yapay Zeka Destekli Profesyonel Maç Analizi")
st.markdown("API'den canlı veri çekerek takımların istatistiklerini hesaplar, **Poisson Dağılımı** ile tahmin üretir ve **KNN Algoritması** ile geçmişteki en benzer maçları bulur.")

# --- ARKA PLAN FONKSİYONLARI ---
@st.cache_data(show_spinner=False)
def veri_cek(api_key, lig_id, sezon):
    url = "https://v3.football.api-sports.io/fixtures"
    querystring = {"league": str(lig_id), "season": str(sezon)}
    headers = {"x-apisports-key": api_key, "x-apisports-host": "v3.football.api-sports.io"}
    
    response = requests.get(url, headers=headers, params=querystring)
    data = response.json()
    
    # 1. DEDEKTİF: API direkt hata mesajı gönderdiyse yakala
    if "errors" in data and data["errors"]:
        # errors bazen liste, bazen sözlük (dictionary) olarak gelebiliyor
        st.sidebar.error(f"API Hatası: {data['errors']}")
        return None
        
    maclar = []
    # 2. DEDEKTİF: Veri var mı kontrol et
    if "response" in data and len(data["response"]) > 0:
        for mac in data["response"]:
            if mac["fixture"]["status"]["short"] == "FT":
                maclar.append({
                    "Ev Sahibi": mac["teams"]["home"]["name"],
                    "Deplasman": mac["teams"]["away"]["name"],
                    "Ev Gol": mac["goals"]["home"],
                    "Dep Gol": mac["goals"]["away"],
                    "Skor": f"{mac['goals']['home']}-{mac['goals']['away']}"
                })
        
        df = pd.DataFrame(maclar)
        # 3. DEDEKTİF: Veri geldi ama hepsi oynanmamış maçlar mı?
        if df.empty:
            st.sidebar.warning("API veri gönderdi ancak bu sezonda 'Bitmiş Maç (FT)' bulunamadı. Sezon henüz başlamamış olabilir.")
        return df
    else:
        st.sidebar.warning("API bu lig ve sezon için HİÇ veri bulamadı. (Sezon yılını veya ligi kontrol edin)")
        return None

def poisson_hesapla(beklenen_gol, atilacak_gol):
    return ((beklenen_gol ** atilacak_gol) * math.exp(-beklenen_gol)) / math.factorial(atilacak_gol)
@st.cache_data(show_spinner=False)
def puan_durumu_cek(api_key, lig_id, sezon):
    url = "https://v3.football.api-sports.io/standings"
    querystring = {"league": str(lig_id), "season": str(sezon)}
    headers = {"x-apisports-key": api_key, "x-apisports-host": "v3.football.api-sports.io"}
    
    response = requests.get(url, headers=headers, params=querystring)
    data = response.json()
    
    if "response" in data and len(data["response"]) > 0:
        # API'den gelen karmaşık verinin içinden puan tablosunu buluyoruz
        standings = data["response"][0]["league"]["standings"][0]
        tablo = []
        for takim in standings:
            tablo.append({
                "Sıra": takim["rank"],
                "Takım": takim["team"]["name"],
                "O": takim["all"]["played"],
                "G": takim["all"]["win"],
                "B": takim["all"]["draw"],
                "M": takim["all"]["lose"],
                "AG": takim["all"]["goals"]["for"],
                "YG": takim["all"]["goals"]["against"],
                "Av": takim["goalsDiff"],
                "Puan": takim["points"]
            })
        return pd.DataFrame(tablo)
    return None

# --- SOL MENÜ (API VE VERİ GİRİŞİ) ---
st.sidebar.header("1. Veri Bağlantısı")
api_key_input = st.sidebar.text_input("API-Football Anahtarı", type="password", help="API-Sports üzerinden aldığınız ücretsiz anahtar.")
# YENİ EKLENEN KISIM:
# Ligleri ve ID numaralarını bir sözlük (dictionary) içinde tanımlıyoruz
ligler = {
    "🇹🇷 Türkiye Süper Lig": 203,
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 İngiltere Premier Lig": 39,
    "🇪🇸 İspanya La Liga": 140,
    "🇮🇹 İtalya Serie A": 135,
    "🇩🇪 Almanya Bundesliga": 78,
    "🇫🇷 Fransa Ligue 1": 61,
    "🌍 UEFA Şampiyonlar Ligi": 2
}

# Kullanıcıya sadece isimleri gösteriyoruz
secilen_lig_adi = st.sidebar.selectbox("Lig Seçin", list(ligler.keys()))

# Arka planda API'ye göndermek için seçilen ligin ID'sini alıyoruz
lig_id = ligler[secilen_lig_adi]

# Sezonu güncelleyelim (Kullanıcı 2024 veya 2025 yazabilir)
sezon_secimi = st.sidebar.text_input("Sezon Yılı (Örn: 2023, 2024, 2025)", "2025")

if st.sidebar.button("Verileri Çek ve Hazırla"):
    if not api_key_input:
        st.sidebar.error("Lütfen API Anahtarınızı girin!")
    else:
        with st.spinner("Veriler API'den çekiliyor ve yapay zeka eğitiliyor..."):
            df = veri_cek(api_key_input, lig_id, sezon_secimi)
            if df is not None and not df.empty:
                st.session_state['veri'] = df
                st.sidebar.success(f"Başarılı! {len(df)} maç çekildi.")
            else:
                st.sidebar.error("Veri çekilemedi. API anahtarınızı veya limitinizi kontrol edin.")

# --- ANA EKRAN (ANALİZ VE TAHMİN) ---
if 'veri' in st.session_state:
    df = st.session_state['veri']
    
    # Takımların gol ortalamalarını (Beklenen Gol - xG) hesaplama
    ev_ortalamalari = df.groupby('Ev Sahibi')['Ev Gol'].mean().reset_index()
    dep_ortalamalari = df.groupby('Deplasman')['Dep Gol'].mean().reset_index()
    
    # KNN Algoritması için veriyi hazırlama
    knn_verisi = df.copy()
    knn_verisi = knn_verisi.merge(ev_ortalamalari, on='Ev Sahibi', suffixes=('', '_Ort'))
    knn_verisi = knn_verisi.merge(dep_ortalamalari, on='Deplasman', suffixes=('', '_Ort'))
    
    st.divider()

# --- PUAN DURUMU TABLOSU (GİZLENEBİLİR PENCERE) ---
    puan_tablosu = puan_durumu_cek(api_key_input, lig_id, sezon_secimi)
    if puan_tablosu is not None:
        # st.expander sayesinde tabloyu açılır/kapanır bir kutu içine alıyoruz ki ekranı çok kaplamasın
        with st.expander(f"🏆 {secilen_lig_adi} - Güncel Puan Durumu"):
            # hide_index=True ile baştaki gereksiz sıra numaralarını siliyoruz
            st.dataframe(puan_tablosu, use_container_width=True, hide_index=True)

    st.subheader("2. Maç Seçimi")
    
    col1, col2 = st.columns(2)
    takimlar = sorted(df['Ev Sahibi'].unique())
    
    with col1:
        secilen_ev = st.selectbox("Ev Sahibi Takım", takimlar, index=0)
    with col2:
        secilen_dep = st.selectbox("Deplasman Takımı", takimlar, index=1 if len(takimlar)>1 else 0)

    if st.button("Yapay Zeka Analizini Başlat", type="primary"):
        # Seçilen takımların ortalamalarını buluyoruz
        ev_beklenen = ev_ortalamalari[ev_ortalamalari['Ev Sahibi'] == secilen_ev]['Ev Gol'].values
        dep_beklenen = dep_ortalamalari[dep_ortalamalari['Deplasman'] == secilen_dep]['Dep Gol'].values
        
        if len(ev_beklenen) > 0 and len(dep_beklenen) > 0:
            ev_xg = float(ev_beklenen[0])
            dep_xg = float(dep_beklenen[0])
            
            st.info(f"**İstatistiksel Beklenti (xG):** {secilen_ev} ({ev_xg:.2f}) - {secilen_dep} ({dep_xg:.2f})")
            
            # --- POISSON TAHMİNLERİ ---
            ev_kazanma = beraberlik = dep_kazanma = ust_2_5 = kg_var = 0
            skor_ihtimalleri = {}
            
            for eg in range(6):
                for dg in range(6):
                    olasilik = poisson_hesapla(ev_xg, eg) * poisson_hesapla(dep_xg, dg)
                    skor_ihtimalleri[f"{eg}-{dg}"] = olasilik
                    if eg > dg: ev_kazanma += olasilik
                    elif eg == dg: beraberlik += olasilik
                    else: dep_kazanma += olasilik
                    
                    if (eg + dg) > 2.5: ust_2_5 += olasilik
                    if eg > 0 and dg > 0: kg_var += olasilik
            
            # --- İHTİMALLERİ ADİL ORANLARA ÇEVİRME MODÜLÜ ---
            # Sıfıra bölünme hatası almamak için küçük bir kontrol yapıyoruz
            oran_1 = 1 / ev_kazanma if ev_kazanma > 0 else 0
            oran_x = 1 / beraberlik if beraberlik > 0 else 0
            oran_2 = 1 / dep_kazanma if dep_kazanma > 0 else 0
            
            oran_ust = 1 / ust_2_5 if ust_2_5 > 0 else 0
            oran_kg = 1 / kg_var if kg_var > 0 else 0
            
            # --- EKRANA ŞIK BİR ŞEKİLDE YANSITMA ---
            st.subheader("📊 Maç Sonucu, İhtimal ve Adil Oran Analizi")
            c1, c2, c3 = st.columns(3)
            
            c1.metric("Ev Sahibi (1)", f"%{ev_kazanma*100:.1f}", f"Adil Oran: {oran_1:.2f}", delta_color="off")
            c2.metric("Beraberlik (X)", f"%{beraberlik*100:.1f}", f"Adil Oran: {oran_x:.2f}", delta_color="off")
            c3.metric("Deplasman (2)", f"%{dep_kazanma*100:.1f}", f"Adil Oran: {oran_2:.2f}", delta_color="off")
            
            st.divider() # Araya estetik bir çizgi çekiyoruz
            
            c4, c5 = st.columns(2)
            c4.metric("2.5 ÜST", f"%{ust_2_5*100:.1f}", f"Adil Oran: {oran_ust:.2f}", delta_color="off")
            c5.metric("Karşılıklı Gol VAR", f"%{kg_var*100:.1f}", f"Adil Oran: {oran_kg:.2f}", delta_color="off")
            
            # --- KNN BENZER MAÇLAR ---
            st.subheader("🔍 Algoritmanın Bulduğu En Benzer Geçmiş Maçlar")
            X = knn_verisi[['Ev Gol_Ort', 'Dep Gol_Ort']]
            # Uyarıyı önlemek için .values kullanıyoruz
            model = NearestNeighbors(n_neighbors=3).fit(X.values)
            mesafeler, indeksler = model.kneighbors([[ev_xg, dep_xg]])
            
            benzerler = knn_verisi.iloc[indeksler[0]][['Ev Sahibi', 'Deplasman', 'Skor']]
            st.dataframe(benzerler, use_container_width=True, hide_index=True)
            
        else:
            st.warning("Bu takımlar için yeterli veri bulunamadı.")
else:
    st.info("Lütfen önce sol menüden API anahtarınızı girip verileri çekin.")