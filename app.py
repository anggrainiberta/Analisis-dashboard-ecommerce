import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="E-Commerce Dashboard", page_icon="🛒", layout="wide")
st.title("🛒 E-Commerce Analytics Dashboard")
st.markdown("---")

BASE_URL = "https://raw.githubusercontent.com/angeliayuflih/dashboard-ecommerce/main/ecommerce"

@st.cache_data
def load_data():
    df_master   = pd.read_csv(f"{BASE_URL}/main_data.csv")
    df_event    = pd.read_csv(f"{BASE_URL}/customer_behavior.csv")
    customer_df = pd.read_csv(f"{BASE_URL}/customer_segment.csv")
    category_df = pd.read_csv(f"{BASE_URL}/category_segment.csv")
    return df_master, df_event, customer_df, category_df

with st.spinner("Memuat data..."):
    df_master, df_event, customer_df, category_df = load_data()

df_master['transaction_time'] = pd.to_datetime(df_master['transaction_time'], errors='coerce')

# ── Sidebar Filter ────────────────────────────────────────────────────────────
st.sidebar.header("🔍 Filter Data")

# Filter Tahun
tahun_tersedia = sorted(df_master['transaction_time'].dt.year.dropna().unique().astype(int))
tahun_dipilih = st.sidebar.multiselect(
    "Pilih Tahun:",
    options=tahun_tersedia,
    default=tahun_tersedia
)

# Filter Gender
gender_tersedia = df_master['gender'].dropna().unique().tolist()
gender_dipilih = st.sidebar.multiselect(
    "Pilih Gender:",
    options=gender_tersedia,
    default=gender_tersedia
)

# Filter Negara (top 10)
top_negara = df_master['country'].value_counts().head(10).index.tolist()
negara_dipilih = st.sidebar.multiselect(
    "Pilih Negara (Top 10):",
    options=top_negara,
    default=top_negara
)

# Terapkan filter
if tahun_dipilih:
    df_filtered = df_master[
        (df_master['transaction_time'].dt.year.isin(tahun_dipilih)) &
        (df_master['gender'].isin(gender_dipilih)) &
        (df_master['country'].isin(negara_dipilih))
    ]
else:
    df_filtered = df_master.copy()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Total data:** {len(df_filtered):,} transaksi")

st.success(f"✅ Menampilkan {len(df_filtered):,} transaksi | Tahun: {', '.join(map(str, tahun_dipilih)) if tahun_dipilih else 'Semua'}")

tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "📈 Tren Revenue", "👥 Segmentasi Pelanggan", "🏷️ Segmentasi Kategori"])

# ── TAB 1 EDA ────────────────────────────────────────────────────────────────
with tab1:
    st.header("Exploratory Data Analysis")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transaksi", f"{len(df_filtered):,}")
    col2.metric("Total Revenue",   f"${df_filtered['sale_price'].sum():,.0f}")
    col3.metric("Rata-rata Harga", f"${df_filtered['sale_price'].mean():,.2f}")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Distribusi Usia Pelanggan")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df_filtered['age'].dropna(), bins=20, ax=ax)
        ax.set_xlabel("Usia"); ax.set_ylabel("Jumlah")
        st.pyplot(fig); plt.close(fig)

    with col_b:
        st.subheader("Distribusi Gender Pelanggan")
        gender = df_filtered['gender'].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(gender.values, labels=gender.index, autopct='%1.1f%%')
        st.pyplot(fig); plt.close(fig)

    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Top 10 Negara Pelanggan")
        country = df_filtered['country'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=country.index, y=country.values, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig); plt.close(fig)

    with col_d:
        st.subheader("Top 10 Kategori Produk")
        category = df_filtered['category'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=category.index, y=category.values, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig); plt.close(fig)

    st.subheader("Rata-rata Nilai Transaksi per Traffic Source")
    traffic_df = df_filtered.groupby('traffic_source')['sale_price'].mean().reset_index().sort_values('sale_price', ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(traffic_df['traffic_source'], traffic_df['sale_price'])
    ax.set_xlabel("Rata-rata Nilai Transaksi")
    st.pyplot(fig); plt.close(fig)

    if 'event_type' in df_event.columns:
        st.markdown("---")
        st.subheader("Aktivitas Pengguna")
        event = df_event['event_type'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(event.index, event.values)
        ax.set_xlabel("Jenis Aktivitas"); ax.set_ylabel("Jumlah")
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig); plt.close(fig)

# ── TAB 2 TREN REVENUE ───────────────────────────────────────────────────────
with tab2:
    st.header("Tren Revenue Harian")

    daily = (
        df_filtered.groupby(df_filtered['transaction_time'].dt.date)
        .agg(total_revenue=('sale_price', 'sum'), total_transaction=('sale_price', 'count'))
        .reset_index().rename(columns={'transaction_time': 'date'})
    )
    daily['moving_avg'] = daily['total_revenue'].rolling(window=7).mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily['date'], daily['total_revenue'], label='Revenue Harian', alpha=0.6)
    ax.plot(daily['date'], daily['moving_avg'], label='Moving Avg (7 hari)', linewidth=2)
    ax.legend(); ax.set_xlabel("Tanggal"); ax.set_ylabel("Revenue")
    plt.xticks(rotation=45)
    st.pyplot(fig); plt.close(fig)

    col1, col2 = st.columns(2)
    col1.metric("Total Revenue",   f"${daily['total_revenue'].sum():,.0f}")
    col2.metric("Total Transaksi", f"{daily['total_transaction'].sum():,}")

    # Tren per bulan
    st.subheader("Tren Revenue per Bulan")
    df_filtered2 = df_filtered.copy()
    df_filtered2['bulan'] = df_filtered2['transaction_time'].dt.to_period('M').astype(str)
    monthly = df_filtered2.groupby('bulan')['sale_price'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(monthly['bulan'], monthly['sale_price'])
    ax.set_xlabel("Bulan"); ax.set_ylabel("Revenue")
    ax.tick_params(axis='x', rotation=90)
    st.pyplot(fig); plt.close(fig)

# ── TAB 3 SEGMENTASI PELANGGAN ───────────────────────────────────────────────
with tab3:
    st.header("Segmentasi Pelanggan (K-Means)")

    fitur = customer_df[['total_transaction', 'total_spent']].dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(fitur)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
    cust = customer_df.loc[fitur.index].copy()
    cust['cluster'] = kmeans.fit_predict(scaled)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Visualisasi Cluster")
        fig, ax = plt.subplots(figsize=(6, 4))
        for c in sorted(cust['cluster'].unique()):
            sub = cust[cust['cluster'] == c]
            ax.scatter(sub['total_transaction'], sub['total_spent'], label=f"Cluster {c}", alpha=0.6)
        ax.set_xlabel("Total Transaksi"); ax.set_ylabel("Total Pengeluaran")
        ax.legend()
        st.pyplot(fig); plt.close(fig)

    with col_b:
        st.subheader("Rata-rata per Cluster")
        summary = cust.groupby('cluster')[['total_transaction', 'total_spent']].mean().round(2)
        st.dataframe(summary, use_container_width=True)

    st.markdown("---")
    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Proporsi Kelompok Umur")
        age_group = cust['age_group'].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(age_group.values, labels=age_group.index, autopct='%1.1f%%')
        st.pyplot(fig); plt.close(fig)

    with col_d:
        st.subheader("Total Transaksi per Kelompok Umur")
        age_tx = cust.groupby('age_group')['total_transaction'].sum().sort_values()
        fig, ax = plt.subplots(figsize=(6, 4)
)
        ax.barh(age_tx.index, age_tx.values)
        ax.set_xlabel("Total Transaksi")
        st.pyplot(fig); plt.close(fig)

    st.subheader("Distribusi Gender per Cluster")
    gender_cluster = cust.groupby(['cluster', 'gender']).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    gender_cluster.plot(kind='bar', ax=ax)
    ax.set_xlabel("Cluster"); ax.set_ylabel("Jumlah")
    ax.tick_params(axis='x', rotation=0)
    st.pyplot(fig); plt.close(fig)

# ── TAB 4 SEGMENTASI KATEGORI ────────────────────────────────────────────────
with tab4:
    st.header("Segmentasi Kategori Produk (K-Means)")

    cat = category_df.copy()
    cat['banyak_transaksi'] = cat.groupby('category')['product_name'].transform('count')

    fitur_cat = cat[['total_revenue_kategori', 'banyak_transaksi']].dropna()
    scaler2 = StandardScaler()
    scaled2 = scaler2.fit_transform(fitur_cat)
    kmeans2 = KMeans(n_clusters=4, random_state=42, n_init='auto')
    cat = cat.loc[fitur_cat.index].copy()
    cat['cluster'] = kmeans2.fit_predict(scaled2)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Top 10 Kategori berdasarkan Revenue")
        top_cat = category_df.groupby('category')['total_revenue_kategori'].sum().sort_values().tail(10)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(top_cat.index, top_cat.values)
        ax.set_xlabel("Total Revenue")
        st.pyplot(fig); plt.close(fig)

    with col_b:
        st.subheader("Visualisasi Cluster Kategori")
        fig, ax = plt.subplots(figsize=(6, 4))
        for c in sorted(cat['cluster'].unique()):
            sub = cat[cat['cluster'] == c]
            ax.scatter(sub['banyak_transaksi'], sub['total_revenue_kategori'], label=f"Cluster {c}", alpha=0.6)
        ax.set_xlabel("Jumlah Transaksi"); ax.set_ylabel("Total Revenue")
        ax.legend()
        st.pyplot(fig); plt.close(fig)

    st.subheader("Top 10 Produk berdasarkan Revenue")
    top_prod = category_df.groupby('product_name')['total_revenue_kategori'].sum().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(top_prod.index, top_prod.values)
    ax.set_xticklabels(top_prod.index, rotation=90)
    ax.set_ylabel("Total Revenue")
    st.pyplot(fig); plt.close(fig)

    st.subheader("Rata-rata per Cluster Kategori")
    summary2 = cat.groupby('cluster')[['total_revenue_kategori', 'banyak_transaksi']].mean().round(2)
    st.dataframe(summary2, use_container_width=True)
