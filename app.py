import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Dashboard",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 E-Commerce Analytics Dashboard")
st.markdown("---")

# ── Upload CSV ────────────────────────────────────────────────────────────────
st.sidebar.header("📂 Upload Data")
st.sidebar.markdown("Upload file CSV hasil export dari Colab:")

main_file     = st.sidebar.file_uploader("main_data.csv",          type="csv")
event_file    = st.sidebar.file_uploader("customer_behavior.csv",  type="csv")
customer_file = st.sidebar.file_uploader("customer_segment.csv",   type="csv")
category_file = st.sidebar.file_uploader("category_segment.csv",   type="csv")

# Helper: load or show warning
def load(f, name):
    if f:
        return pd.read_csv(f)
    st.warning(f"⬆️ Upload **{name}** di sidebar untuk melihat bagian ini.")
    return None

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 EDA",
    "📈 Tren Revenue",
    "👥 Segmentasi Pelanggan",
    "🏷️ Segmentasi Kategori"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Exploratory Data Analysis")

    df_master = load(main_file, "main_data.csv")
    df_event  = load(event_file, "customer_behavior.csv")

    if df_master is not None:
        df_master['transaction_time'] = pd.to_datetime(
            df_master['transaction_time'], errors='coerce'
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Transaksi",   f"{len(df_master):,}")
        col2.metric("Total Revenue",     f"${df_master['sale_price'].sum():,.0f}")
        col3.metric("Rata-rata Harga",   f"${df_master['sale_price'].mean():,.2f}")

        st.markdown("---")
        col_a, col_b = st.columns(2)

        # Distribusi Usia
        with col_a:
            st.subheader("Distribusi Usia Pelanggan")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df_master['age'].dropna(), bins=20, ax=ax)
            ax.set_xlabel("Usia"); ax.set_ylabel("Jumlah")
            st.pyplot(fig); plt.close(fig)

        # Distribusi Gender
        with col_b:
            st.subheader("Distribusi Gender Pelanggan")
            gender = df_master['gender'].value_counts()
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.pie(gender.values, labels=gender.index, autopct='%1.1f%%')
            st.pyplot(fig); plt.close(fig)

        col_c, col_d = st.columns(2)

        # Top 10 Negara
        with col_c:
            st.subheader("Top 10 Negara Pelanggan")
            country = df_master['country'].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x=country.index, y=country.values, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            st.pyplot(fig); plt.close(fig)

        # Top 10 Kategori
        with col_d:
            st.subheader("Top 10 Kategori Produk")
            category = df_master['category'].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x=category.index, y=category.values, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            st.pyplot(fig); plt.close(fig)

        # Rata-rata Transaksi per Traffic Source
        st.subheader("Rata-rata Nilai Transaksi per Traffic Source")
        traffic_df = (
            df_master.groupby('traffic_source')['sale_price']
            .mean().reset_index()
            .sort_values('sale_price', ascending=False)
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(traffic_df['traffic_source'], traffic_df['sale_price'])
        ax.set_xlabel("Rata-rata Nilai Transaksi")
        st.pyplot(fig); plt.close(fig)

    if df_event is not None:
        st.markdown("---")
        st.subheader("Aktivitas Pengguna")
        event = df_event['event_type'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(event.index, event.values)
        ax.set_xlabel("Jenis Aktivitas"); ax.set_ylabel("Jumlah")
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig); plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Tren Revenue
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Tren Revenue Harian")

    df_master2 = load(main_file, "main_data.csv")
    if df_master2 is not None:
        df_master2['transaction_time'] = pd.to_datetime(
            df_master2['transaction_time'], errors='coerce'
        )
        daily = (
            df_master2.groupby(df_master2['transaction_time'].dt.date)
            .agg(total_revenue=('sale_price', 'sum'),
                 total_transaction=('sale_price', 'count'))
            .reset_index()
            .rename(columns={'transaction_time': 'date'})
        )
        daily['moving_avg'] = daily['total_revenue'].rolling(window=7).mean()

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(daily['date'], daily['total_revenue'], label='Revenue Harian', alpha=0.6)
        ax.plot(daily['date'], daily['moving_avg'],    label='Moving Avg (7 hari)', linewidth=2)
        ax.legend(); ax.set_xlabel("Tanggal"); ax.set_ylabel("Revenue")
        plt.xticks(rotation=45)
        st.pyplot(fig); plt.close(fig)

        col1, col2 = st.columns(2)
        col1.metric("Total Revenue",      f"${daily['total_revenue'].sum():,.0f}")
        col2.metric("Total Transaksi",    f"{daily['total_transaction'].sum():,}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – Segmentasi Pelanggan
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Segmentasi Pelanggan (K-Means)")

    customer_df = load(customer_file, "customer_segment.csv")
    if customer_df is not None:

        # Re-run KMeans
        fitur = customer_df[['total_transaction_pelanggan', 'total_spent_pelanggan']].dropna()
        scaler = StandardScaler()
        scaled = scaler.fit_transform(fitur)
        kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
        customer_df = customer_df.loc[fitur.index].copy()
        customer_df['cluster'] = kmeans.fit_predict(scaled)

        col_a, col_b = st.columns(2)

        # Scatter Cluster
        with col_a:
            st.subheader("Visualisasi Cluster")
            fig, ax = plt.subplots(figsize=(6, 4))
            for c in sorted(customer_df['cluster'].unique()):
                sub = customer_df[customer_df['cluster'] == c]
                ax.scatter(
                    sub['total_transaction_pelanggan'],
                    sub['total_spent_pelanggan'],
                    label=f"Cluster {c}", alpha=0.6
                )
            ax.set_xlabel("Total Transaksi"); ax.set_ylabel("Total Pengeluaran")
            ax.legend()
            st.pyplot(fig); plt.close(fig)

        # Rata-rata per Cluster
        with col_b:
            st.subheader("Rata-rata per Cluster")
            summary = customer_df.groupby('cluster')[[
                'total_transaction_pelanggan', 'total_spent_pelanggan'
            ]].mean().round(2)
            st.dataframe(summary, use_container_width=True)

        # Kelompok Umur (jika ada kolom age_group)
        if 'age_group' in customer_df.columns:
            st.markdown("---")
            col_c, col_d = st.columns(2)

            with col_c:
                st.subheader("Proporsi Kelompok Umur")
                age_group = customer_df['age_group'].value_counts()
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.pie(age_group.values, labels=age_group.index, autopct='%1.1f%%')
                st.pyplot(fig); plt.close(fig)

            with col_d:
                st.subheader("Total Transaksi per Kelompok Umur")
                age_tx = (
                    customer_df.groupby('age_group')['total_transaction_pelanggan']
                    .sum().sort_values()
                )
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.barh(age_tx.index, age_tx.values)
                ax.set_xlabel("Total Transaksi")
                st.pyplot(fig); plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – Segmentasi Kategori
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Segmentasi Kategori Produk (K-Means)")

    category_df = load(category_file, "category_segment.csv")
    if category_df is not None:

        # Re-run KMeans
        fitur_cat = category_df[['total_revenue_kategori', 'banyak_transaksi_categori']].dropna()
        scaler2 = StandardScaler()
        scaled2 = scaler2.fit_transform(fitur_cat)
        kmeans2 = KMeans(n_clusters=4, random_state=42, n_init='auto')
        category_df = category_df.loc[fitur_cat.index].copy()
        category_df['cluster'] = kmeans2.fit_predict(scaled2)

        col_a, col_b = st.columns(2)

        # Top Revenue Kategori
        with col_a:
            st.subheader("Top Revenue per Kategori")
            if 'category' in category_df.columns:
                top_cat = (
                    category_df.groupby('category')['total_revenue_kategori']
                    .sum().sort_values().tail(10)
                )
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.barh(top_cat.index, top_cat.values)
                ax.set_xlabel("Total Revenue")
                st.pyplot(fig); plt.close(fig)

        # Scatter Cluster Kategori
        with col_b:
            st.subheader("Visualisasi Cluster Kategori")
            fig, ax = plt.subplots(figsize=(6, 4))
            for c in sorted(category_df['cluster'].unique()):
                sub = category_df[category_df['cluster'] == c]
                ax.scatter(
                    sub['banyak_transaksi_categori'],
                    sub['total_revenue_kategori'],
                    label=f"Cluster {c}", alpha=0.6
                )
            ax.set_xlabel("Jumlah Transaksi"); ax.set_ylabel("Total Revenue")
            ax.legend()
            st.pyplot(fig); plt.close(fig)

        # Top 10 Produk
        if 'product_name' in category_df.columns:
            st.subheader("Top 10 Produk berdasarkan Revenue")
            top_prod = (
                category_df.groupby('product_name')['total_revenue_kategori']
                .sum().sort_values(ascending=False).head(10)
            )
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.bar(top_prod.index, top_prod.values)
            ax.set_xticklabels(top_prod.index, rotation=90)
            ax.set_ylabel("Total Revenue")
            st.pyplot(fig); plt.close(fig)

        # Summary cluster
        st.subheader("Rata-rata per Cluster Kategori")
        summary2 = category_df.groupby('cluster')[[
            'total_revenue_kategori', 'banyak_transaksi_categori'
        ]].mean().round(2)
        st.dataframe(summary2, use_container_width=True)
