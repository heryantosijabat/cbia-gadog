from pathlib import Path

import streamlit as st
import pandas as pd
import pickle
import altair as alt

st.title("Dashboard Analitik DJPb")

# Sidebar Filter
st.sidebar.header("Menu Filter")
opsi = st.sidebar.selectbox("Pilih Model:", ["Klasifikasi", "Regresi"])

app_dir = Path(__file__).resolve().parent
data = pd.read_csv(app_dir / "data" / "02_realisasi_anggaran_klasifikasi.csv")
model_path = app_dir / "models" / "Best_model.pkcls"

tab_prediksi, tab_visualisasi = st.tabs(["Menu Prediksi", "Visualisasi"])
with tab_prediksi:
    st.header("Prediksi Realisasi 95%")
    st.markdown("Masukkan data input berikut dan klik tombol untuk melihat hasil prediksi.")

    col1, col2 = st.columns(2)
    with col1:
        jumlah_spm = st.number_input(
            "Jumlah SPM",
            min_value=1,
            max_value=200,
            value=10,
            step=1,
            format="%d",
        )
        revisi_dipa = st.number_input(
            "Revisi DIPA",
            min_value=0,
            max_value=5,
            value=1,
            step=1,
            format="%d",
        )
        deviasi_rpd_persen = st.number_input(
            "Deviasi RPD (%)",
            min_value=0.00,
            max_value=30.00,
            value=5.50,
            format="%.2f",
        )
    with col2:
        skor_ikpa = st.slider(
            "Skor IKPA",
            min_value=70.00,
            max_value=100.00,
            value=85.00,
            step=0.01,
            format="%.2f",
        )
        tipe_satker = st.selectbox(
            "Tipe Satker",
            ["Dekonsentrasi", "Kantor Daerah", "Kantor Pusat", "Tugas Pembantuan"],
        )

    if st.button("Prediksi", type="primary"):
        try:
            import Orange

            with model_path.open("rb") as model_file:
                model = pickle.load(model_file)

            tipe_values = [
                "Dekonsentrasi",
                "Kantor Daerah",
                "Kantor Pusat",
                "Tugas Pembantuan",
            ]
            tipe_indicators = [
                1.0 if tipe_satker == tipe_value else 0.0
                for tipe_value in tipe_values
            ]
            input_row = [
                float(jumlah_spm),
                float(revisi_dipa),
                float(deviasi_rpd_persen),
                float(skor_ikpa),
                *tipe_indicators,
            ]

            table = Orange.data.Table.from_list(model.domain, [input_row])
            prediction_index = int(model(table)[0])
            probabilities = model(table, model.Probs)[0]
            class_name = model.domain.class_var.values[prediction_index]
            confidence = float(probabilities[prediction_index])

            if class_name == "Ya":
                st.success(f"Prediksi: {class_name} (keyakinan {confidence:.1%})")
            else:
                st.warning(f"Prediksi: {class_name} (keyakinan {confidence:.1%})")
        except (
            OSError,
            ImportError,
            pickle.UnpicklingError,
            EOFError,
            AttributeError,
            ValueError,
            IndexError,
            TypeError,
        ) as exc:
            st.error(f"Prediksi gagal dijalankan: {exc}")


with tab_visualisasi:
    st.header("Visualisasi Bubble Plot")
    st.markdown(
        "Grafik bubble menunjukkan skor IKPA terhadap deviasi RPD, dengan warna berdasarkan kolom `realisasi_tercapai_95persen`."
    )

    chart = (
        alt.Chart(data)
        .mark_circle(opacity=0.7, size=100)
        .encode(
            x=alt.X(
                "skor_ikpa",
                title="Skor IKPA",
                scale=alt.Scale(domain=[70, 100]),
            ),
            y=alt.Y("deviasi_rpd_persen", title="Deviasi RPD (%)"),
            color=alt.Color(
                "realisasi_tercapai_95persen:N",
                title="Realisasi Tercapai 95%",
                scale=alt.Scale(
                    domain=["Tidak", "Ya"],
                    range=["#ef4444", "#22c55e"],
                ),
            ),
            tooltip=[
                "kode_satker",
                "tipe_satker",
                "skor_ikpa",
                "deviasi_rpd_persen",
                "realisasi_tercapai_95persen",
            ],
        )
        .properties(width=900, height=550)
        .interactive()
    )
    st.altair_chart(chart, width=900)

    st.header("Diagram Batang Status Realisasi")

    data_status = (
        data["realisasi_tercapai_95persen"]
        .value_counts()
        .rename_axis("Realisasi Tercapai 95%")
        .reset_index(name="Jumlah")
    )

    bar_chart = (
        alt.Chart(data_status)
        .mark_bar()
        .encode(
            x=alt.X(
                "Realisasi Tercapai 95%:N",
                title="Realisasi Tercapai 95%",
            ),
            y=alt.Y("Jumlah:Q", title="Jumlah Satuan Kerja"),
            color=alt.Color(
                "Realisasi Tercapai 95%:N",
                scale=alt.Scale(
                    domain=["Tidak", "Ya"],
                    range=["#facc15", "#22c55e"],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Realisasi Tercapai 95%:N"),
                alt.Tooltip("Jumlah:Q", title="Jumlah Satuan Kerja"),
            ],
        )
        .properties(width=900, height=400)
    )

    st.altair_chart(bar_chart, width=900)

    st.header("Diagram Revisi DIPA")
    st.markdown(
        "Diagram batang menampilkan jumlah satuan kerja berdasarkan jumlah revisi DIPA."
    )

    data_revisi = (
        data["revisi_dipa"]
        .value_counts()
        .rename_axis("Jumlah Revisi DIPA")
        .reset_index(name="Jumlah Satuan Kerja")
        .sort_values("Jumlah Revisi DIPA")
    )

    revisi_chart = (
        alt.Chart(data_revisi)
        .mark_bar(color="#3b82f6", cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(
                "Jumlah Revisi DIPA:O",
                title="Jumlah Revisi DIPA",
            ),
            y=alt.Y(
                "Jumlah Satuan Kerja:Q",
                title="Jumlah Satuan Kerja",
            ),
            tooltip=[
                alt.Tooltip(
                    "Jumlah Revisi DIPA:O",
                    title="Revisi DIPA",
                ),
                alt.Tooltip(
                    "Jumlah Satuan Kerja:Q",
                    title="Jumlah Satuan Kerja",
                ),
            ],
        )
        .properties(width=900, height=400)
    )

    revisi_labels = revisi_chart.mark_text(
        dy=-8,
        color="#1e3a5f",
        fontWeight="bold",
    ).encode(text="Jumlah Satuan Kerja:Q")

    st.altair_chart(revisi_chart + revisi_labels, width=900)

    st.header("Ringkasan Kinerja Penyerapan Anggaran per Satker")

    ringkasan = data[
        [
            "kode_satker",
            "nama_kementerian",
            "provinsi",
            "realisasi_tw1_persen",
            "realisasi_tw2_persen",
            "realisasi_tw3_persen",
            "deviasi_rpd_persen",
            "skor_ikpa",
            "realisasi_tercapai_95persen",
        ]
    ].rename(
        columns={
            "kode_satker": "Kode Satker",
            "nama_kementerian": "Kementerian",
            "provinsi": "Provinsi",
            "realisasi_tw1_persen": "Realisasi TW1 (%)",
            "realisasi_tw2_persen": "Realisasi TW2 (%)",
            "realisasi_tw3_persen": "Realisasi TW3 (%)",
            "deviasi_rpd_persen": "Deviasi RPD (%)",
            "skor_ikpa": "Skor IKPA",
            "realisasi_tercapai_95persen": "Tercapai 95%",
        }
    )

    ringkasan = ringkasan.sort_values(
        ["Realisasi TW3 (%)", "Skor IKPA"],
        ascending=[False, False],
    )

    st.dataframe(
        ringkasan,
        column_config={
            "Realisasi TW1 (%)": st.column_config.ProgressColumn(
                "Realisasi TW1 (%)",
                format="%.2f%%",
                min_value=0,
                max_value=100,
            ),
            "Realisasi TW2 (%)": st.column_config.ProgressColumn(
                "Realisasi TW2 (%)",
                format="%.2f%%",
                min_value=0,
                max_value=100,
            ),
            "Realisasi TW3 (%)": st.column_config.ProgressColumn(
                "Realisasi TW3 (%)",
                format="%.2f%%",
                min_value=0,
                max_value=100,
            ),
            "Deviasi RPD (%)": st.column_config.NumberColumn(
                "Deviasi RPD (%)",
                format="%.2f%%",
            ),
            "Skor IKPA": st.column_config.ProgressColumn(
                "Skor IKPA",
                format="%.2f",
                min_value=0,
                max_value=100,
            ),
        },
        hide_index=True,
        width="stretch",
    )