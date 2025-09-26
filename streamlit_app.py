# streamlit_app.py
# --------------------------------------------------
# Oferta Turística — Macizo Colombiano (Cauca)
# + Tab 2: Text mining (WordClouds + unigram/bigram tables)
# --------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap, Fullscreen
from PIL import Image
from io import BytesIO
import base64
import os
import unicodedata
import re
from pathlib import Path

# NEW: NLP deps
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import Counter
import matplotlib.pyplot as plt

# =============  PAGE SETUP  =============
st.set_page_config(
    page_title="Oferta Turística — Macizo Colombiano (Cauca)",
    page_icon="🏔️",
    layout="wide"
)

# =============  STYLES  =============
st.markdown(
    """
    <style>
        .stMultiSelect [data-baseweb="tag"]      {background:#9c3675!important;}
        .stMultiSelect [data-baseweb="tag"] div  {color:white!important;}
        .stMultiSelect>div                        {border-color:#9c3675!important;}
        input[type="checkbox"]+div svg           {color:#9c3675!important;stroke:#fff!important;fill:#9c3675!important;}
        .legend-badge{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:middle;}
        .legend-row{font-size:12px;margin-bottom:6px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# =============  LOGOS (sidebar)  =============
def _load_b64(path: str):
    try:
        img = Image.open(path)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""

with st.sidebar:
    st.markdown(
        """
        <style>
            .logo img{margin:0;padding:0;border-radius:0;box-shadow:none;}
            .powered{display:flex;justify-content:center;align-items:center;gap:8px;font-size:11px;color:grey;}
            .powered img{height:45px;width:45px;border-radius:50%;object-fit:cover;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="logo">', unsafe_allow_html=True)
    if os.path.exists("assets/logo_mincit_fontur.jpeg"):
        st.image("assets/logo_mincit_fontur.jpeg", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    b64 = _load_b64("assets/datad_logo.jpeg")
    if b64:
        st.markdown(
            f"""<div class="powered"><img src="data:image/png;base64,{b64}"/><span>Powered by DataD</span></div>""",
            unsafe_allow_html=True,
        )

# =============  MARCADORES POR PLACE_TYPE  =============
def _norm(s: str) -> str:
    s = str(s or "").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.split())

PLACE_TYPE_LABELS = [
    'Hoteles', 'Posadas rurales', 'Camping', 'Glamping',
    'Viviendas turísticas', 'Fincas turísticas', 'Restaurantes',
    'Gastronomía local', 'Dulces típicos', 'Agencias de viaje',
    'Guía turístico', 'Transporte terrestre típico',
    'Transporte turístico especial ', 'Puntos de información',
    'Lugares religiosos', 'lugares culturales', 'Atractivos hídricos',
    'Atractivos volcánicos', 'Atractivos de montaña',
    'Flora y botánica', 'Comunidad indígena', 'Comunidad campesina',
    'Servicios de salud básicos', 'Clínicas', 'Hospitales', 'Banco',
    'Cajero', 'Sucursal Bancaria', 'Casa de Cambio',
    'Terminal de transporte', 'Transporte fluvial', 'Puntos wifi',
    'Peajes', 'Estaciones de servicio', 'Bomberos', 'Policía',
    'Artesanías y oficios', 'Cultura y Comunidad',
    'Arqueología y fósiles', 'Patrimonio colonial',
    'Arquitectura simbólica', 'Cultura ancestral',
    'Espacios naturales', 'Naturaleza Extrema',
    'Biodiversidad y selva', 'Avistamiento de aves',
    'Avistamiento de ranas', 'Senderismo', 'Turismo de aventura',
    'Experiencia de la caficultura', 'rutas agroturísticas',
    'Tratamientos y terapias de relajación',
    'Terapias alternativas y holísticas', 'Espiritual y ancestral',
    'Termales', 'Retiros', 'Hotel campestre',
    'Avistamiento de fauna salvaje', 'Ruta de flora y botánica'
]
DISPLAY_LABEL = {_norm(x): x.strip() for x in PLACE_TYPE_LABELS}

# Mapeo a íconos FA4 + colores AwesomeMarkers
PLACE_TYPE_MARKERS = {
    _norm('Hoteles'): ("bed", "green"),
    _norm('Posadas rurales'): ("home", "lightgreen"),
    _norm('Camping'): ("tree", "darkgreen"),
    _norm('Glamping'): ("star", "pink"),
    _norm('Viviendas turísticas'): ("home", "lightgreen"),
    _norm('Fincas turísticas'): ("pagelines", "darkgreen"),
    _norm('Restaurantes'): ("cutlery", "blue"),
    _norm('Gastronomía local'): ("cutlery", "lightblue"),
    _norm('Dulces típicos'): ("shopping-basket", "orange"),
    _norm('Agencias de viaje'): ("suitcase", "purple"),
    _norm('Guía turístico'): ("compass", "cadetblue"),
    _norm('Transporte terrestre típico'): ("car", "darkblue"),
    _norm('Transporte turístico especial'): ("car", "blue"),
    _norm('Puntos de información'): ("info-circle", "cadetblue"),
    _norm('Lugares religiosos'): ("building", "lightgray"),
    _norm('lugares culturales'): ("university", "orange"),
    _norm('Atractivos hídricos'): ("tint", "blue"),
    _norm('Atractivos volcánicos'): ("fire", "red"),
    _norm('Atractivos de montaña'): ("flag", "darkpurple"),
    _norm('Flora y botánica'): ("leaf", "green"),
    _norm('Comunidad indígena'): ("users", "beige"),
    _norm('Comunidad campesina'): ("users", "beige"),
    _norm('Servicios de salud básicos'): ("medkit", "red"),
    _norm('Clínicas'): ("hospital-o", "lightred"),
    _norm('Hospitales'): ("h-square", "darkred"),
    _norm('Banco'): ("university", "darkred"),
    _norm('Cajero'): ("credit-card", "darkred"),
    _norm('Sucursal Bancaria'): ("university", "darkred"),
    _norm('Casa de Cambio'): ("money", "darkred"),
    _norm('Terminal de transporte'): ("bus", "darkpurple"),
    _norm('Transporte fluvial'): ("ship", "blue"),
    _norm('Puntos wifi'): ("wifi", "lightblue"),
    _norm('Peajes'): ("road", "gray"),
    _norm('Estaciones de servicio'): ("tint", "black"),
    _norm('Bomberos'): ("fire-extinguisher", "red"),
    _norm('Policía'): ("shield", "darkblue"),
    _norm('Artesanías y oficios'): ("paint-brush", "orange"),
    _norm('Cultura y Comunidad'): ("users", "orange"),
    _norm('Arqueología y fósiles'): ("history", "gray"),
    _norm('Patrimonio colonial'): ("university", "orange"),
    _norm('Arquitectura simbólica'): ("building", "lightgray"),
    _norm('Cultura ancestral'): ("book", "beige"),
    _norm('Espacios naturales'): ("tree", "green"),
    _norm('Naturaleza Extrema'): ("bolt", "black"),
    _norm('Biodiversidad y selva'): ("leaf", "darkgreen"),
    _norm('Avistamiento de aves'): ("binoculars", "green"),
    _norm('Avistamiento de ranas'): ("binoculars", "lightgreen"),
    _norm('Senderismo'): ("map", "darkgreen"),
    _norm('Turismo de aventura'): ("bicycle", "darkblue"),
    _norm('Experiencia de la caficultura'): ("coffee", "beige"),
    _norm('rutas agroturísticas'): ("map", "lightgreen"),
    _norm('Tratamientos y terapias de relajación'): ("heart", "pink"),
    _norm('Terapias alternativas y holísticas'): ("heart", "lightred"),
    _norm('Espiritual y ancestral'): ("leaf", "purple"),
    _norm('Termales'): ("tint", "lightblue"),
    _norm('Retiros'): ("home", "pink"),
    _norm('Hotel campestre'): ("bed", "lightgreen"),
    _norm('Avistamiento de fauna salvaje'): ("binoculars", "green"),
    _norm('Ruta de flora y botánica'): ("map", "green"),
}
DEFAULT_PT_MARKER = ("map-marker", "gray")

def marker_of_place_type(place_type: str):
    return PLACE_TYPE_MARKERS.get(_norm(place_type), DEFAULT_PT_MARKER)

def label_of_place_type(place_type: str):
    return DISPLAY_LABEL.get(_norm(place_type), str(place_type or "").strip())

# =============  POPUP HELPER  =============
def make_popup_html(r: dict) -> str:
    name  = r.get("name", "Sin nombre")
    muni  = r.get("municipio", "No Info")
    dim   = r.get("dimension", "")
    subd  = r.get("sub_dimension", "")
    cat   = r.get("category", "")
    ptype = label_of_place_type(r.get("place_type"))
    rating  = r.get("average_rating", "No Info")
    reviews = r.get("user_ratings_total", 0)
    link    = r.get("map_link", "")

    link_html = f"<br><b>Google:</b> <a href='{link}' target='_blank'>Ver en Google</a>" if link else ""
    return (
        f"<div style='font-size:13px'>"
        f"<b>{name}</b>"
        f"<br><b>Municipio:</b> {muni}"
        f"<br><b>Dimensión:</b> {dim}"
        f"<br><b>Sub-dimensión:</b> {subd}"
        f"<br><b>Categoría:</b> {cat}"
        f"<br><b>Tipo de lugar:</b> {ptype}"
        f"<br><b>Rating:</b> {rating} ({reviews} reviews)"
        f"{link_html}</div>"
    )

# =============  DATA  =============
@st.cache_data
def load_data(csv_path: str = "map_data.csv") -> pd.DataFrame:
    path = csv_path if os.path.exists(csv_path) else os.path.join("datain", "map_data.csv")
    df = pd.read_csv(path)

    for col, default in [
        ("municipio", "No Info"),
        ("corredor", "Macizo Colombiano"),
        ("dimension", None),
        ("sub_dimension", None),
        ("category", None),
        ("place_type", None),
        ("name", None),
        ("average_rating", "No Info"),
        ("user_ratings_total", 0),
        ("latitude", None),
        ("longitude", None),
        # NEW: ensure text column exists (if not, create empty)
        ("text_es", "")
    ]:
        if col not in df.columns:
            df[col] = default

    df["average_rating"] = df["average_rating"].fillna("No Info").astype(str)
    df["user_ratings_total"] = pd.to_numeric(df["user_ratings_total"], errors="coerce").fillna(0).astype(int)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    link_cols = [c for c in ["place_link", "google_maps_url", "gmaps_url_canonical", "url"] if c in df.columns]
    df["map_link"] = df[link_cols[0]] if link_cols else ""

    df = df.dropna(subset=["latitude", "longitude"])
    return df

df = load_data()

# =============  FILTROS SIDEBAR  =============
st.sidebar.title("Filtros geográficos y de contenido")
corr_all = sorted(df["corredor"].dropna().unique().tolist())
sel_corr = st.sidebar.multiselect("Corredor", corr_all, default=[])
if sel_corr:
    mun_all = sorted(df[df["corredor"].isin(sel_corr)]["municipio"].dropna().unique().tolist())
else:
    mun_all = sorted(df["municipio"].dropna().unique().tolist())
sel_mun = st.sidebar.multiselect("Municipio", mun_all, default=(mun_all if sel_corr else []))

if sel_mun:
    dim_all = sorted(df[df["municipio"].isin(sel_mun)]["dimension"].dropna().unique().tolist())
else:
    dim_all = sorted(df["dimension"].dropna().unique().tolist())
sel_dim = st.sidebar.multiselect("Dimensión", dim_all, default=[])

if sel_dim:
    tmp_sub = df[df["dimension"].isin(sel_dim)]
    if sel_mun: tmp_sub = tmp_sub[tmp_sub["municipio"].isin(sel_mun)]
else:
    tmp_sub = df[df["municipio"].isin(sel_mun)] if sel_mun else df
subdim_all = sorted(tmp_sub["sub_dimension"].dropna().unique().tolist())
sel_subdim = st.sidebar.multiselect("Sub-dimensión", subdim_all, default=[])

tmp_cat = tmp_sub[tmp_sub["sub_dimension"].isin(sel_subdim)] if sel_subdim else tmp_sub
cat_all = sorted(tmp_cat["category"].dropna().unique().tolist())
sel_cat = st.sidebar.multiselect("Categoría", cat_all, default=[])

tmp_ptype = tmp_cat[tmp_cat["category"].isin(sel_cat)] if sel_cat else tmp_cat
ptype_all = sorted(tmp_ptype["place_type"].dropna().unique().tolist())
sel_ptype = st.sidebar.multiselect("Tipo de lugar", ptype_all, default=[])

st.sidebar.markdown("---")
show_markers = st.sidebar.checkbox("Mostrar marcadores", True)
show_heatmap = st.sidebar.checkbox("Mostrar mapa de calor", False)

# Leyenda por place_type
with st.sidebar.expander("Leyenda por tipo de lugar", expanded=False):
    for key_norm, (ico, col) in sorted(PLACE_TYPE_MARKERS.items(), key=lambda x: label_of_place_type(x[0])):
        lbl = label_of_place_type(key_norm)
        st.markdown(
            f"""<div class="legend-row"><span class="legend-badge" style="background:{col};"></span>{lbl}</div>""",
            unsafe_allow_html=True
        )

# =============  FILTRADO  =============
fdf = df.copy()
if sel_corr:   fdf = fdf[fdf["corredor"].isin(sel_corr)]
if sel_mun:    fdf = fdf[fdf["municipio"].isin(sel_mun)]
if sel_dim:    fdf = fdf[fdf["dimension"].isin(sel_dim)]
if sel_subdim: fdf = fdf[fdf["sub_dimension"].isin(sel_subdim)]
if sel_cat:    fdf = fdf[fdf["category"].isin(sel_cat)]
if sel_ptype:  fdf = fdf[fdf["place_type"].isin(sel_ptype)]

ready_to_plot = bool(sel_dim)

# ==========================================================
# =============  NLP UTILITIES (Text mining)  ==============
# ==========================================================
TEXT_COL_CANDIDATES = ["text_es", "review_text_es", "review_text", "text"]

SPANISH_STOP = {
    "a","acá","ahora","al","algo","algunas","algunos","allí","allá","ante","antes","aquel","aquella",
    "aquellas","aquellos","aqui","aquí","así","aunque","cada","como","con","contra","cual","cuales",
    "cuando","cuanto","de","del","desde","donde","dos","el","ella","ellas","ellos","en","entre","era",
    "eran","eres","es","esa","esas","ese","eso","esos","esta","estaba","estaban","estamos","estan",
    "están","estar","este","esto","estos","fue","fueron","ha","haber","había","hay","la","las","le",
    "les","lo","los","mas","más","me","mientras","muy","ni","no","nos","nosotros","o","otra",
    "otros","para","pero","poco","por","porque","que","quien","quién","quienes","se","sin","sobre",
    "su","sus","te","tiene","tienen","tu","tus","un","una","uno","unos","y","ya"
}
DOMAIN_STOP = {"lugar", "sitio"}  # tune if generic words dominate
BASE_STOP = SPANISH_STOP | DOMAIN_STOP

def normalize_spanish(text: str) -> str:
    s = str(text or "").lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = re.sub(r"[^a-zñü\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def preprocess_docs(series: pd.Series) -> list[str]:
    docs = []
    for raw in series.fillna("").astype(str):
        s = normalize_spanish(raw)
        toks = [t for t in s.split() if len(t) > 2 and t not in BASE_STOP]
        docs.append(" ".join(toks))
    return docs

def adaptive_df(n_docs: int):
    if n_docs >= 200: return 3, 0.60
    if n_docs >= 50:  return 2, 0.75
    return 1, 1.0

def build_freq(docs: list[str], topk: int = 400) -> dict:
    if not docs: return {}
    token_total = sum(len(d.split()) for d in docs)
    if token_total < 25: return {}

    n_docs = len(docs)
    min_df, max_df = adaptive_df(n_docs)

    attempts = [
        TfidfVectorizer(ngram_range=(1,2), min_df=min_df, max_df=max_df, stop_words=list(BASE_STOP), norm=None),
        TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=1.0,      stop_words=list(BASE_STOP), norm=None),
        TfidfVectorizer(ngram_range=(1,1), min_df=1, max_df=1.0,      stop_words=list(BASE_STOP), norm=None),
        CountVectorizer(ngram_range=(1,2), min_df=1, max_df=1.0,      stop_words=list(BASE_STOP))
    ]
    for vec in attempts:
        try:
            X = vec.fit_transform(docs)
            if X.shape[1] == 0: continue
            terms = vec.get_feature_names_out()
            weights = np.asarray(X.sum(axis=0)).ravel()
            freq = dict(zip(terms, weights))
            return dict(sorted(freq.items(), key=lambda x: x[1], reverse=True)[:topk])
        except ValueError:
            continue

    # last resort
    bag = Counter()
    for d in docs:
        t = [w for w in d.split() if w not in BASE_STOP]
        bag.update(t)
        bag.update([" ".join(p) for p in zip(t, t[1:])])
    return dict(sorted(bag.items(), key=lambda x: x[1], reverse=True)[:topk])

def top_table(freq_map: dict, n=25) -> pd.DataFrame:
    if not freq_map:
        return pd.DataFrame(columns=["term","weight","share_%"])
    s = pd.Series(freq_map).sort_values(ascending=False).head(n)
    df = s.rename_axis("term").reset_index(name="weight")
    df["share_%"] = (100 * df["weight"] / df["weight"].sum()).round(2)
    return df

def draw_wordcloud(freq_map: dict, title: str):
    wc = WordCloud(
        width=1400, height=800, background_color="white",
        prefer_horizontal=0.8, collocations=False,
        max_words=300, relative_scaling=0.3,
        min_font_size=10, contour_width=0.8, contour_color="grey"
    ).generate_from_frequencies(freq_map)
    fig = plt.figure(figsize=(12, 7), dpi=140)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, pad=8)
    return fig

# =============  CONTENIDO  =============
st.title("Oferta turística — Macizo Colombiano (Cauca)")
st.markdown(
    "Panel interactivo de **servicios y atractivos turísticos** identificados vía Google Maps, "
    "en municipios del **Macizo Colombiano (Cauca)**. "
)

# NEW: two tabs
tab_map, tab_text = st.tabs(["Mapa", "Text mining"])

with tab_map:
    if ready_to_plot and not fdf.empty:
        st.markdown("### Resultados filtrados")
        show_cols = [
            "name","municipio","dimension","sub_dimension","category","place_type",
            "average_rating","user_ratings_total","latitude","longitude","map_link"
        ]
        cols_present = [c for c in show_cols if c in fdf.columns]
        table = fdf.copy()
        if "average_rating" in table.columns:
            table["_avg_num"] = pd.to_numeric(table["average_rating"], errors="coerce")
            table = table.sort_values(by=["_avg_num"], ascending=False).drop(columns=["_avg_num"], errors="ignore")
        st.dataframe(table[cols_present], use_container_width=True)
        st.download_button(
            "Descargar CSV",
            data=table[cols_present].to_csv(index=False),
            file_name="oferta_turistica_filtrada.csv",
            mime="text/csv"
        )

        # ======= MAPA =======
        mlat, mlon = fdf["latitude"].mean(), fdf["longitude"].mean()
        fmap = folium.Map([mlat, mlon], zoom_start=9, control_scale=True)
        Fullscreen(position="topright", title="Pantalla completa", title_cancel="Salir").add_to(fmap)

        if show_heatmap:
            HeatMap(fdf[["latitude","longitude"]].values.tolist(), radius=12, blur=15).add_to(fmap)

        if show_markers:
            for _, r in fdf.iterrows():
                icon_name, color = marker_of_place_type(r.get("place_type"))
                name = r.get("name", "Sin nombre")
                html = make_popup_html(r)

                folium.Marker(
                    [r["latitude"], r["longitude"]],
                    icon=folium.Icon(icon=icon_name, color=color, prefix="fa"),
                    tooltip=name,
                    popup=folium.Popup(html, max_width=320)
                ).add_to(fmap)

        folium.LayerControl(collapsed=False).add_to(fmap)
        st_folium(fmap, height=650, use_container_width=True)

    elif ready_to_plot and fdf.empty:
        st.warning("No se encontraron resultados con los filtros seleccionados. Ajuste la segmentación.")
    else:
        st.info("Para visualizar resultados, seleccione al menos una **Dimensión** (y refine con Sub-dimensión, Categoría y Tipo de lugar).")

# ==================== TAB 2: TEXT MINING ====================
with tab_text:
    st.subheader("Insights de texto: WordClouds + conteo de términos")
    text_col = next((c for c in TEXT_COL_CANDIDATES if c in fdf.columns), None)

    if not ready_to_plot:
        st.info("Para activar el análisis de texto, seleccione al menos una **Dimensión** en el panel de filtros.")
    elif text_col is None:
        st.warning("No se encontró una columna de texto. Agregue `text_es` (o actualice TEXT_COL_CANDIDATES).")
    elif fdf[text_col].fillna("").str.strip().eq("").all():
        st.warning("No hay texto disponible en los registros filtrados.")
    else:
        # Controls
        topk = st.slider("Top términos para tablas", min_value=10, max_value=100, value=25, step=5)
        min_docs = st.slider("Mínimo de registros por corte", min_value=5, max_value=50, value=10, step=5)

        # Preprocess once
        working = fdf.copy()
        working["doc_clean"] = preprocess_docs(working[text_col])

        def render_block(title, docs):
            docs = [d for d in docs if isinstance(d, str) and d]
            if len(docs) < min_docs:
                st.info(f"{title}: muestra insuficiente (N={len(docs)}).")
                return
            freq = build_freq(docs, topk=400)
            if not freq:
                st.info(f"{title}: sin vocabulario útil tras limpieza.")
                return

            col1, col2 = st.columns([3, 2])
            with col1:
                fig = draw_wordcloud(freq, f"{title} — Unigramas + Bigramas")
                st.pyplot(fig, use_container_width=True)
                bi = {k:v for k,v in freq.items() if " " in k}
                if len(bi) >= 10:
                    fig2 = draw_wordcloud(bi, f"{title} — Solo Bigramas (frases)")
                    st.pyplot(fig2, use_container_width=True)

            with col2:
                uni = {k:v for k,v in freq.items() if " " not in k}
                bi  = {k:v for k,v in freq.items() if " " in k}
                df_uni = top_table(uni, n=topk)
                df_bi  = top_table(bi,  n=topk)

                st.markdown("**Top Unigramas**")
                st.dataframe(df_uni, use_container_width=True, height=270)
                st.download_button(
                    "Descargar unigramas (CSV)",
                    data=df_uni.to_csv(index=False),
                    file_name=f"{re.sub(r'\\W+','_',title.lower())}_top_unigrams.csv",
                    mime="text/csv"
                )
                st.markdown("---")
                st.markdown("**Top Bigramas**")
                st.dataframe(df_bi, use_container_width=True, height=270)
                st.download_button(
                    "Descargar bigramas (CSV)",
                    data=df_bi.to_csv(index=False),
                    file_name=f"{re.sub(r'\\W+','_',title.lower())}_top_bigrams.csv",
                    mime="text/csv"
                )

        # 1) General
        st.markdown("### 1) General — todos los municipios y categorías")
        render_block("General (todos los municipios y categorías)", working["doc_clean"].tolist())

        # 2) By municipio
        st.markdown("### 2) Desagregado por municipio")
        for muni, g in working.groupby("municipio", dropna=False):
            muni_label = "Sin municipio" if pd.isna(muni) else str(muni)
            with st.expander(f"Municipio: {muni_label}  —  N={len(g)}", expanded=False):
                render_block(f"Municipio: {muni_label}", g["doc_clean"].tolist())

        # 3) By municipio + category
        st.markdown("### 3) Desagregado por municipio y categoría")
        # To avoid overload, cap to first 60 slices by size desc; adjust as needed
        groups = (
            working.groupby(["municipio","category"], dropna=False)
                   .size().sort_values(ascending=False).head(60).index.tolist()
        )
        for (muni, cat) in groups:
            g = working[(working["municipio"].eq(muni)) & (working["category"].eq(cat))]
            muni_label = "Sin municipio" if pd.isna(muni) else str(muni)
            cat_label  = "Sin categoría" if pd.isna(cat) else str(cat)
            with st.expander(f"{muni_label}  |  {cat_label}  —  N={len(g)}", expanded=False):
                render_block(f"Municipio: {muni_label} | Categoría: {cat_label}", g["doc_clean"].tolist())
