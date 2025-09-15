# streamlit_app.py

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap, Fullscreen
from PIL import Image
from io import BytesIO
import base64
import os

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

# =============  CATEGORY MARKERS (icon + color)  =============
# Colores válidos (AwesomeMarkers): red, darkred, lightred, orange, beige,
# green, darkgreen, lightgreen, blue, darkblue, lightblue, purple, darkpurple,
# pink, cadetblue, white, gray, lightgray, black
# Íconos FA "fa" (compatibles con versiones comunes de folium/awesome-markers)
CATEGORY_MARKERS = {
    "Alojamiento / Hospedajes":           ("bed",             "green"),
    "Restaurantes":                       ("cutlery",         "blue"),
    "Agencias de viaje y guías":          ("suitcase",        "purple"),
    "Transporte turístico":               ("car",             "darkblue"),
    "Puntos de información":              ("info-circle",     "cadetblue"),
    "Atractivos culturales":              ("university",      "orange"),
    "Atractivos naturales":               ("tree",            "darkgreen"),
    "Atractivos comunitarios":            ("users",           "lightgray"),
    "Salud":                              ("medkit",          "red"),
    "Financieros":                        ("money",           "darkred"),
    "Transporte Público":                 ("bus",             "darkpurple"),
    "Internet":                           ("wifi",            "lightblue"),
    "Peajes":                             ("road",            "gray"),
    "Estaciones de servicio":             ("tint",            "black"),
    "Servicios de emergencia":            ("ambulance",       "lightred"),
    "Experiencias y rutas culturales":    ("map",             "beige"),
    "Experiencias de naturaleza y montaña":("leaf",           "lightgreen"),
    "Avistamiento de flora y fauna":      ("binoculars",      "green"),
    "Experiencias de aventura y agroturismo": ("bicycle",     "darkblue"),
    "Servicios Wellness":                 ("heart",           "pink"),
}
DEFAULT_MARKER = ("map-marker", "gray")

def marker_of(category: str):
    key = (category or "").strip()
    return CATEGORY_MARKERS.get(key, DEFAULT_MARKER)

# =============  DATA  =============
@st.cache_data
def load_data(csv_path: str = "map_data.csv") -> pd.DataFrame:
    path = csv_path if os.path.exists(csv_path) else os.path.join("datain", "map_data.csv")
    df = pd.read_csv(path)

    # columnas mínimas esperadas; si no existen, se crean para robustez
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
    ]:
        if col not in df.columns:
            df[col] = default

    # tipos
    df["average_rating"] = df["average_rating"].fillna("No Info").astype(str)
    df["user_ratings_total"] = pd.to_numeric(df["user_ratings_total"], errors="coerce").fillna(0).astype(int)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # link a Google Maps (col flexible)
    link_cols = [c for c in ["place_link", "google_maps_url", "gmaps_url_canonical", "url"] if c in df.columns]
    df["map_link"] = df[link_cols[0]] if link_cols else ""

    # normalizar categoría con strip para que coincida con el mapping
    df["category"] = df["category"].astype(str).str.strip()

    # descartamos filas sin coordenadas
    df = df.dropna(subset=["latitude", "longitude"])
    return df

df = load_data()

# =============  SIDEBAR — FILTROS  =============
st.sidebar.title("Filtros geográficos y de contenido")

# 1) Corredor
corr_all = sorted(df["corredor"].dropna().unique().tolist())
sel_corr = st.sidebar.multiselect("Corredor", corr_all, default=[])

# 2) Municipios (dependen del corredor)
if sel_corr:
    mun_all = sorted(df[df["corredor"].isin(sel_corr)]["municipio"].dropna().unique().tolist())
else:
    mun_all = sorted(df["municipio"].dropna().unique().tolist())
sel_mun = st.sidebar.multiselect("Municipio", mun_all, default=(mun_all if sel_corr else []))

# 3) Dimensión
if sel_mun:
    dim_all = sorted(df[df["municipio"].isin(sel_mun)]["dimension"].dropna().unique().tolist())
else:
    dim_all = sorted(df["dimension"].dropna().unique().tolist())
sel_dim = st.sidebar.multiselect("Dimensión", dim_all, default=[])

# 4) Sub-dimensión
if sel_dim:
    tmp_sub = df[df["dimension"].isin(sel_dim)]
    if sel_mun: tmp_sub = tmp_sub[tmp_sub["municipio"].isin(sel_mun)]
else:
    tmp_sub = df[df["municipio"].isin(sel_mun)] if sel_mun else df
subdim_all = sorted(tmp_sub["sub_dimension"].dropna().unique().tolist())
sel_subdim = st.sidebar.multiselect("Sub-dimensión", subdim_all, default=[])

# 5) Categoría
tmp_cat = tmp_sub[tmp_sub["sub_dimension"].isin(sel_subdim)] if sel_subdim else tmp_sub
cat_all = sorted(tmp_cat["category"].dropna().unique().tolist())
sel_cat = st.sidebar.multiselect("Categoría", cat_all, default=[])

# 6) Tipo de lugar
tmp_ptype = tmp_cat[tmp_cat["category"].isin(sel_cat)] if sel_cat else tmp_cat
ptype_all = sorted(tmp_ptype["place_type"].dropna().unique().tolist())
sel_ptype = st.sidebar.multiselect("Tipo de lugar", ptype_all, default=[])

st.sidebar.markdown("---")
show_markers = st.sidebar.checkbox("Mostrar marcadores", True)
show_heatmap = st.sidebar.checkbox("Mostrar mapa de calor", False)

# ======= MINI-LEYENDA EN SIDEBAR (por categoría) =======
with st.sidebar.expander("Leyenda de categorías", expanded=False):
    for cat, (ico, col) in CATEGORY_MARKERS.items():
        st.markdown(
            f"""<div class="legend-row"><span class="legend-badge" style="background:{col};"></span>{cat}</div>""",
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

# =============  HEADER  =============
st.title("Oferta turística — Macizo Colombiano (Cauca)")
st.markdown(
    "Panel interactivo de **servicios y atractivos turísticos** identificados vía Google Maps, "
    "en municipios del **Macizo Colombiano (Cauca)**. "
    "Jerarquía de filtros: **Dimensión → Sub-dimensión → Categoría → Tipo de lugar**. "
    "Los marcadores del mapa son **únicos por categoría** (icono + color)."
)

# =============  CONTENIDO  =============
if ready_to_plot and not fdf.empty:
    # -------- Tabla --------
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

    # -------- Mapa --------
    mlat, mlon = fdf["latitude"].mean(), fdf["longitude"].mean()
    fmap = folium.Map([mlat, mlon], zoom_start=9, control_scale=True)
    Fullscreen(position="topright", title="Pantalla completa", title_cancel="Salir").add_to(fmap)

    if show_heatmap:
        HeatMap(fdf[["latitude","longitude"]].values.tolist(), radius=12, blur=15).add_to(fmap)

    if show_markers:
        for _, r in fdf.iterrows():
            cat = (r.get("category") or "").strip()
            icon_name, color = marker_of(cat)

            name = r.get("name", "Sin nombre")
            muni = r.get("municipio", "No Info")
            dim  = r.get("dimension", "")
            subd = r.get("sub_dimension", "")
            ptype = r.get("place_type", "")
            rating = r.get("average_rating", "No Info")
            reviews = r.get("user_ratings_total", 0)
            link = r.get("map_link", "")

            html = (
                f"<b>{name}</b>"
                f"<br>Municipio: {muni}"
                f"<br>Dimensión: {dim}"
                f"<br>Sub-dimensión: {subd}"
                f"<br>Categoría: {cat}"
                f"<br>Tipo de lugar: {ptype}"
                f"<br>Rating: {rating} ({reviews} reviews)"
            )
            if link:
                html += f"<br><a href='{link}' target='_blank'>Ver en Google</a>"

            folium.Marker(
                [r["latitude"], r["longitude"]],
                icon=folium.Icon(icon=icon_name, color=color, prefix="fa"),
                tooltip=name,
                popup=html
            ).add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    st_folium(fmap, height=650, use_container_width=True)

elif ready_to_plot and fdf.empty:
    st.warning("No se encontraron resultados con los filtros seleccionados. Ajuste la segmentación.")
else:
    st.info("Para visualizar resultados, seleccione al menos una **Dimensión** (y refine con Sub-dimensión, Categoría y Tipo de lugar).")