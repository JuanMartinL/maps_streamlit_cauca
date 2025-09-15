# streamlit_app.py

# streamlit_app.py
# --------------------------------------------------
# Oferta Turística — Macizo Colombiano (Cauca)
# Jerarquía: dimension > sub_dimension > category > place_type
# Marcadores únicos por place_type (icono + color)
# --------------------------------------------------

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap, Fullscreen
from PIL import Image
from io import BytesIO
import base64
import os
import unicodedata

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

# =============  CONTENIDO  =============
st.title("Oferta turística — Macizo Colombiano (Cauca)")
st.markdown(
    "Panel interactivo de **servicios y atractivos turísticos** identificados vía Google Maps, "
    "en municipios del **Macizo Colombiano (Cauca)**. "
    "Jerarquía: **Dimensión → Sub-dimensión → Categoría → Tipo de lugar**. "
    "Los marcadores son **únicos por tipo de lugar**."
)

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
