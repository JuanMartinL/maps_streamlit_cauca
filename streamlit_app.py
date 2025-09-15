# streamlit_app.py
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import itertools
from folium.plugins import HeatMap, Fullscreen
from PIL import Image
import base64
from io import BytesIO
import os

# --------------------------------------------------
# ░S░E░T░U░P░   (wide page)
# --------------------------------------------------
st.set_page_config(layout="wide",
                   page_title="Oferta Turística — Macizo Colombiano (Cauca)",
                   page_icon="🏔️")

# --------------------------------------------------
# ░S░T░Y░L░E░S░
# --------------------------------------------------
st.markdown(
    """
    <style>
        .stMultiSelect [data-baseweb="tag"]      {background:#9c3675!important;}
        .stMultiSelect [data-baseweb="tag"] div  {color:white!important;}
        .stMultiSelect>div                        {border-color:#9c3675!important;}
        input[type="checkbox"]+div svg           {color:#9c3675!important;stroke:#fff!important;fill:#9c3675!important;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# LOGOS  (sidebar)
# --------------------------------------------------
try:
    mincit_logo = Image.open("assets/logo_mincit_fontur.jpeg")
    icon_datad  = Image.open("assets/datad_logo.jpeg")
    _buff = BytesIO(); icon_datad.save(_buff, format="PNG")
    icon_b64 = base64.b64encode(_buff.getvalue()).decode()
except Exception:
    mincit_logo = None
    icon_b64 = ""

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
    if mincit_logo: st.image(mincit_logo, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    if icon_b64:
        st.markdown(
            f"""<div class="powered"><img src="data:image/png;base64,{icon_b64}"/><span>Powered by DataD</span></div>""",
            unsafe_allow_html=True,
        )

# --------------------------------------------------
# ░I░C░O░N░  M░A░P░
# (fallbacks por tipo de lugar; puedes ampliar si lo deseas)
# --------------------------------------------------
icon_map = {
    "restaurant": ("utensils", "blue"),
    "lodging": ("bed", "green"),
    "park": ("tree", "darkgreen"),
    "viewpoint": ("binoculars", "purple"),
    "museum": ("university", "orange"),
    "church": ("church", "cadetblue"),
    "market": ("shopping-basket", "darkred"),
    "default": ("map-marker", "gray"),
}

# --------------------------------------------------
# ░D░A░T░A░
# --------------------------------------------------
@st.cache_data
def load_data():
    # Ruta principal
    primary = "map_data.csv"
    # Fallback opcional si lo mantienes en una subcarpeta:
    alt = "datain/map_data.csv"
    path = primary if os.path.exists(primary) else alt
    df = pd.read_csv(path)

    # Normalizaciones suaves para evitar KeyError si faltan columnas
    for col, default in [
        ("average_rating", "No Info"),
        ("user_ratings_total", 0),
        ("municipio", "No Info"),
        ("dimension", None),
        ("sub_dimension", None),
        ("category", None),
        ("place_type", None),
        ("latitude", None),
        ("longitude", None),
        ("name", None),
    ]:
        if col not in df.columns:
            df[col] = default

    # asegurar tipos
    df["average_rating"] = df["average_rating"].fillna("No Info").astype(str)
    df["user_ratings_total"] = pd.to_numeric(df["user_ratings_total"], errors="coerce").fillna(0).astype(int)
    df["municipio"] = df["municipio"].fillna("No Info")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # columna con link a Google (prioriza las existentes)
    link_cols = [c for c in ["place_link", "google_maps_url", "gmaps_url_canonical", "url"] if c in df.columns]
    df["map_link"] = df[link_cols[0]] if link_cols else ""

    # corredor puede no venir; si no, crear vacío
    if "corredor" not in df.columns:
        df["corredor"] = "Macizo Colombiano"

    return df.dropna(subset=["latitude", "longitude"])

df = load_data()

# --------------------------------------------------
# ░S░I░D░E░B░A░R░  F░I░L░T░E░R░S░
# (nuevo esquema: dimension, sub_dimension, category, place_type)
# --------------------------------------------------
st.sidebar.title("Filtros geográficos y de contenido")

# 1️⃣ Corredor
corr_all = sorted(df["corredor"].dropna().unique())
sel_corr = st.sidebar.multiselect("Seleccione corredor(es)", corr_all, default=[])

# 2️⃣ Municipios
if sel_corr:
    mun_all = sorted(df[df["corredor"].isin(sel_corr)]["municipio"].dropna().unique())
else:
    mun_all = sorted(df["municipio"].dropna().unique())
sel_mun = st.sidebar.multiselect("Seleccione municipios", mun_all, default=mun_all if sel_corr else [])

# 3️⃣ Dimensión
if sel_mun:
    dim_all = sorted(df[df["municipio"].isin(sel_mun)]["dimension"].dropna().unique())
else:
    dim_all = sorted(df["dimension"].dropna().unique())
sel_dim = st.sidebar.multiselect("Seleccione la dimensión", dim_all, default=[])

# 4️⃣ Categoría (dentro de la dimensión)
if sel_dim:
    tmp = df[df["dimension"].isin(sel_dim)]
    cat_all = sorted(tmp["category"].dropna().unique())
else:
    tmp = df
    cat_all = sorted(df["category"].dropna().unique())
sel_cat = st.sidebar.multiselect("Seleccione la categoría", cat_all, default=[])

# 5️⃣ Sub-dimensión (opcional)
tmp2 = tmp[tmp["category"].isin(sel_cat)] if sel_cat else tmp
subdim_all = sorted(tmp2["sub_dimension"].dropna().unique())
sel_subdim = st.sidebar.multiselect("Seleccione la sub-dimensión", subdim_all, default=[])

# 6️⃣ Tipo de lugar (place_type) (opcional)
tmp3 = tmp2[tmp2["sub_dimension"].isin(sel_subdim)] if sel_subdim else tmp2
ptype_all = sorted(tmp3["place_type"].dropna().unique())
sel_ptype = st.sidebar.multiselect("Seleccione el tipo de lugar", ptype_all, default=[])

st.sidebar.markdown("----")
show_markers = st.sidebar.checkbox("Mostrar marcadores", True)
show_heatmap = st.sidebar.checkbox("Mostrar mapa de calor", False)

# --------------------------------------------------
# ░F░I░L░T░E░R░  D░A░T░A░
# --------------------------------------------------
fdf = df.copy()
if sel_corr:   fdf = fdf[fdf["corredor"].isin(sel_corr)]
if sel_mun:    fdf = fdf[fdf["municipio"].isin(sel_mun)]
if sel_dim:    fdf = fdf[fdf["dimension"].isin(sel_dim)]
if sel_cat:    fdf = fdf[fdf["category"].isin(sel_cat)]
if sel_subdim: fdf = fdf[fdf["sub_dimension"].isin(sel_subdim)]
if sel_ptype:  fdf = fdf[fdf["place_type"].isin(sel_ptype)]

# Para habilitar el render pedimos al menos dimensión seleccionada (como antes con info_type)
ready_to_plot = bool(sel_dim)

# --------------------------------------------------
# ░M░A░I░N░  C░O░N░T░E░N░T░
# --------------------------------------------------
st.title("Oferta turística — Macizo Colombiano (Cauca)")

st.markdown(
    "Visualizador interactivo de **servicios y atractivos turísticos** en municipios del "
    "**Macizo Colombiano (Cauca)**, a partir de búsquedas en Google Maps. "
    "Use los filtros de la izquierda en secuencia: **corredor → municipios → dimensión → categoría → sub-dimensión → tipo de lugar**."
)

# Nota: Eliminamos la capa de termales priorizados según tu instrucción.

if ready_to_plot and not fdf.empty:
    # ---------------- Tabla & descarga ----------------
    st.markdown("### Tabla de datos filtrados")
    show_cols = [
        "name","municipio","dimension","category","sub_dimension","place_type",
        "average_rating","user_ratings_total","latitude","longitude"
    ]
    existing_cols = [c for c in show_cols if c in fdf.columns]
    table = fdf[existing_cols].sort_values(by=["average_rating"], ascending=False, key=lambda s: pd.to_numeric(s, errors="coerce"))
    st.dataframe(table, use_container_width=True)
    st.download_button(
        "Descargar CSV",
        data=table.to_csv(index=False),
        file_name="oferta_turistica_filtrada.csv",
        mime="text/csv"
    )

    # ---------------- Mapa ----------------
    # ciclo de colores por place_type (o sub_dimension si place_type vacío)
    color_basis = fdf["place_type"].fillna(fdf["sub_dimension"])
    color_cycle = itertools.cycle(["blue","green","red","orange","purple","darkred","cadetblue","pink","darkblue","lightgray"])
    current_colors = {s: next(color_cycle) for s in color_basis.dropna().unique()}

    # centro del mapa
    mlat, mlon = fdf["latitude"].mean(), fdf["longitude"].mean()
    fmap = folium.Map([mlat, mlon], zoom_start=9, control_scale=True)
    Fullscreen(position="topright", title="Pantalla completa", title_cancel="Salir").add_to(fmap)

    # heatmap
    if show_heatmap:
        HeatMap(fdf[["latitude","longitude"]].values.tolist(), radius=12, blur=15).add_to(fmap)

    # markers
    if show_markers:
        for _, row in fdf.iterrows():
            # escoger clave para color
            color_key = row["place_type"] if pd.notna(row["place_type"]) and row["place_type"] != "" else row["sub_dimension"]
            color = current_colors.get(color_key, "gray")

            # icono por place_type (si coincide con alguno base), si no default
            itype = str(row.get("place_type") or "").lower()
            icon_name, default_color = icon_map.get(itype, icon_map["default"])

            name = row.get("name", "Sin nombre")
            tool = f"{name}"
            rating = row.get("average_rating", "No Info")
            reviews = row.get("user_ratings_total", 0)
            muni = row.get("municipio", "No Info")
            dim = row.get("dimension", "")
            cat = row.get("category", "")
            subd = row.get("sub_dimension", "")
            ptype = row.get("place_type", "")
            link = row.get("map_link", "")

            html = (
                f"<b>{name}</b>"
                f"<br>Municipio: {muni}"
                f"<br>Dimensión: {dim}"
                f"<br>Categoría: {cat}"
                f"<br>Sub-dimensión: {subd}"
                f"<br>Tipo de lugar: {ptype}"
                f"<br>Rating: {rating} ({reviews} reviews)"
            )
            if link:
                html += f"<br><a href='{link}' target='_blank'>Ver en Google</a>"

            folium.Marker(
                [row["latitude"], row["longitude"]],
                icon=folium.Icon(icon=icon_name, color=color, prefix="fa"),
                tooltip=tool,
                popup=html
            ).add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    st_folium(fmap, height=650, use_container_width=True)

else:
    st.info("Seleccione corredor, municipios y **dimensión** para mostrar el mapa y la tabla de datos.")

# V1.1
