import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import itertools
from folium.plugins import HeatMap, Fullscreen
from PIL import Image
import base64
from io import BytesIO

# --------------------------------------------------
# ░S░E░T░U░P░   (wide page to avoid skinny map)
# --------------------------------------------------
st.set_page_config(layout="wide", page_title="Mapa Termales", page_icon="🌋")

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
mincit_logo = Image.open("assets/logo_mincit_fontur.jpeg")
icon_datad  = Image.open("assets/datad_logo.jpeg")
_buff = BytesIO(); icon_datad.save(_buff, format="PNG")
icon_b64 = base64.b64encode(_buff.getvalue()).decode()

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
    st.image(mincit_logo, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="powered"><img src="data:image/png;base64,{icon_b64}"/><span>Powered by DataD</span></div>
        """,
        unsafe_allow_html=True,
    )

# --------------------------------------------------
# ░I░C░O░N░  M░A░P░
# --------------------------------------------------
icon_map = {
    "Turismo Termal y Balnearios":("spa","green"),"Restaurantes":("utensils","blue"),
    "default":("map-marker","gray")
}

# --------------------------------------------------
# ░D░A░T░A░
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("datain/map_data.csv")

@st.cache_data
def load_termales():
    df = pd.read_csv("datain/termales_priorizados.csv", encoding="ISO-8859-1")
    latlon = df["Georreferenciación"].str.split(",", expand=True)
    df["latitude"] = latlon[0].astype(float)
    df["longitude"] = latlon[1].astype(float)
    return df[["Centro Termal","Municipio","Priorizado","latitude","longitude"]].dropna(subset=["latitude"])

df        = load_data()
termales  = load_termales()

# clean
df["average_rating"]      = df["average_rating"].fillna("No Info").astype(str)
df["user_ratings_total"]  = df["user_ratings_total"].fillna(0).astype(int)
df["municipio"]           = df["municipio"].fillna("No Info")

# --------------------------------------------------
# ░S░I░D░E░B░A░R░  F░I░L░T░E░R░S░
# --------------------------------------------------
st.sidebar.title("Filtros Geográficos y de Contenido")

# 1️⃣ Corredor
corr_all = sorted(df["corredor"].dropna().unique())
sel_corr = st.sidebar.multiselect("Seleccione corredor(es)", corr_all, default=[])

# 2️⃣ Municipios (appear after corredor)
if sel_corr:
    mun_all = sorted(df[df["corredor"].isin(sel_corr)]["municipio"].dropna().unique())
    sel_mun = st.sidebar.multiselect("Seleccione municipios", mun_all, default=mun_all)
else:
    sel_mun = []

# 3️⃣ info_type (after municipios)
if sel_mun:
    info_all = sorted(df[df["municipio"].isin(sel_mun)]["info_type"].dropna().unique())
    sel_info = st.sidebar.multiselect("Seleccione la categoría", info_all, default=[])
else:
    sel_info = []

# 4️⃣ category & subcategory (after info_type)
if sel_info:
    tmp = df[df["info_type"].isin(sel_info)]
    cat_all = sorted(tmp["category"].dropna().unique())
    sel_cat = st.sidebar.multiselect("Seleccione la sub-categoría", cat_all, default=[])
    tmp2 = tmp[tmp["category"].isin(sel_cat)] if sel_cat else tmp
    sub_all = sorted(tmp2["sub_category"].dropna().unique())
    sel_sub = st.sidebar.multiselect("Seleccione las actividades turísticas", sub_all, default=[])
else:
    sel_cat = []
    sel_sub = []

st.sidebar.markdown("----")
show_markers = st.sidebar.checkbox("Mostrar marcadores", True)
show_heatmap = st.sidebar.checkbox("Mostrar mapa de calor", False)

# --------------------------------------------------
# ░F░I░L░T░E░R░  D░A░T░A░
# --------------------------------------------------
fdf = df.copy()
if sel_corr: fdf = fdf[fdf["corredor"].isin(sel_corr)]
if sel_mun:  fdf = fdf[fdf["municipio"].isin(sel_mun)]
if sel_info: fdf = fdf[fdf["info_type"].isin(sel_info)]
if sel_cat:  fdf = fdf[fdf["category"].isin(sel_cat)]
if sel_sub:  fdf = fdf[fdf["sub_category"].isin(sel_sub)]

# Flag to show viz only after info_type chosen
ready_to_plot = bool(sel_info)

# --------------------------------------------------
# ░M░A░I░N░  C░O░N░T░E░N░T░
# --------------------------------------------------
st.title("Mapa interactivo de productos turísticos")

st.markdown("Este mapa muestra lugares de interés relacionados con infraestructura turística alrededor de aguas termales.")
st.markdown("Use los filtros a la izquierda en secuencia para explorar.: **corredor → municipios → categoría** para visualizar datos.")

if ready_to_plot and not fdf.empty:
    # ---------------- Table & download ----------------
    st.markdown("### Tabla de datos filtrados")
    show_cols = [
        "name","municipio","sub_category","types","average_rating","user_ratings_total","latitude","longitude"
    ]
    table = fdf[show_cols].sort_values(by="average_rating", ascending=False)
    st.dataframe(table, use_container_width=True)
    st.download_button(
        "Descargar CSV",
        data=table.to_csv(index=False),
        file_name="datos_filtrados.csv",
        mime="text/csv"
    )
    
    # color map per current subs
    color_cycle = itertools.cycle(["blue","green","red","orange","purple","darkred","cadetblue","pink"])
    current_colors = {s: next(color_cycle) for s in (fdf["sub_category"].unique())}

    # center map
    mlat, mlon = fdf["latitude"].mean(), fdf["longitude"].mean()
    fmap = folium.Map([mlat, mlon], zoom_start=11, control_scale=True)
    Fullscreen(position="topright", title="Pantalla completa", title_cancel="Salir").add_to(fmap)

    # heatmap
    if show_heatmap:
        HeatMap(fdf[["latitude","longitude"]].values.tolist(), radius=12, blur=15).add_to(fmap)

    # markers
    if show_markers:
        for _, row in fdf.iterrows():
            icon_name, default_color = icon_map.get(row["sub_category"], icon_map["default"])
            folium.Marker(
                [row["latitude"], row["longitude"]],
                icon=folium.Icon(icon=icon_name, color=current_colors.get(row["sub_category"], default_color), prefix="fa"),
                tooltip=row["name"],
                popup=(f"<b>{row['name']}</b><br>Municipio: {row['municipio']}<br>Tipo: {row['sub_category']}<br>Rating: {row['average_rating']} ({row['user_ratings_total']} reviews)<br>"  # noqa: E501
                       f"<a href='{row['place_link']}' target='_blank'>Ver en Google</a>")
            ).add_to(fmap)

    # prioritized hot springs
    for _, r in termales.iterrows():
        folium.Marker(
            [r["latitude"], r["longitude"]],
            icon=folium.Icon(color="darkpurple", icon="water", prefix="fa"),
            tooltip=r["Centro Termal"],
            popup=f"<b>{r['Centro Termal']}</b><br>Municipio: {r['Municipio']}<br>Priorizado: {r['Priorizado']}"
        ).add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    st_folium(fmap, height=600, use_container_width=True)


else:
    st.info("Seleccione corredor, municipios e info_type para mostrar el mapa y la tabla de datos.")
