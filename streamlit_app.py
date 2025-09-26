# streamlit_app.py
# --------------------------------------------------
# Oferta Turística — Macizo Colombiano (Cauca)
# Tab 1: Mapa
# Tab 2: Text mining 
# --------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap, Fullscreen
from PIL import Image
from io import BytesIO
import base64, os, unicodedata, re
from collections import Counter
import matplotlib.pyplot as plt

# Optional dependency
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False

# =============  PAGE SETUP  =============
st.set_page_config(page_title="Oferta Turística — Macizo Colombiano (Cauca)", page_icon="🏔️", layout="wide")

# =============  STYLES  =============
st.markdown("""
<style>
.stMultiSelect [data-baseweb="tag"]{background:#9c3675!important;}
.stMultiSelect [data-baseweb="tag"] div{color:white!important;}
.stMultiSelect>div{border-color:#9c3675!important;}
input[type="checkbox"]+div svg{color:#9c3675!important;stroke:#fff!important;fill:#9c3675!important;}
.legend-badge{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:middle;}
.legend-row{font-size:12px;margin-bottom:6px;}
</style>
""", unsafe_allow_html=True)

# ======== LOGOS (sidebar) ========
def _load_b64(path: str):
    try:
        from PIL import Image
        from io import BytesIO
        import base64
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
            .brand img{display:block;margin:6px auto 14px auto;}
            .powered{display:flex;justify-content:center;align-items:center;gap:8px;font-size:11px;color:grey;}
            .powered img{height:45px;width:45px;border-radius:50%;object-fit:cover;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ÚNICA imagen compuesta de logos
    if os.path.exists("assets/logos.png"):
        st.markdown('<div class="brand">', unsafe_allow_html=True)
        st.image("assets/logos.png", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Powered by DataD
    b64 = _load_b64("assets/datad_logo.jpeg")
    if b64:
        st.markdown(
            f"""<div class="powered">
                    <img src="data:image/png;base64,{b64}"/>
                    <span>Powered by DataD</span>
                </div>""",
            unsafe_allow_html=True,
        )


# =============  HELPERS  =============
def _norm(s: str) -> str:
    s = str(s or "").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\s+", " ", s)
    return s

# Marcadores por place_type (igual que antes; se omite lista larga por brevedad)
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

def make_popup_html(r: dict) -> str:
    link = r.get("map_link","")
    link_html = f"<br><b>Google:</b> <a href='{link}' target='_blank'>Ver en Google</a>" if link else ""
    return (f"<div style='font-size:13px'><b>{r.get('name','Sin nombre')}</b>"
            f"<br><b>Municipio:</b> {r.get('municipio','No Info')}"
            f"<br><b>Dimensión:</b> {r.get('dimension','')}"
            f"<br><b>Sub-dimensión:</b> {r.get('sub_dimension','')}"
            f"<br><b>Categoría:</b> {r.get('category','')}"
            f"<br><b>Tipo de lugar:</b> {r.get('place_type','')}"
            f"<br><b>Rating:</b> {r.get('average_rating','No Info')} ({r.get('user_ratings_total',0)} reviews)"
            f"{link_html}</div>")

# =============  DATA LOADERS  =============
@st.cache_data
def _read_csv_any(name: str) -> pd.DataFrame:
    if os.path.exists(name): return pd.read_csv(name)
    alt = os.path.join("datain", name)
    if os.path.exists(alt):  return pd.read_csv(alt)
    return pd.DataFrame()

@st.cache_data
def load_map() -> pd.DataFrame:
    df = _read_csv_any("map_data.csv")
    if df.empty: return df
    for col, default in [
        ("municipio","No Info"),("corredor","Macizo Colombiano"),
        ("dimension",None),("sub_dimension",None),("category",None),
        ("place_type",None),("name",None),("average_rating","No Info"),
        ("user_ratings_total",0),("latitude",None),("longitude",None),("map_link","")
    ]:
        if col not in df.columns: df[col] = default
    df["average_rating"] = df["average_rating"].fillna("No Info").astype(str)
    df["user_ratings_total"] = pd.to_numeric(df["user_ratings_total"], errors="coerce").fillna(0).astype(int)
    df["latitude"]  = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["mun_norm"] = df["municipio"].map(_norm)
    df["cat_norm"] = df["category"].map(_norm)
    return df.dropna(subset=["latitude","longitude"])

@st.cache_data
@st.cache_data
def load_reviews() -> pd.DataFrame:
    df = _read_csv_any("map_data_review.csv")
    if df.empty: return df

    # Normaliza nombres
    rnm = {}
    for c in ["text_es","texto_es","review_text_es","review_text","text","comentario"]:
        if c in df.columns: rnm[c]="text_es"; break
    for c in ["municipio","Municipio","muni","city","town"]:
        if c in df.columns: rnm[c]="municipio"; break
    for c in ["category","Category","categoria","cat"]:
        if c in df.columns: rnm[c]="category"; break
    df = df.rename(columns=rnm)

    # Columnas mínimas
    if "text_es" not in df.columns: df["text_es"] = ""
    if "municipio" not in df.columns: df["municipio"] = "No Info"
    if "category"  not in df.columns: df["category"]  = "No Info"

    # >>> CRÍTICO: evita 'nan' como texto
    df["text_es"]  = df["text_es"].astype("string").fillna("")

    # Normalizados para join con mapa
    df["mun_norm"] = df["municipio"].map(_norm)
    df["cat_norm"] = df["category"].map(_norm)
    return df


df_map = load_map()
df_reviews = load_reviews()

# =============  SIDEBAR FILTERS  =============
st.sidebar.title("Filtros geográficos y de contenido")
corr_all = sorted(df_map["corredor"].dropna().unique().tolist())
sel_corr = st.sidebar.multiselect("Corredor", corr_all, default=[])

if sel_corr:
    mun_all = sorted(df_map[df_map["corredor"].isin(sel_corr)]["municipio"].dropna().unique().tolist())
else:
    mun_all = sorted(df_map["municipio"].dropna().unique().tolist())
sel_mun = st.sidebar.multiselect("Municipio", mun_all, default=(mun_all if sel_corr else []))

if sel_mun:
    dim_all = sorted(df_map[df_map["municipio"].isin(sel_mun)]["dimension"].dropna().unique().tolist())
else:
    dim_all = sorted(df_map["dimension"].dropna().unique().tolist())
sel_dim = st.sidebar.multiselect("Dimensión", dim_all, default=[])

if sel_dim:
    tmp_sub = df_map[df_map["dimension"].isin(sel_dim)]
    if sel_mun: tmp_sub = tmp_sub[tmp_sub["municipio"].isin(sel_mun)]
else:
    tmp_sub = df_map[df_map["municipio"].isin(sel_mun)] if sel_mun else df_map
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

# =============  FILTERED MAP DF  =============
fdf = df_map.copy()
if sel_corr:   fdf = fdf[fdf["corredor"].isin(sel_corr)]
if sel_mun:    fdf = fdf[fdf["municipio"].isin(sel_mun)]
if sel_dim:    fdf = fdf[fdf["dimension"].isin(sel_dim)]
if sel_subdim: fdf = fdf[fdf["sub_dimension"].isin(sel_subdim)]
if sel_cat:    fdf = fdf[fdf["category"].isin(sel_cat)]
if sel_ptype:  fdf = fdf[fdf["place_type"].isin(sel_ptype)]
ready_map = bool(sel_dim)

# =============  NLP UTILS  =============
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
# --- STOPWORDS (agrega nan/none/null) ---
DOMAIN_STOP = {"lugar", "sitio", "nan", "none", "null"}
BASE_STOP = SPANISH_STOP | DOMAIN_STOP

def normalize_spanish(text: str) -> str:
    s = str(text or "").lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = re.sub(r"[^a-zñü\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def preprocess_docs(series: pd.Series) -> list[str]:
    out = []
    # segunda malla de seguridad contra 'nan'
    for raw in series.fillna("").astype(str):
        if raw.strip().lower() in {"nan", "none", "null"}:
            out.append("")
            continue
        s = normalize_spanish(raw)
        toks = [t for t in s.split() if len(t) > 2 and t not in BASE_STOP]
        out.append(" ".join(toks))
    return out


def adaptive_df(n_docs:int):
    if n_docs>=200: return 3,0.60
    if n_docs>=50:  return 2,0.75
    return 1,1.0

def build_freq(docs:list[str], topk:int=400)->dict:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
        if not docs: return {}
        tok_tot = sum(len(d.split()) for d in docs)
        if tok_tot < 25: return {}
        min_df, max_df = adaptive_df(len(docs))
        vecs = [
            TfidfVectorizer(ngram_range=(1,2), min_df=min_df, max_df=max_df, stop_words=list(BASE_STOP), norm=None),
            TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=1.0, stop_words=list(BASE_STOP), norm=None),
            TfidfVectorizer(ngram_range=(1,1), min_df=1, max_df=1.0, stop_words=list(BASE_STOP), norm=None),
            CountVectorizer(ngram_range=(1,2), min_df=1, max_df=1.0, stop_words=list(BASE_STOP))
        ]
        for v in vecs:
            try:
                X = v.fit_transform(docs)
                if X.shape[1]==0: continue
                terms = v.get_feature_names_out()
                weights = np.asarray(X.sum(axis=0)).ravel()
                freq = dict(zip(terms, weights))
                return dict(sorted(freq.items(), key=lambda x:x[1], reverse=True)[:topk])
            except ValueError:
                continue
    except Exception:
        pass
    # fallback manual
    bag=Counter()
    for d in docs:
        t=[w for w in d.split() if w not in BASE_STOP]
        bag.update(t); bag.update([" ".join(p) for p in zip(t,t[1:])])
    return dict(sorted(bag.items(), key=lambda x:x[1], reverse=True)[:topk])

def top_table(freq_map:dict, n=25)->pd.DataFrame:
    if not freq_map: return pd.DataFrame(columns=["term","weight","share_%"])
    s = pd.Series(freq_map).sort_values(ascending=False).head(n)
    df = s.rename_axis("term").reset_index(name="weight")
    df["share_%"] = (100*df["weight"]/df["weight"].sum()).round(2)
    return df

def draw_wordcloud(freq_map:dict, title:str):
    if not WORDCLOUD_AVAILABLE: return None
    wc = WordCloud(width=1400, height=800, background_color="white",
                   prefer_horizontal=0.8, collocations=False, max_words=300,
                   relative_scaling=0.3, min_font_size=10, contour_width=0.8, contour_color="grey"
                  ).generate_from_frequencies(freq_map)
    fig = plt.figure(figsize=(12,7), dpi=140); plt.imshow(wc, interpolation="bilinear")
    plt.axis("off"); plt.title(title, pad=8); return fig

def draw_bar_chart(freq_map:dict, title:str, topn:int=40):
    if not freq_map: st.info(f"{title}: sin términos para graficar."); return
    df = pd.Series(freq_map).sort_values(ascending=False).head(topn).rename_axis("term").reset_index(name="weight")
    fig = plt.figure(figsize=(10,8), dpi=140); plt.barh(df["term"][::-1], df["weight"][::-1])
    plt.title(title); plt.xlabel("Peso"); plt.tight_layout(); st.pyplot(fig, use_container_width=True)

# =============  CONTENT  =============
st.title("Oferta turística — Macizo Colombiano (Cauca)")
st.markdown("Panel interactivo de **servicios y atractivos turísticos** identificados vía Google Maps, en municipios del **Macizo Colombiano (Cauca)**.")

tab_map, tab_text = st.tabs(["Mapa", "Text mining"])

# ---------------- TAB 1: MAPA ----------------
with tab_map:
    if ready_map and not fdf.empty:
        st.markdown("### Resultados filtrados")
        show_cols = ["name","municipio","dimension","sub_dimension","category","place_type",
                     "average_rating","user_ratings_total","latitude","longitude","map_link"]
        show_cols = [c for c in show_cols if c in fdf.columns]
        table = fdf.copy()
        if "average_rating" in table.columns:
            table["_avg_num"] = pd.to_numeric(table["average_rating"], errors="coerce")
            table = table.sort_values(by=["_avg_num"], ascending=False).drop(columns=["_avg_num"], errors="ignore")
        st.dataframe(table[show_cols], use_container_width=True)
        st.download_button("Descargar CSV", data=table[show_cols].to_csv(index=False),
                           file_name="oferta_turistica_filtrada.csv", mime="text/csv")

        mlat, mlon = fdf["latitude"].mean(), fdf["longitude"].mean()
        fmap = folium.Map([mlat, mlon], zoom_start=9, control_scale=True)
        Fullscreen(position="topright", title="Pantalla completa", title_cancel="Salir").add_to(fmap)
        if show_heatmap:
            HeatMap(fdf[["latitude","longitude"]].values.tolist(), radius=12, blur=15).add_to(fmap)
        if show_markers:
            for _, r in fdf.iterrows():
                icon_name, color = marker_of_place_type(r.get("place_type"))
                folium.Marker(
                    [r["latitude"], r["longitude"]],
                    icon=folium.Icon(icon=icon_name, color=color, prefix="fa"),
                    tooltip=r.get("name","Sin nombre"),
                    popup=folium.Popup(make_popup_html(r), max_width=320)
                ).add_to(fmap)
        folium.LayerControl(collapsed=False).add_to(fmap)
        st_folium(fmap, height=650, use_container_width=True)

    elif ready_map and fdf.empty:
        st.warning("No se encontraron resultados con los filtros seleccionados. Ajuste la segmentación.")
    else:
        st.info("Para visualizar resultados, seleccione al menos una **Dimensión**.")

# ---------------- TAB 2: TEXT MINING (UN solo WordCloud) ----------------
with tab_text:
    st.subheader("Insights de texto — WordCloud y conteo (según filtros activos)")

    if df_reviews.empty:
        st.warning("No se encontró `map_data_review.csv`. Colóquelo en la raíz o en `./datain/`.")
    else:
        # 1) Determinar el subconjunto de reviews *derivado* de los filtros del mapa
        #    - Siempre filtramos por los municipios presentes en el mapa filtrado
        #    - Categorías: si el filtro de mapa dejó categorías activas, usamos esas; si no, usamos todas las del mapa filtrado
        if ready_map:
            muni_set = set(fdf["mun_norm"].unique())
            cat_set  = set(fdf["cat_norm"].unique())
        else:
            muni_set = set(df_map["mun_norm"].unique())
            cat_set  = set(df_map["cat_norm"].unique())

        rdf = df_reviews.copy()
        if muni_set:
            rdf = rdf[rdf["mun_norm"].isin(muni_set)]
        if cat_set:
            rdf = rdf[rdf["cat_norm"].isin(cat_set)]

        # 2) Guardrails
        if rdf.empty or rdf["text_es"].fillna("").str.strip().eq("").all():
            st.warning("No hay texto disponible en los registros que cumplen los filtros.")
        else:
            # 3) Preprocess + build frequencies
            topk = st.slider("Top términos para tablas", 10, 100, 25, 5)
            docs = preprocess_docs(rdf["text_es"])
            n_docs = sum(1 for d in docs if d)
            token_total = sum(len(d.split()) for d in docs)

            if n_docs < 5 or token_total < 25:
                st.info(f"Muestra insuficiente para nube (docs={n_docs}, tokens={token_total}). Amplíe filtros.")
            else:
                freq = build_freq(docs, topk=400)
                title = "Reviews — según filtros activos"

                # 4) ONE WordCloud (o barras si no hay lib)
                fig = draw_wordcloud(freq, title)
                if fig is not None:
                    st.pyplot(fig, use_container_width=True)
                else:
                    st.info("WordCloud no disponible en el entorno; mostrando ranking de términos.")
                    draw_bar_chart(freq, title)

                # 5) Tablas de apoyo (unigramas / bigramas)
                uni = {k:v for k,v in freq.items() if " " not in k}
                bi  = {k:v for k,v in freq.items() if " " in k}
                df_uni = top_table(uni, n=topk)
                df_bi  = top_table(bi,  n=topk)

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Top Unigramas**")
                    st.dataframe(df_uni, use_container_width=True, height=320)
                    st.download_button("Descargar unigramas (CSV)",
                        data=df_uni.to_csv(index=False),
                        file_name="reviews_filtros_top_unigrams.csv",
                        mime="text/csv")
                with c2:
                    st.markdown("**Top Bigramas**")
                    st.dataframe(df_bi, use_container_width=True, height=320)
                    st.download_button("Descargar bigramas (CSV)",
                        data=df_bi.to_csv(index=False),
                        file_name="reviews_filtros_top_bigrams.csv",
                        mime="text/csv")
