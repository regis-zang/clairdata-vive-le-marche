
# app_vive_le_marche.py
# Streamlit dashboard - Vive le Marché (fatos + dimensões com labels em francês)
# Execução (exemplo):
#   streamlit run app_vive_le_marche.py -- --data_root "I:/Projetos_Python/clairdata-vive-le-marche/data"
#
# Requisitos (ambiente vlm311):
#   conda install -y pandas pyarrow plotly fastparquet streamlit

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Dict
import argparse

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# ============================================
# Config
# ============================================
st.set_page_config(page_title="Vive le Marché — Dashboard (FR)", layout="wide")

DIM_TARGETS = ["Ano", "GS", "NA5", "NA10", "ZELT", "REGLT", "SEXE", "ANAI"]
METRIC = "QTD"
DESC_OUT = {
    "Ano": "ano_desc_fr",
    "GS": "gs_desc_fr",
    "NA5": "na5_desc_fr",
    "NA10": "na10_desc_fr",
    "ZELT": "zelt_desc_fr",
    "REGLT": "reglt_desc_fr",
    "SEXE": "sexe_desc_fr",
    "ANAI": "anai_desc_fr",
}
DIM_FILENAMES = {
    "Ano":   ["dim_ano", "Dim_ANO", "Dim_Ano", "dim_Ano"],
    "GS":    ["dim_gs", "Dim_GS"],
    "NA5":   ["dim_na5", "Dim_NA5"],
    "NA10":  ["dim_na10", "Dim_NA10"],
    "ZELT":  ["dim_zelt", "Dim_ZELT"],
    "REGLT": ["dim_reglt", "Dim_REGLT"],
    "SEXE":  ["dim_sexe", "Dim_SEXE"],
    "ANAI":  ["dim_anai", "Dim_ANAI"],
}
GEO_FILES = {
    "REGLT": ["fr_regions.geojson", "regions.geojson", "reglt.geojson"],
    "ZELT":  ["fr_zones_emploi.geojson", "zelt.geojson", "zones_emploi.geojson"],
}

# ============================================
# Helpers
# ============================================
def _read_any(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    if path.suffix.lower() == ".parquet":
        for eng in ("pyarrow", "fastparquet"):
            try:
                return pd.read_parquet(path, engine=eng)
            except Exception:
                continue
        return None
    if path.suffix.lower() == ".csv":
        for sep in (",", ";", "\t"):
            try:
                return pd.read_csv(path, sep=sep)
            except Exception:
                continue
        return None
    # fallback: tenta parquet mesmo se extensão for estranha
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except Exception:
        return None

def find_dims_root(data_root: Path) -> Path:
    for p in (data_root/"dimensions", data_root/"dimension", data_root/"dimensions_parquet", data_root/"dim"):
        if p.exists():
            return p
    return data_root

def detect_desc_col(df: pd.DataFrame, key: str) -> Optional[str]:
    cl = {c.lower(): c for c in df.columns}
    for k in ("desc_lbl_fr", "desc_fr", f"{key.lower()}_desc_fr",
              "label_fr", "libelle_fr", "libellé_fr",
              "libelle", "libellé", "label", "description", "desc"):
        if k in cl:
            return cl[k]
    return None

def detect_key_col(df: pd.DataFrame, key: str) -> Optional[str]:
    cl = {c.lower(): c for c in df.columns}
    for k in (key.lower(), f"cod_{key.lower()}", "codigo", "code", "id", "key"):
        if k in cl:
            return cl[k]
    # fallback: se houver só uma coluna inteira, usa
    ints = [c for c in df.columns if pd.api.types.is_integer_dtype(df[c])]
    if len(ints) == 1:
        return ints[0]
    return None

@st.cache_data(show_spinner=False)
def load_fact(data_root: Path) -> pd.DataFrame:
    # prioriza versão FR (já denormalizada)
    for name in ("fact_vive_le_marche_fr.parquet", "fact_vive_le_marche.parquet"):
        p = data_root / name
        if p.exists():
            df = _read_any(p)
            if df is not None and not df.empty:
                return df
    st.error("Não encontrei fact_vive_le_marche(_fr).parquet no data_root.")
    st.stop()

@st.cache_data(show_spinner=False)
def load_dimension(dims_root: Path, key: str) -> Optional[pd.DataFrame]:
    # busca por nomes mais comuns
    names = DIM_FILENAMES.get(key, [])
    paths = []
    for base in names:
        paths += [dims_root / f"{base}.parquet", dims_root / f"{base}.csv"]
    # varre diretório como fallback
    paths += list(dims_root.glob(f"*{key}*.parquet"))
    paths += list(dims_root.glob(f"*{key}*.csv"))

    for p in paths:
        if not p.exists(): 
            continue
        df = _read_any(p)
        if df is None or df.empty:
            continue
        kcol = detect_key_col(df, key)
        dcol = detect_desc_col(df, key)
        if not kcol or not dcol:
            continue
        out = df[[kcol, dcol]].drop_duplicates()
        out.columns = [key, DESC_OUT[key]]
        out[key] = out[key].astype(str).str.strip()
        return out
    return None

def enrich_with_dims(df: pd.DataFrame, dims_root: Path) -> pd.DataFrame:
    out = df.copy()
    # força chaves do fato para string
    for k in DIM_TARGETS:
        if k in out.columns:
            out[k] = out[k].astype(str).str.strip()

    for k in DIM_TARGETS:
        if k not in out.columns:
            continue
        dim = load_dimension(dims_root, k)
        if dim is None:
            if k == "Ano" and DESC_OUT[k] not in out.columns:
                out[DESC_OUT[k]] = "Année " + out["Ano"].astype(str)
            continue
        before = len(out)
        out = out.merge(dim, how="left", on=k)
        after = len(out)
        if before != after:
            st.warning(f"Dimensão '{k}' alterou número de linhas ({before} → {after}). Verifique duplicidades na dimensão.")
    return out

def find_geojson(data_root: Path, key: str) -> Optional[Path]:
    geo = data_root / "geo"
    if not geo.exists():
        return None
    for name in GEO_FILES.get(key, []):
        p = geo / name
        if p.exists():
            return p
    # fallback: primeiro .geojson que contenha o nome
    for p in geo.glob("*.geojson"):
        if key.lower() in p.stem.lower():
            return p
    return None

# ============================================
# Args via linha de comando (streamlit run ... -- --data_root X)
# ============================================
def build_parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--data_root", type=str, default=str(Path.cwd() / "data"))
    return p

def get_args() -> argparse.Namespace:
    try:
        args, _ = build_parser().parse_known_args()
    except SystemExit:
        args = build_parser().parse_args([])
    return args

args = get_args()
DATA_ROOT = Path(args.data_root).resolve()
DIMS_ROOT = find_dims_root(DATA_ROOT)

st.sidebar.title("⚙️ Configuração")
st.sidebar.write(f"**Data root:** `{DATA_ROOT}`")
st.sidebar.write(f"**Dims root:** `{DIMS_ROOT}`")

# ============================================
# Load & Enrich
# ============================================
fact_raw = load_fact(DATA_ROOT)
with st.spinner("Fazendo merge das dimensões (FR)…"):
    fact = enrich_with_dims(fact_raw, DIMS_ROOT)
st.success("Dados prontos!")

# ============================================
# Filtros
# ============================================
st.sidebar.subheader("Filtros")
filters: Dict[str, list] = {}

if "Ano" in fact.columns:
    anos = sorted(fact["Ano"].dropna().unique().tolist())
    sel_anos = st.sidebar.multiselect("Ano", anos, default=anos)
    filters["Ano"] = sel_anos

for k in ["GS","NA10","NA5","REGLT","ZELT","SEXE","ANAI"]:
    dcol = DESC_OUT.get(k)
    if k in fact.columns and dcol in fact.columns:
        vals = fact[[k, dcol]].drop_duplicates().sort_values(dcol)
        options = vals[dcol].tolist()
        default = options[:min(10, len(options))]
        sel = st.sidebar.multiselect(f"{k} (FR)", options, default=default)
        if sel:
            codes = vals.loc[vals[dcol].isin(sel), k].astype(str).tolist()
            filters[k] = codes

df_view = fact.copy()
for col, vals in filters.items():
    if vals:
        df_view = df_view[df_view[col].astype(str).isin([str(v) for v in vals])]

# ============================================
# KPIs
# ============================================
st.title("Vive le Marché — Dashboard")
c1, c2, c3 = st.columns(3)
total = int(df_view[METRIC].sum()) if METRIC in df_view.columns else len(df_view)
n_regs = df_view["REGLT"].nunique() if "REGLT" in df_view.columns else 0
n_zelt = df_view["ZELT"].nunique() if "ZELT" in df_view.columns else 0
c1.metric("Total (QTD)", f"{total:,}".replace(",", "."))
c2.metric("Regiões (REGLT) filtradas", n_regs)
c3.metric("Zonas de emprego (ZELT) filtradas", n_zelt)

# ============================================
# Tabela com labels FR
# ============================================
st.subheader("Tabela (labels em francês)")
show_cols = [c for c in [
    "Ano","gs_desc_fr","na10_desc_fr","na5_desc_fr",
    "reglt_desc_fr","zelt_desc_fr","sexe_desc_fr","anai_desc_fr",METRIC
] if c in df_view.columns]
if not show_cols:
    show_cols = df_view.columns.tolist()[:10]
st.dataframe(df_view[show_cols].head(500))

# Pivot por Ano x Dimensão
st.subheader("Pivot por Ano × Dimensão")
dim_choice = st.selectbox("Dimensão", [d for d in ["GS","NA10","NA5","REGLT","ZELT","SEXE","ANAI"] if d in df_view.columns], index=0)
dcol = DESC_OUT.get(dim_choice)
if dcol and dcol in df_view.columns and METRIC in df_view.columns:
    piv = (df_view.groupby(["Ano", dcol], dropna=False)[METRIC]
           .sum().reset_index().pivot(index=dcol, columns="Ano", values=METRIC)
           .fillna(0).astype(int))
    st.dataframe(piv)

# ============================================
# Mapas (GeoJSON local)
# ============================================
def plot_geo(df: pd.DataFrame, key: str, value_col: str) -> None:
    if not PLOTLY_OK:
        st.info("Plotly não disponível neste ambiente.")
        return
    geo_p = find_geojson(DATA_ROOT, key)
    if geo_p is None:
        st.info(f"GeoJSON para {key} não encontrado. Coloque um arquivo em {DATA_ROOT/'geo'} com um destes nomes: {', '.join(GEO_FILES.get(key, []))}.")
        return
    import json
    gj = json.loads(geo_p.read_text(encoding="utf-8"))
    # agrega por código
    df_agg = df.groupby(key, dropna=False)[value_col].sum().reset_index()
    df_agg[key] = df_agg[key].astype(str)
    # detecta propriedade de código no GeoJSON
    feature_key = None
    if "features" in gj and gj["features"]:
        props = gj["features"][0].get("properties", {})
        for pk in ["code","CODE","INSEE_REG","REG","ZE2020","ZE2024",key,key.lower()]:
            if pk in props:
                feature_key = pk
                break
    if not feature_key:
        st.warning(f"Não foi possível detectar a coluna de código no GeoJSON {geo_p.name}.")
        return
    fig = px.choropleth(
        df_agg, geojson=gj, locations=key, featureidkey=f"properties.{feature_key}",
        color=value_col, color_continuous_scale="Blues",
        labels={value_col: METRIC}, title=f"Mapa — {key}"
    )
    fig.update_geos(fitbounds="locations", visible=False)
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Mapas")
colA, colB = st.columns(2)
with colA:
    if "REGLT" in df_view.columns and METRIC in df_view.columns:
        plot_geo(df_view, "REGLT", METRIC)
with colB:
    if "ZELT" in df_view.columns and METRIC in df_view.columns:
        plot_geo(df_view, "ZELT", METRIC)

# ============================================
# Export
# ============================================
st.subheader("Exportar recorte atual")
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, sep=";").encode("utf-8-sig")
st.download_button("Baixar CSV (recorte)", data=to_csv_bytes(df_view[show_cols]), file_name="vive_le_marche_recorte.csv", mime="text/csv")

st.caption("💡 Coloque GeoJSONs em data/geo. O app reconhece automaticamente campos comuns de código (INSEE_REG/REG/ZE2020…).")
