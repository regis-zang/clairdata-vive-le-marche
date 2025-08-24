
# app_vive_le_marche.py — multi-onglets (Adhoc, Tableau de bord, Tableau, Cartes, Aperçu)
# Mapas (aba Cartes) temporariamente desativados — manteremos apenas o placeholder.
from __future__ import annotations
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

st.set_page_config(page_title="Vive le Marché", layout="wide")

# ---------------- Config ----------------
METRIC = "QTD"
FILTER_KEYS = ["Ano", "GS", "NA5", "NA10"]
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

# --------------- IO helpers ---------------
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
    ints = [c for c in df.columns if pd.api.types.is_integer_dtype(df[c])]
    if len(ints) == 1:
        return ints[0]
    return None

@st.cache_data(show_spinner=False)
def load_fact(data_root: Path) -> pd.DataFrame:
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
    names = DIM_FILENAMES.get(key, [])
    paths = []
    for base in names:
        paths += [dims_root / f"{base}.parquet", dims_root / f"{base}.csv"]
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
        out.columns = [key, DESC_OUT.get(key, f"{key.lower()}_desc_fr")]
        out[key] = out[key].astype(str).str.strip()
        return out
    return None

def enrich_with_dims(df: pd.DataFrame, dims_root: Path) -> pd.DataFrame:
    out = df.copy()
    # normaliza chaves
    for k in list(DESC_OUT.keys()):
        if k in out.columns:
            out[k] = out[k].astype(str).str.strip()

    # une dimensões necessárias para filtros/visualizações
    for k in set(FILTER_KEYS + ["REGLT", "ZELT", "SEXE", "ANAI"]):
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

# ------------- Args & Load -------------
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

# ------------- Sidebar (filtros) -------------
st.sidebar.title("⚙️ Configuração")
st.sidebar.write(f"**Data root:** `{DATA_ROOT}`")
st.sidebar.write(f"**Dims root:** `{DIMS_ROOT}`")

fact_raw = load_fact(DATA_ROOT)
with st.spinner("Fazendo merge das dimensões (FR)…"):
    fact = enrich_with_dims(fact_raw, DIMS_ROOT)
st.success("Dados prontos!")

# Filtros compartilhados
st.sidebar.subheader("Filtros")
filters: Dict[str, list] = {}
if "Ano" in fact.columns:
    anos = sorted(fact["Ano"].dropna().unique().tolist())
    sel_anos = st.sidebar.multiselect("Ano", anos, default=anos)
    filters["Ano"] = sel_anos

for k in ["GS","NA10","NA5"]:
    dcol = DESC_OUT.get(k)
    if k in fact.columns and dcol in fact.columns:
        vals = fact[[k, dcol]].drop_duplicates().sort_values(dcol)
        options = vals[dcol].tolist()
        default = options[:min(10, len(options))]
        sel = st.sidebar.multiselect(f"{k} (FR)", options, default=default)
        if sel:
            codes = vals.loc[vals[dcol].isin(sel), k].astype(str).tolist()
            filters[k] = codes

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, vals in filters.items():
        if vals:
            out = out[out[col].astype(str).isin([str(v) for v in vals])]
    return out

df_view = apply_filters(fact)

# ---------------- Tabs ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["1 · Adhoc", "2 · Tableau de bord", "3 · Tableau", "4 · Cartes", "5 · Aperçu"])

# -------- Tab 1: Adhoc --------
with tab1:
    st.header("Adhoc")
    c1, c2, c3 = st.columns(3)
    total = int(df_view[METRIC].sum()) if METRIC in df_view.columns else len(df_view)
    n_gs = df_view["GS"].nunique() if "GS" in df_view.columns else 0
    n_na10 = df_view["NA10"].nunique() if "NA10" in df_view.columns else 0
    c1.metric("Total (QTD)", f"{total:,}".replace(",", "."))
    c2.metric("Segmentos (GS) filtrados", n_gs)
    c3.metric("NA10 filtrados", n_na10)

    st.subheader("Tabela (labels em francês)")
    show_cols = [c for c in ["Ano","gs_desc_fr","na10_desc_fr","na5_desc_fr", METRIC] if c in df_view.columns]
    if not show_cols:
        show_cols = df_view.columns.tolist()[:10]
    st.dataframe(df_view[show_cols].head(1000))

    st.subheader("Pivot por Ano × Dimensão")
    dim_choice = st.selectbox("Dimensão", [d for d in ["GS","NA10","NA5"] if d in df_view.columns], index=0, key="adhoc_dim")
    dcol = DESC_OUT.get(dim_choice)
    if dcol and dcol in df_view.columns and METRIC in df_view.columns:
        piv = (df_view.groupby(["Ano", dcol], dropna=False)[METRIC]
               .sum().reset_index().pivot(index=dcol, columns="Ano", values=METRIC)
               .fillna(0).astype(int))
        st.dataframe(piv)

# -------- Tab 2: Tableau de bord --------
with tab2:
    st.header("Tableau de bord")
    # KPIs: Total, YoY, #GS, #NA10
    k1, k2, k3, k4 = st.columns(4)
    total = int(df_view[METRIC].sum()) if METRIC in df_view.columns else 0
    # YoY: compara último ano do filtro com o anterior dentro do recorte
    yoy_txt = "—"
    if "Ano" in df_view.columns and METRIC in df_view.columns:
        anos_sorted = sorted(pd.to_numeric(df_view["Ano"], errors="coerce").dropna().astype(int).unique().tolist())
        if len(anos_sorted) >= 2:
            last, prev = anos_sorted[-1], anos_sorted[-2]
            v_last = df_view.loc[pd.to_numeric(df_view["Ano"], errors="coerce").astype("Int64")==last, METRIC].sum()
            v_prev = df_view.loc[pd.to_numeric(df_view["Ano"], errors="coerce").astype("Int64")==prev, METRIC].sum()
            if v_prev != 0:
                yoy = (v_last - v_prev) / v_prev * 100
                yoy_txt = f"{yoy:+.1f}%"
            else:
                yoy_txt = "N/A"
    top_gs = "—"
    if "gs_desc_fr" in df_view.columns and METRIC in df_view.columns:
        s = df_view.groupby("gs_desc_fr")[METRIC].sum().sort_values(ascending=False)
        if len(s) > 0:
            top_gs = s.index[0]
    top_na10 = "—"
    if "na10_desc_fr" in df_view.columns and METRIC in df_view.columns:
        s = df_view.groupby("na10_desc_fr")[METRIC].sum().sort_values(ascending=False)
        if len(s) > 0:
            top_na10 = s.index[0]

    k1.metric("Total (QTD)", f"{total:,}".replace(",", "."))
    k2.metric("Δ YoY", yoy_txt)
    k3.metric("Top GS", top_gs)
    k4.metric("Top NA10", top_na10)

    # Charts grid
    r1c1, r1c2 = st.columns(2)
    if PLOTLY_OK and METRIC in df_view.columns:
        # Série temporal por Ano
        with r1c1:
            st.subheader("Série temporal (Ano)")
            ser = df_view.groupby("Ano", dropna=False)[METRIC].sum().reset_index()
            fig = px.line(ser, x="Ano", y=METRIC, markers=True)
            st.plotly_chart(fig, use_container_width=True)
        # Top 10 NA10
        with r1c2:
            st.subheader("Top 10 — NA10")
            if "na10_desc_fr" in df_view.columns:
                top = (df_view.groupby("na10_desc_fr")[METRIC].sum()
                       .sort_values(ascending=False).head(10).reset_index())
                fig = px.bar(top, x=METRIC, y="na10_desc_fr", orientation="h")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Sem rótulos FR para NA10.")

        r2c1, r2c2 = st.columns(2)
        # Breakdown GS por Ano (stacked)
        with r2c1:
            st.subheader("GS por Ano (stacked)")
            if "gs_desc_fr" in df_view.columns and "Ano" in df_view.columns:
                g = (df_view.groupby(["Ano","gs_desc_fr"])[METRIC].sum().reset_index())
                fig = px.bar(g, x="Ano", y=METRIC, color="gs_desc_fr")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Sem rótulos FR para GS.")

        # Treemap NA5
        with r2c2:
            st.subheader("NA5 (treemap)")
            if "na5_desc_fr" in df_view.columns:
                g = (df_view.groupby("na5_desc_fr")[METRIC].sum().reset_index())
                fig = px.treemap(g, path=["na5_desc_fr"], values=METRIC)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Sem rótulos FR para NA5.")

# -------- Tab 3: Tableau --------
with tab3:
    st.header("Tableau")
    q = st.text_input("🔎 Filtrar por texto (qualquer coluna visível):", "")
    base_cols = ["Ano","gs_desc_fr","na10_desc_fr","na5_desc_fr", METRIC]
    cols = [c for c in base_cols if c in df_view.columns]
    data = df_view[cols].copy()
    if q:
        ql = q.lower()
        data = data[data.apply(lambda r: ql in (" ".join(map(str, r.values))).lower(), axis=1)]
    st.dataframe(data.head(2000))
    st.download_button("Baixar CSV (recorte)", data=data.to_csv(index=False, sep=";").encode("utf-8-sig"),
                       file_name="tableau.csv", mime="text/csv")

# -------- Tab 4: Cartes (placeholder) --------
with tab4:
    st.header("Cartes")
    st.info("🗺️ Os mapas foram temporariamente desativados. A aba permanece para configurarmos amanhã.")
    st.caption("Dica: deixe prontos os arquivos em data/geo (GeoJSON) ou um CSV de centróides; amanhã plugamos aqui.")

# -------- Tab 5: Aperçu --------
with tab5:
    st.header("Aperçu")
    st.markdown('''
**Vive le Marché** — painel exploratório com dados normalizados e dimensões com rótulos em francês.

**Filtros ativos:** Ano, GS, NA5, NA10.  
**Métrica padrão:** `QTD` (soma).

**Pastas esperadas**
- `data/fact_vive_le_marche_fr.parquet` *(ou `fact_vive_le_marche.parquet`)*
- `data/dimensions` **ou** `data/dimension` **ou** `data/dimensions_parquet`
- `data/geo` *(opcional; GeoJSONs futuros para REGLT/ZELT)*

Se precisar de novos gráficos ou KPIs, peça na conversa 😉
''')
    # Pequeno sumário de linhas
    st.write("Registros no recorte atual:", len(df_view))
    if "Ano" in df_view.columns:
        anos_list = ", ".join(map(str, sorted(pd.to_numeric(df_view['Ano'], errors='coerce').dropna().unique().astype(int))))
        st.write("Anos presentes:", anos_list)
