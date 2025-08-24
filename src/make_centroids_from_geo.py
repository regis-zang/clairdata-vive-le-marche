

# make_centroids_from_geo.py
# Uso:
#   conda activate vlm311
#   python make_centroids_from_geo.py --geo "I:/.../fr_zones_emploi.geojson" --code_col ZE2020 --name_col LIBGEO --out "I:/.../data/geo/zelt_centroids.csv"

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geo", required=True, help="GeoJSON de entrada")
    ap.add_argument("--code_col", required=True, help="Coluna de código (ex.: ZE2020)")
    ap.add_argument("--name_col", default=None, help="Coluna de nome (opcional)")
    ap.add_argument("--out", required=True, help="CSV de saída (ZELT,lon,lat,zelt_desc_fr)")
    args = ap.parse_args()

    geo = Path(args.geo)
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True, parents=True)

    gdf = gpd.read_file(geo)
    # garante WGS84
    if gdf.crs is None or str(gdf.crs).upper() != "EPSG:4326":
        gdf = gdf.to_crs(4326)

    # centróides (lon/lat)
    gdf["centroid"] = gdf.geometry.centroid
    df = pd.DataFrame({
        "ZELT": gdf[args.code_col].astype(str),
        "lon": gdf["centroid"].x,
        "lat": gdf["centroid"].y,
    })
    if args.name_col and args.name_col in gdf.columns:
        df["zelt_desc_fr"] = gdf[args.name_col].astype(str)

    df.to_csv(out, index=False)
    print(f"CSV salvo em: {out}")

if __name__ == "__main__":
    main()
