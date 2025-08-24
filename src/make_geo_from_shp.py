

# make_geo_from_shp.py
# Uso:
#   conda activate vlm311
#   python make_geo_from_shp.py --shp "C:/data/ZELT.shp" --code_col ZE2020 --name_col LIBGEO --out "I:/Projetos_Python/clairdata-vive-le-marche/data/geo/fr_zones_emploi.geojson"
#
# Ou para regiões:
#   python make_geo_from_shp.py --shp "C:/data/REGION.shp" --code_col INSEE_REG --name_col NOM_REG --out "I:/Projetos_Python/clairdata-vive-le-marche/data/geo/fr_regions.geojson"

import argparse
from pathlib import Path

import geopandas as gpd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shp", required=True, help="Caminho do shapefile (.shp) de entrada")
    ap.add_argument("--code_col", required=True, help="Coluna com o código (ex.: ZE2020, INSEE_REG, REG, code)")
    ap.add_argument("--name_col", default=None, help="Coluna com o nome (opcional)")
    ap.add_argument("--out", required=True, help="Caminho do GeoJSON de saída")
    args = ap.parse_args()

    shp = Path(args.shp)
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True, parents=True)

    gdf = gpd.read_file(shp)
    # reprojeta para WGS84
    if gdf.crs is None or str(gdf.crs).upper() != "EPSG:4326":
        gdf = gdf.to_crs(4326)

    keep = [args.code_col]
    if args.name_col and args.name_col in gdf.columns:
        keep.append(args.name_col)
    gdf = gdf[keep + ["geometry"]].copy()

    # renomeia para campos padrão amigáveis
    ren = {args.code_col: args.code_col}
    if args.name_col and args.name_col in gdf.columns:
        ren[args.name_col] = args.name_col
    gdf = gdf.rename(columns=ren)

    gdf.to_file(out, driver="GeoJSON")
    print(f"GeoJSON salvo em: {out}")

if __name__ == "__main__":
    main()
