import argparse
import csv
import io
import struct
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

import requests


GEONAMES_BASE = "https://download.geonames.org/export/dump"
NATURAL_EARTH_ADMIN0 = (
    "https://naturalearth.s3.amazonaws.com/110m_cultural/"
    "ne_110m_admin_0_countries.zip"
)
SCALE_LEVELS = {
    "1": "1. Global Scale",
    "2": "2. Multi-national / Continental Scale",
    "3": "3. National / Single-country Scale",
    "4": "4. Multi-provincial / Sub-national Regional Scale",
    "5": "5. Single-provincial / State Scale",
    "6": "6. Multi-city / Megaregion Scale",
    "7": "7. Single-city / Municipal Scale",
    "8": "8. District / County Scale",
    "9": "9. Micro / Neighborhood / Block Scale",
}


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    path.write_bytes(response.content)


def clean(value: object) -> str:
    if value is None:
        return ""
    text = str(value).replace("\x00", "")
    return "".join(ch for ch in text if ch >= " " or ch in "\t\n\r").strip()


def add_entity(
    entities: Dict[str, Dict[str, object]],
    aliases: List[Dict[str, str]],
    geo_id: str,
    canonical_name: str,
    mapped_scale_level: str,
    *,
    country_code: str = "",
    admin_level: str = "",
    feature_code: str = "",
    parent_geo_id: str = "",
    population: int = 0,
    source: str = "",
    hierarchy_path: str = "",
    extra_aliases: Iterable[str] = (),
) -> None:
    geo_id = clean(geo_id)
    canonical_name = clean(canonical_name)
    if not geo_id or not canonical_name or not mapped_scale_level:
        return
    entities[geo_id] = {
        "geo_id": geo_id,
        "canonical_name": canonical_name,
        "country_code": clean(country_code),
        "admin_level": clean(admin_level),
        "feature_code": clean(feature_code),
        "parent_geo_id": clean(parent_geo_id),
        "population": int(population or 0),
        "source": clean(source),
        "mapped_scale_level": mapped_scale_level,
        "hierarchy_path": clean(hierarchy_path),
    }
    for alias in {canonical_name, *[clean(item) for item in extra_aliases if clean(item)]}:
        aliases.append({"geo_id": geo_id, "alias": alias, "source": clean(source)})


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def iter_geonames_country_info(path: Path) -> Iterator[Dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 17:
                continue
            yield {
                "iso": parts[0],
                "iso3": parts[1],
                "name": parts[4],
                "capital": parts[5],
                "continent": parts[8],
                "geonameid": parts[16],
            }


def iter_geonames_admin_codes(path: Path, admin_level: str) -> Iterator[Dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            code, name, ascii_name, geonameid = parts[:4]
            country_code = code.split(".", 1)[0]
            yield {
                "code": code,
                "name": name,
                "ascii_name": ascii_name,
                "geonameid": geonameid,
                "country_code": country_code,
                "admin_level": admin_level,
            }


def iter_geonames_cities(zip_path: Path) -> Iterator[Dict[str, str]]:
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open("cities15000.txt") as raw:
            for binary in raw:
                line = binary.decode("utf-8", errors="ignore").rstrip("\n")
                parts = line.split("\t")
                if len(parts) < 19:
                    continue
                yield {
                    "geonameid": parts[0],
                    "name": parts[1],
                    "ascii_name": parts[2],
                    "alternate_names": parts[3],
                    "country_code": parts[8],
                    "feature_code": parts[7],
                    "population": parts[14],
                    "admin1": parts[10],
                    "admin2": parts[11],
                }


def iter_dbf_records(dbf_bytes: bytes) -> Iterator[Dict[str, str]]:
    record_count = struct.unpack("<I", dbf_bytes[4:8])[0]
    header_length = struct.unpack("<H", dbf_bytes[8:10])[0]
    record_length = struct.unpack("<H", dbf_bytes[10:12])[0]
    fields = []
    offset = 32
    while offset < header_length and dbf_bytes[offset] != 0x0D:
        descriptor = dbf_bytes[offset : offset + 32]
        name = descriptor[:11].split(b"\x00", 1)[0].decode("ascii", errors="ignore")
        length = descriptor[16]
        fields.append((name, length))
        offset += 32
    base = header_length
    for index in range(record_count):
        record = dbf_bytes[base + index * record_length : base + (index + 1) * record_length]
        if not record or record[0:1] == b"*":
            continue
        pos = 1
        row = {}
        for name, length in fields:
            value = record[pos : pos + length].decode("utf-8", errors="ignore").strip()
            row[name] = value
            pos += length
        yield row


def iter_natural_earth_macro_regions(zip_path: Path) -> Iterator[Dict[str, str]]:
    with zipfile.ZipFile(zip_path) as archive:
        dbf_name = next(name for name in archive.namelist() if name.lower().endswith(".dbf"))
        dbf_bytes = archive.read(dbf_name)
    seen = set()
    for row in iter_dbf_records(dbf_bytes):
        for field in ("CONTINENT", "REGION_UN", "SUBREGION"):
            value = clean(row.get(field))
            if not value or value in {"Seven seas (open ocean)", "Antarctica"}:
                continue
            key = (field, value)
            if key in seen:
                continue
            seen.add(key)
            yield {"geo_id": f"ne-{field.lower()}-{value.lower().replace(' ', '-')}", "name": value}


def build_reference(output_dir: Path, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    download(f"{GEONAMES_BASE}/countryInfo.txt", raw_dir / "countryInfo.txt")
    download(f"{GEONAMES_BASE}/admin1CodesASCII.txt", raw_dir / "admin1CodesASCII.txt")
    download(f"{GEONAMES_BASE}/admin2Codes.txt", raw_dir / "admin2Codes.txt")
    download(f"{GEONAMES_BASE}/cities15000.zip", raw_dir / "cities15000.zip")
    download(NATURAL_EARTH_ADMIN0, raw_dir / "ne_110m_admin_0_countries.zip")

    entities: Dict[str, Dict[str, object]] = {}
    aliases: List[Dict[str, str]] = []

    add_entity(
        entities,
        aliases,
        "global",
        "Global",
        SCALE_LEVELS["1"],
        source="project_overlay",
        extra_aliases=["World", "Worldwide"],
    )

    for row in iter_natural_earth_macro_regions(raw_dir / "ne_110m_admin_0_countries.zip"):
        add_entity(
            entities,
            aliases,
            row["geo_id"],
            row["name"],
            SCALE_LEVELS["2"],
            source="natural_earth",
        )

    for row in iter_geonames_country_info(raw_dir / "countryInfo.txt"):
        add_entity(
            entities,
            aliases,
            f"geonames:{row['geonameid']}",
            row["name"],
            SCALE_LEVELS["3"],
            country_code=row["iso"],
            feature_code="PCLI",
            source="geonames_countryInfo",
            extra_aliases=[row["iso"], row["iso3"]],
        )

    for row in iter_geonames_admin_codes(raw_dir / "admin1CodesASCII.txt", "ADM1"):
        add_entity(
            entities,
            aliases,
            f"geonames:{row['geonameid']}",
            row["name"],
            SCALE_LEVELS["5"],
            country_code=row["country_code"],
            admin_level="ADM1",
            feature_code="ADM1",
            source="geonames_admin1",
            extra_aliases=[row["ascii_name"]],
        )

    for row in iter_geonames_admin_codes(raw_dir / "admin2Codes.txt", "ADM2"):
        add_entity(
            entities,
            aliases,
            f"geonames:{row['geonameid']}",
            row["name"],
            SCALE_LEVELS["8"],
            country_code=row["country_code"],
            admin_level="ADM2",
            feature_code="ADM2",
            source="geonames_admin2",
            extra_aliases=[row["ascii_name"]],
        )

    for row in iter_geonames_cities(raw_dir / "cities15000.zip"):
        alternate_names = [item for item in row["alternate_names"].split(",")[:20] if item]
        add_entity(
            entities,
            aliases,
            f"geonames:{row['geonameid']}",
            row["name"],
            SCALE_LEVELS["7"],
            country_code=row["country_code"],
            admin_level="PPL",
            feature_code=row["feature_code"],
            population=int(row["population"] or 0),
            source="geonames_cities15000",
            extra_aliases=[row["ascii_name"], *alternate_names],
        )

    for geo_id, name in {
        "project:yangtze_river_delta": "Yangtze River Delta",
        "project:pearl_river_delta": "Pearl River Delta",
        "project:greater_bay_area": "Greater Bay Area",
        "project:global_south": "Global South",
    }.items():
        scale = SCALE_LEVELS["6"] if "global_south" not in geo_id else SCALE_LEVELS["2"]
        add_entity(entities, aliases, geo_id, name, scale, source="project_overlay")

    entity_fields = [
        "geo_id",
        "canonical_name",
        "country_code",
        "admin_level",
        "feature_code",
        "parent_geo_id",
        "population",
        "source",
        "mapped_scale_level",
        "hierarchy_path",
    ]
    alias_fields = ["geo_id", "alias", "source"]
    write_csv(output_dir / "geo_entities.csv", entities.values(), entity_fields)
    write_csv(output_dir / "geo_aliases.csv", aliases, alias_fields)
    (output_dir / "geo_source_attribution.md").write_text(
        "\n".join(
            [
                "# Geography Reference Sources",
                "",
                "- GeoNames data: https://www.geonames.org/about.html; license: Creative Commons Attribution.",
                "- Natural Earth data: https://www.naturalearthdata.com/about/terms-of-use; public domain.",
                "- Project overlay: locally curated regions for urban-renewal literature analysis.",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local geography reference tables.")
    parser.add_argument(
        "--output-dir",
        default="Data/reference/geography",
        help="Directory for geo_entities.csv, geo_aliases.csv, and attribution.",
    )
    parser.add_argument(
        "--raw-dir",
        default="Data/reference/geography/raw",
        help="Directory for downloaded public source files.",
    )
    args = parser.parse_args()
    build_reference(Path(args.output_dir), Path(args.raw_dir))


if __name__ == "__main__":
    main()
