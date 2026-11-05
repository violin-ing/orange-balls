# Data versions

Records exactly which version of each outside dataset this project uses, so
that every result can be reproduced. The data files themselves are not in
git (`data/` is ignored); this file is the record of them.

**Update this file whenever a dataset is downloaded again or replaced.**
Add the new entry and keep the old one under "Previous versions", so a
result can always be traced to the data it came from.

To fill in or check a checksum, run from the repository root (Git Bash):

```
sha256sum data/reference/vdem/*.csv data/reference/wgi/*.xlsx
```

A checksum that differs from the one recorded here means the file is not
the one the results were produced from.

---

## Reference datasets (downloaded by hand)

### V-Dem (Varieties of Democracy)

| Field | Value |
|---|---|
| Dataset | Country-Year: V-Dem Full+Others |
| Version | v16 (published March 2026) |
| Format | CSV |
| Downloaded | 2026-10-21 |
| Source | https://v-dem.net/data/the-v-dem-dataset/ |
| Saved to | `data/reference/vdem/` |
| File name | `V-Dem-CY-Full+Others-v16.csv` |
| SHA-256 | d65f32bacb7b358c58ca9c908424444170058fc283acb625f3a6bce74faa1994 |
| Codebook | Included in the download ZIP; saved alongside the CSV |
| Licence | CC BY-SA 4.0 |
| Used for | `v2mecenefm` (government censorship effort, media), `v2meharjrn` (harassment of journalists), internet shutdown capacity (variable name to confirm against the codebook) |

### WGI (Worldwide Governance Indicators)

| Field | Value |
|---|---|
| Dataset | WGI 2026: Governance Estimates and Scores (1996-2025) |
| Edition | 2026 update (data years 1996 to 2025) |
| Format | Excel (no CSV is offered) |
| Downloaded | 2026-10-21 |
| Source | https://www.worldbank.org/en/publication/worldwide-governance-indicators |
| Saved to | `data/reference/wgi/` |
| File name | `wgidataset_with_sourcedata-2026.xlsx` |
| SHA-256 | 67c38d2e9253fb1a72ee302c3b7c47b2ba7b177a04673163796ca1666f506ecb |
| Licence | CC BY 4.0 |
| Used for | Government Effectiveness, Political Stability and Absence of Violence/Terrorism (estimates and standard errors) |

### World Bank

| Field | Value |
|---|---|
| Dataset | World Bank World Development Indicators (WDI), NY.GDP.PCAP.PP.KD: GDP per capita, PPP (constant 2021 international $) |
| Edition | WDI, last updated 2026-10-08 (from line 3 of the API file); data years 1990 to 20XX |
| Format | CSV (zip of 3 files: values, country metadata, indicator metadata) |
| Downloaded | 2026-11-05 |
| Source | https://api.worldbank.org/v2/en/indicator/NY.GDP.PCAP.PP.KD?downloadformat=csv |
| Saved to | `data/reference/worldbank/` |
| File name | `API_NY.GDP.PCAP.PP.KD_DS2_en_csv_v2_1629.csv` (plus its `Metadata_Country_…csv`) |
| SHA-256 | 35ddabcb95a270a58b5db32eb547f0326c91f9652d1329e4b877abe1ec928411 |
| Licence | CC BY 4.0 |
| Used for | Validation only |
---

## Live dataset (pulled by code)

### GDELT 2.0 Events

| Field | Value |
|---|---|
| Dataset | GDELT 2.0 Event Database, export stream |
| Version | Live; updated every 15 minutes, so there is no fixed version |
| Index | http://data.gdeltproject.org/gdeltv2/masterfilelist.txt |
| Saved to | `data/cache/gdelt/` (one ZIP per 15-minute file) |
| Pulled by | `src/pull_day.py` |
| Integrity | Each file is checked against the MD5 checksum published in the index |
| Record of pulls | `pull_log` table in `data/processed/country_day.sqlite` (date, time pulled, files read, rows dropped) |
| Column schema | GDELT 2.0 Event Codebook, cross-checked against downloaded data (see `src/gdelt/schema.py`) |

Because GDELT has no version number, the record of exactly what was used is
the `pull_log` table together with the cached ZIP files.

---

## Hand-kept tables (in git)

These are small files maintained by hand in the repository. Their history
is in git, so no checksum is needed here.

| File | What it is | Built from |
|---|---|---|
| `src/countries/fips_iso3.csv` | FIPS 10-4 to ISO3 conversion for GDELT location codes | GDELT's country list (http://data.gdeltproject.org/api/v2/guides/LOOKUP-COUNTRIES.TXT), ISO codes checked with the pycountry package; plus `RB` (Serbia), found in the data but missing from the list |
| `src/countries/cameo_iso3.csv` | CAMEO actor codes that differ from ISO3 | Known differences; grows as runs report unmatched codes |

---

## Previous versions

None yet.