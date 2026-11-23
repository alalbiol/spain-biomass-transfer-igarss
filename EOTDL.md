# Publishing on EOTDL

This is an EOTDL **dataset** entry, `spain-biomass-transfer-igarss`: the tagged repository plus `data/` (about 5 GB, about 2,200 files). The root README carries the YAML metadata that EOTDL reads. The evaluated models themselves are a separate EOTDL model entry, [sentinel-biomass-inference](https://www.eotdl.com/models/sentinel-biomass-inference).

## Published version

GitHub release `v1.0.0` (commit `509b587be7ae71739b023cdbbc8943934468cbe7`) corresponds to EOTDL dataset version **1** (ID `6b0422a6daabc68934fe8591`), uploaded on 23 November 2026. The EOTDL catalog SHA-1 of all 2,171 uploaded files and the uploaded `release-manifest.json` were checked against the snapshot built from that commit. See `provenance/releases/v1.0.0.json`. Changes on `main` after the tag are not part of the upload; a new upload creates a new EOTDL version.

## Build and check the snapshot

On a clean checkout at the release tag, with `data/` in place (maintainers rebuild it with `scripts/build_ifn_data.py`, `build_cci_data.py`, `build_prediction_maps.py` and `build_example_inputs.py`, in that order):

```bash
python scripts/prepare_eotdl.py --tag v1.0.0
cd dist/v1.0.0/spain-biomass-transfer-igarss
python scripts/reproduce.py
python scripts/check_prediction_maps.py
```

The builder refuses uncommitted changes, a HEAD that is not the tag, and an existing output folder. It copies files (no links) and writes `release-manifest.json` with the commit and a SHA-256 hash per file. Running the reproduction inside the snapshot creates `outputs/`. **Delete `outputs/` (or rebuild into a fresh folder) before uploading**, so only the manifest-listed files are ingested.

## Upload

Use the EOTDL client in its own Python 3.12+ environment:

```bash
eotdl auth login
eotdl datasets ingest -p ./dist/v1.0.0/spain-biomass-transfer-igarss
```

EOTDL assigns the version number on ingestion; it is not derived from the Git tag. Afterwards:

1. download the published `release-manifest.json` and check it against the local one;
2. record the mapping (tag, commit, EOTDL version, dataset ID, manifest hash) in `provenance/releases/v1.0.0.json` in a commit after the tag;
3. add the EOTDL version to the README and to the AI4DIS "EOTDL resources" table.

Publish the GitHub repository before ingesting, so the `source` link in the metadata resolves.
