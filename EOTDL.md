# Publishing on EOTDL

This is an EOTDL **dataset** entry, `spain-biomass-deployment`: the tagged repository plus `data/` (about 1.3 GB). The root README carries the YAML metadata that EOTDL reads. The evaluated models themselves are a separate EOTDL model entry, [sentinel-biomass-inference](https://www.eotdl.com/models/sentinel-biomass-inference).

## Build and check the snapshot

On a clean checkout at the release tag, with `data/` in place (maintainers rebuild it with `scripts/build_ifn_data.py` and `scripts/build_cci_data.py`):

```bash
python scripts/prepare_eotdl.py --tag v1.0.0
cd dist/v1.0.0/spain-biomass-deployment
python scripts/reproduce.py
```

The builder refuses uncommitted changes, a HEAD that is not the tag, and an existing output folder. It copies files (no links) and writes `release-manifest.json` with the commit and a SHA-256 hash per file. Running the reproduction inside the snapshot creates `outputs/`. **Delete `outputs/` (or rebuild into a fresh folder) before uploading**, so only the manifest-listed files are ingested.

## Upload

Use the EOTDL client in its own Python 3.12+ environment:

```bash
eotdl auth login
eotdl datasets ingest -p ./dist/v1.0.0/spain-biomass-deployment
```

EOTDL assigns the version number on ingestion; it is not derived from the Git tag. Afterwards:

1. download the published `release-manifest.json` and check it against the local one;
2. record the mapping (tag, commit, EOTDL version, dataset ID, manifest hash) in `provenance/releases/v1.0.0.json` in a commit after the tag;
3. add the EOTDL version to the README and to the AI4DIS "EOTDL resources" table.

Publish the GitHub repository before ingesting, so the `source` link in the metadata resolves.
