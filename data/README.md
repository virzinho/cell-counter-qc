# Data provenance

Source: https://bbbc.broadinstitute.org/BBBC039 (BBBC039v1, CC0).

The downloader fetches `images.zip`, `masks.zip` and `metadata.zip` from Broad. SHA-256 checksums from the executed run are tracked in `checksums.json`. Full archives and extracted datasets are excluded from Git to keep the repository small.

Three original 16-bit validation images and their source filenames are included in `examples/` for an offline demo. They are the first three entries of the published validation list, not selected by test performance. The bundled manifest lists reference nucleus counts decoded from the source annotations.

Recommended citation: BBBC039v1, Caicedo et al. 2018; Broad Bioimage Benchmark Collection, Ljosa et al., Nature Methods 2012. Source masks, dataset splits and the archive's metadata remain unchanged.
