# Local data only

Official train/valid rows are not distributed with the code. The preparation
scripts generate role manifests from the user's local official CSV files,
validate counts, positive counts and the historical SHA256, and write them only
inside a new runtime project. No label, age or image list is needed in this folder.

`../` source provenance remains in the vendor directory's SOURCE_SHA256.json,
UPSTREAM_SHA256.json and DATA_PROVENANCE.json. The upstream index records original
hashes, including excluded data files; it is not a list of files to distribute.
