"""One loader module per dataset (`berlin`, `finnish`, `deutsche_bahn`), each exposing roughly
the same shape — raw loaders, a `build_feature_table`/`load_*_table` tabular feature builder,
`encode_categoricals`, and a train/test split function — so `tabpfn_lab/evaluate*.py` and the
dashboard's exploration pages can treat any of them similarly. Not a strict shared interface
(the datasets differ too much for that to be worth forcing yet — see `docs/MODULES.md`), just a
shared naming convention.
"""
