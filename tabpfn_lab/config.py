"""Paths, env loading, and constants shared across the package."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", REPO_ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", REPO_ROOT / "results"))

# `data/` holds one subfolder per dataset (see tabpfn_lab/datasets/ — one loader module each):
BERLIN_DATA_DIR = Path(os.environ.get("BERLIN_DATA_DIR", DATA_DIR / "Berlin_Ubahn_Alstom_data"))
# Deutsche Bahn data is fetched on demand from Hugging Face (tabpfn_lab/datasets/deutsche_bahn.py)
# rather than stored locally; DB_DATA_DIR is reserved for an optional local cache/mirror.
DB_DATA_DIR = Path(os.environ.get("DB_DATA_DIR", DATA_DIR / "DB_data"))

DEFAULT_DATA_DIR = BERLIN_DATA_DIR  # back-compat alias; prefer the explicit *_DATA_DIR constants above

TABPFN_MODEL_PATH = "v3.5_default"  # the model version the hackathon targets


def load_dotenv() -> Path | None:
    """Walk up from the cwd looking for a `.env` file and load it. Returns the path found, if any."""
    from dotenv import load_dotenv as _load

    here = Path.cwd().resolve()
    for d in [here, *here.parents]:
        candidate = d / ".env"
        if candidate.exists():
            _load(candidate)
            return candidate
    return None


def tabpfn_token() -> str:
    load_dotenv()
    token = os.environ.get("TABPFN_API_TOKEN")
    if not token:
        raise SystemExit(
            "No TABPFN_API_TOKEN found. Copy .env.example to .env and set it "
            "(get a free token at https://platform.priorlabs.ai/account/api-keys)."
        )
    return token


def authenticate() -> None:
    """Set the TabPFN API token for the `tabpfn_client` module-level session."""
    import tabpfn_client

    tabpfn_client.set_access_token(tabpfn_token())
