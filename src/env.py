"""Token loader for TabPFN (Canon Detector / FEAT-001).

:func:`load_token` reads ``TABPFN_TOKEN`` from the hackathon-root ``.env`` and
registers it with ``tabpfn_client.set_access_token``. The token value is NEVER
printed, logged, or returned. If python-dotenv is installed it is used; otherwise
the ``TABPFN_TOKEN=`` line is parsed manually. A clear error is raised if the file
or the token is missing.

Verification (per project policy, no unit tests)::

    python -c "from src.env import load_token; load_token(); print('ok')"
"""

from __future__ import annotations

from pathlib import Path

import tabpfn_client

# Absolute path to the hackathon-root .env (read-only, git-ignored, never copied).
ENV_PATH = Path("/home/sierra/Desktop/projects/hackathons/tabpfn-3-5/.env")
_TOKEN_VAR = "TABPFN_TOKEN"


def _read_token_from_env_file(path: Path) -> str | None:
    """Return the TABPFN_TOKEN value from the .env file, or None.

    Prefers python-dotenv; falls back to manual parsing of the TABPFN_TOKEN=
    line. Does not mutate os.environ with the secret beyond what dotenv does,
    and never prints the value.
    """
    try:
        from dotenv import dotenv_values

        values = dotenv_values(str(path))
        token = values.get(_TOKEN_VAR)
        if token:
            return token.strip()
    except ImportError:
        pass

    # Manual fallback: parse the TABPFN_TOKEN= line.
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(f"{_TOKEN_VAR}="):
            value = line.split("=", 1)[1].strip()
            # Strip optional surrounding quotes.
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            return value or None
    return None


def load_token() -> None:
    """Load TABPFN_TOKEN from .env and set it as the TabPFN access token.

    Raises FileNotFoundError if the .env file is missing and RuntimeError if the
    token is absent or empty. Returns nothing and never exposes the token.
    """
    if not ENV_PATH.exists():
        raise FileNotFoundError(
            f"TabPFN .env not found at {ENV_PATH}. Expected a line "
            f"'{_TOKEN_VAR}=<token>'."
        )

    token = _read_token_from_env_file(ENV_PATH)
    if not token:
        raise RuntimeError(
            f"{_TOKEN_VAR} is missing or empty in {ENV_PATH}. "
            "Add a line '{_TOKEN_VAR}=<token>'."
        )

    tabpfn_client.set_access_token(token)
    # Intentionally no return / no print of the token.


if __name__ == "__main__":
    load_token()
    print("ok")
