import requests
from jose import jwt

APPLE_KEYS_URL = "https://appleid.apple.com/auth/keys"

_cached_keys = None

def _fetch_apple_public_keys():
    global _cached_keys
    if _cached_keys is not None:
        return _cached_keys
    resp = requests.get(APPLE_KEYS_URL)
    resp.raise_for_status()
    _cached_keys = resp.json().get("keys", [])
    return _cached_keys

def verify_apple_identity_token(token, audience):
    """Verify Apple identity token and return claims."""
    keys = _fetch_apple_public_keys()
    header = jwt.get_unverified_header(token)
    kid = header.get("kid")
    key = next((k for k in keys if k.get("kid") == kid), None)
    if not key:
        raise ValueError("Public key not found for Apple token")
    return jwt.decode(
        token,
        key,
        algorithms=["RS256"],
        audience=audience,
        issuer="https://appleid.apple.com",
    )
