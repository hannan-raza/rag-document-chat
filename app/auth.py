"""
Auth0 JWT validation.
Verifies the RS256 signature against Auth0's JWKS, checks issuer/audience/
expiry, and returns the user's `sub` (their unique ID) for use in endpoints.
"""
import os
import jwt  # PyJWT
from jwt import PyJWKClient
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN", "dev-zg6htujr3rwoqesc.us.auth0.com")
AUTH0_AUDIENCE = os.getenv("AUTH0_AUDIENCE", "https://docmind-api")
ISSUER = f"https://{AUTH0_DOMAIN}/"
JWKS_URL = f"https://{AUTH0_DOMAIN}/.well-known/jwks.json"

# fetches + caches Auth0's public signing keys
_jwks_client = PyJWKClient(JWKS_URL)
_bearer = HTTPBearer()

def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(_bearer),
) -> str:
    """Validate the bearer token and return the user's Auth0 `sub` (user id)."""
    token = creds.credentials
    try:
        signing_key = _jwks_client.get_signing_key_from_jwt(token).key
        payload = jwt.decode(
            token,
            signing_key,
            algorithms=["RS256"],
            audience=AUTH0_AUDIENCE,
            issuer=ISSUER,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}",
        )
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Token missing sub")
    return user_id
