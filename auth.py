# Copyright (C) 2025 Marco Hernandez <ragettyandy@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# For information contact Marco Hernandez <ragettyandy@gmail.com>

"""
Password hashing, JWT access tokens, refresh/reset/verification token hashing, the
@login_required(_page) decorators, and the verification/password-reset emails (sent
via the SMTP2GO HTTP API).
"""
import datetime
import hashlib
import logging
from functools import wraps

import jwt
import requests
from flask import g, jsonify, redirect, request, url_for
from werkzeug.security import check_password_hash, generate_password_hash

import config as cfg

_log = logging.getLogger("pubmed.auth")

ACCESS_COOKIE_NAME = "access_token"
REFRESH_COOKIE_NAME = "refresh_token"

if not cfg.JWT_SECRET_KEY:
    _log.warning(
        "JWT_SECRET_KEY is not set — access tokens will be signed with an empty key. "
        "Set JWT_SECRET_KEY in your .env before relying on auth for anything real."
    )


def hash_password(password: str) -> str:
    return generate_password_hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return check_password_hash(password_hash, password)


def create_access_token(user_id: int) -> str:
    now = datetime.datetime.now(datetime.timezone.utc)
    payload = {
        "sub": str(user_id),  # JWT spec requires "sub" to be a string
        "iat": now,
        "exp": now + datetime.timedelta(minutes=cfg.JWT_ACCESS_TTL_MINUTES),
    }
    return jwt.encode(payload, cfg.JWT_SECRET_KEY, algorithm=cfg.JWT_ALGORITHM)


def decode_access_token(token: str) -> int:
    """Return the user_id encoded in a valid, unexpired access token, or raise."""
    payload = jwt.decode(token, cfg.JWT_SECRET_KEY, algorithms=[cfg.JWT_ALGORITHM])
    return int(payload["sub"])


def _authenticate() -> bool:
    """Try to set g.user_id from the access token cookie. Returns whether it succeeded."""
    token = request.cookies.get(ACCESS_COOKIE_NAME)
    if not token:
        return False
    try:
        g.user_id = decode_access_token(token)
        return True
    except jwt.PyJWTError as exc:
        _log.debug("op=authenticate rejected: %s", exc)
        return False


def login_required(view):
    """For JSON/API routes: responds 401 if not authenticated."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not _authenticate():
            return jsonify({"error": "Not authenticated."}), 401
        return view(*args, **kwargs)
    return wrapped


def login_required_page(view):
    """For HTML page routes: redirects to /login (preserving the original URL) if not authenticated."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not _authenticate():
            return redirect(url_for("login_page", next=request.full_path))
        return view(*args, **kwargs)
    return wrapped


def hash_token(raw_token: str) -> str:
    """Hash an opaque random token (refresh / password-reset) for storage.

    Unlike hash_password, this uses a fast hash (SHA-256) deliberately — these
    tokens are high-entropy random strings (secrets.token_urlsafe), not
    low-entropy user passwords, so there's no brute-force risk a slow adaptive
    hash would need to defend against.
    """
    return hashlib.sha256(raw_token.encode()).hexdigest()


_SMTP2GO_SEND_URL = "https://api.smtp2go.com/v3/email/send"


def _send_email(to_email: str, subject: str, text_body: str) -> bool:
    """POST one email through the SMTP2GO API. Returns whether the send succeeded
    (logs and swallows errors — a failed notification email should never break the
    request that triggered it)."""
    if not cfg.SMTP2GO_API:
        _log.warning("op=send_email skipped: SMTP2GO_API is not configured to=%s", to_email)
        return False

    try:
        resp = requests.post(
            _SMTP2GO_SEND_URL,
            headers={
                "Content-Type": "application/json",
                "X-Smtp2go-Api-Key": cfg.SMTP2GO_API,
                "Accept": "application/json",
            },
            json={
                "sender": cfg.EMAIL_FROM,
                "to": [to_email],
                "subject": subject,
                "text_body": text_body,
            },
            timeout=10,
        )
        resp.raise_for_status()
        # SMTP2GO reports per-recipient failures (bad/unverified sender domain, bounces,
        # etc.) inside a 200 OK body rather than as an HTTP error status.
        data = resp.json().get("data", {})
        if data.get("failed"):
            _log.error(
                "op=send_email rejected to=%s subject=%r failures=%s",
                to_email, subject, data.get("failures"),
            )
            return False
        _log.info("op=send_email sent to=%s subject=%r", to_email, subject)
        return True
    except Exception as exc:
        _log.exception("op=send_email failed to=%s subject=%r: %s", to_email, subject, exc)
        return False


def send_verification_email(to_email: str, raw_token: str) -> bool:
    """Send the new-account email-confirmation link."""
    verify_url = f"{cfg.APP_BASE_URL}/verify-email?token={raw_token}"
    return _send_email(
        to_email,
        "Verify your PubMed AI account",
        "Welcome! Confirm your email to finish creating your account.\n\n"
        f"Verify your email: {verify_url}\n\n"
        f"This link expires in {cfg.EMAIL_VERIFICATION_TTL_MINUTES // 60} hours. "
        "If you didn't create this account, you can ignore this email.",
    )


def send_password_reset_email(to_email: str, raw_token: str) -> bool:
    """Send the reset link."""
    reset_url = f"{cfg.APP_BASE_URL}/reset-password?token={raw_token}"
    return _send_email(
        to_email,
        "Reset your PubMed AI password",
        "Someone requested a password reset for this account.\n\n"
        f"Reset your password: {reset_url}\n\n"
        f"This link expires in {cfg.PASSWORD_RESET_TTL_MINUTES} minutes. "
        "If you didn't request this, you can ignore this email.",
    )
