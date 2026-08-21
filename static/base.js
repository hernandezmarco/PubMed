// Copyright (C) 2025 Marco Hernandez <ragettyandy@gmail.com>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//
// For information contact Marco Hernandez <ragettyandy@gmail.com>

// ── CSRF-aware fetch wrapper ─────────────────────────────────────────────────
// Attaches the CSRF token (from the meta tag base.html renders) to any mutating
// request. On a 401 (access token expired), tries a silent /auth/refresh once and
// retries the original request; only redirects to /login if that also fails.
const CSRF_TOKEN = document.querySelector('meta[name="csrf-token"]')?.content || '';
const SAFE_METHODS = new Set(['GET', 'HEAD', 'OPTIONS', 'TRACE']);

async function authFetch(url, options = {}, _isRetry = false) {
  const method = (options.method || 'GET').toUpperCase();
  const headers = new Headers(options.headers || {});
  if (!SAFE_METHODS.has(method)) {
    headers.set('X-CSRFToken', CSRF_TOKEN);
  }
  const resp = await fetch(url, { ...options, headers });
  // /auth/* routes return 401 for "wrong credentials," not "session expired" —
  // that's a normal response the caller (the login form) needs to handle itself.
  const isAuthEndpoint = typeof url === 'string' && url.startsWith('/auth/');
  if (resp.status === 401 && !isAuthEndpoint) {
    if (!_isRetry) {
      const refreshResp = await fetch('/auth/refresh', {
        method: 'POST',
        headers: { 'X-CSRFToken': CSRF_TOKEN },
      });
      if (refreshResp.ok) {
        return authFetch(url, options, true);
      }
    }
    const next = encodeURIComponent(window.location.pathname + window.location.search);
    window.location.href = `/login?next=${next}`;
  }
  return resp;
}

// ── Logout ────────────────────────────────────────────────────────────────────
const logoutBtn = document.getElementById('logout-btn');
if (logoutBtn) {
  logoutBtn.addEventListener('click', async () => {
    await authFetch('/auth/logout', { method: 'POST' });
    window.location.href = '/login';
  });
}

const bar = document.getElementById('progress-bar');

const Progress = {
  start(durationS = 10, targetPct = 65) {
    bar.classList.remove('error');
    bar.style.display = 'block';
    bar.style.transition = 'none';
    bar.style.width = '0%';
    requestAnimationFrame(() => {
      bar.style.transition = `width ${durationS}s ease`;
      bar.style.width = targetPct + '%';
    });
  },
  set(pct, ms = 400) {
    bar.style.display = 'block';
    bar.style.transition = `width ${ms}ms ease`;
    bar.style.width = pct + '%';
  },
  done() {
    bar.style.transition = 'width .2s ease';
    bar.style.width = '100%';
    setTimeout(() => { bar.style.display = 'none'; bar.style.width = '0%'; }, 350);
  },
  error() {
    bar.classList.add('error');
    this.done();
    setTimeout(() => bar.classList.remove('error'), 600);
  },
};