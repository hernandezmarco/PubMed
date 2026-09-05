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

const VISIBLE_CLASS = 'visible';
const NETWORK_ERROR_MESSAGE = 'Network error — please try again.';
const SUBMIT_BUTTON_SELECTOR = 'button[type="submit"]';
const EMAIL_INPUT_SELECTOR = '#email';
const AUTH_ERROR_ID = 'auth-error';
const AUTH_NOTICE_ID = 'auth-notice';

function nextUrl() {
  const params = new URLSearchParams(window.location.search);
  return params.get('next') || '/';
}

function wireAuthForm(formId, endpoint, { onSuccess, onError } = {}) {
  const form = document.getElementById(formId);
  if (!form) return;
  const errorBox = document.getElementById(AUTH_ERROR_ID);
  const noticeBox = document.getElementById(AUTH_NOTICE_ID);
  const submitBtn = form.querySelector(SUBMIT_BUTTON_SELECTOR);

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    errorBox.classList.remove(VISIBLE_CLASS);
    if (noticeBox) noticeBox.classList.remove(VISIBLE_CLASS);
    const resendBox = document.getElementById('resend-verification');
    if (resendBox) resendBox.style.display = 'none';
    submitBtn.disabled = true;

    const email = form.querySelector(EMAIL_INPUT_SELECTOR).value.trim();
    const password = form.querySelector('#password').value;

    try {
      const res = await authFetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });
      const data = await res.json();
      if (!res.ok) {
        errorBox.textContent = data.error || 'Something went wrong.';
        errorBox.classList.add(VISIBLE_CLASS);
        submitBtn.disabled = false;
        if (onError) onError(data, email);
        return;
      }
      if (onSuccess) {
        onSuccess(data);
      } else {
        window.location.href = nextUrl();
      }
    } catch (err) {
      console.error('Auth request failed:', err);
      errorBox.textContent = NETWORK_ERROR_MESSAGE;
      errorBox.classList.add(VISIBLE_CLASS);
      submitBtn.disabled = false;
    }
  });
}

function wireForgotPasswordForm() {
  const form = document.getElementById('forgot-password-form');
  if (!form) return;
  const errorBox = document.getElementById(AUTH_ERROR_ID);
  const noticeBox = document.getElementById(AUTH_NOTICE_ID);
  const submitBtn = form.querySelector(SUBMIT_BUTTON_SELECTOR);

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    errorBox.classList.remove(VISIBLE_CLASS);
    noticeBox.classList.remove(VISIBLE_CLASS);
    submitBtn.disabled = true;

    const email = form.querySelector(EMAIL_INPUT_SELECTOR).value.trim();
    try {
      const res = await authFetch('/auth/forgot-password', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email }),
      });
      const data = await res.json();
      // The server always responds the same way regardless of whether the
      // account exists, so this branch is effectively always the success path.
      noticeBox.textContent = data.message || 'If an account with that email exists, a reset link has been sent.';
      noticeBox.classList.add(VISIBLE_CLASS);
    } catch (err) {
      console.error('Forgot-password request failed:', err);
      errorBox.textContent = NETWORK_ERROR_MESSAGE;
      errorBox.classList.add(VISIBLE_CLASS);
    } finally {
      submitBtn.disabled = false;
    }
  });
}

function wireResetPasswordForm() {
  const form = document.getElementById('reset-password-form');
  if (!form) return;
  const errorBox = document.getElementById(AUTH_ERROR_ID);
  const submitBtn = form.querySelector(SUBMIT_BUTTON_SELECTOR);
  const token = new URLSearchParams(window.location.search).get('token') || '';

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    errorBox.classList.remove(VISIBLE_CLASS);
    submitBtn.disabled = true;

    const password = form.querySelector('#password').value;
    try {
      const res = await authFetch('/auth/reset-password', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token, password }),
      });
      const data = await res.json();
      if (!res.ok) {
        errorBox.textContent = data.error || 'Something went wrong.';
        errorBox.classList.add(VISIBLE_CLASS);
        submitBtn.disabled = false;
        return;
      }
      window.location.href = '/login';
    } catch (err) {
      console.error('Reset-password request failed:', err);
      errorBox.textContent = NETWORK_ERROR_MESSAGE;
      errorBox.classList.add(VISIBLE_CLASS);
      submitBtn.disabled = false;
    }
  });
}

function wireResendVerification(email) {
  const errorBox = document.getElementById(AUTH_ERROR_ID);
  const noticeBox = document.getElementById(AUTH_NOTICE_ID);
  return async () => {
    try {
      const res = await authFetch('/auth/resend-verification', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email }),
      });
      const data = await res.json();
      noticeBox.textContent = data.message || 'If an unverified account with that email exists, a new link has been sent.';
      noticeBox.classList.add(VISIBLE_CLASS);
      if (errorBox) errorBox.classList.remove(VISIBLE_CLASS);
    } catch (err) {
      console.error('Resend-verification request failed:', err);
      noticeBox.textContent = NETWORK_ERROR_MESSAGE;
      noticeBox.classList.add(VISIBLE_CLASS);
    }
  };
}

wireAuthForm('login-form', '/auth/login', {
  onError(data, email) {
    const resendBox = document.getElementById('resend-verification');
    const resendLink = document.getElementById('resend-verification-link');
    if (!resendBox || !resendLink) return;
    if (data.code === 'email_not_verified') {
      resendBox.style.display = '';
      resendLink.onclick = (e) => { e.preventDefault(); wireResendVerification(email)(); };
    } else {
      resendBox.style.display = 'none';
    }
  },
});

wireAuthForm('register-form', '/auth/register', {
  onSuccess(data) {
    const form = document.getElementById('register-form');
    const noticeBox = document.getElementById(AUTH_NOTICE_ID);
    noticeBox.textContent = data.message || 'Check your email for a link to verify your account before logging in.';
    noticeBox.classList.add(VISIBLE_CLASS);
    form.reset();
    form.querySelector(SUBMIT_BUTTON_SELECTOR).disabled = false;
  },
});

wireForgotPasswordForm();
wireResetPasswordForm();

// Standalone resend form on the verify-email error page (no email/password auth call —
// posts directly since /auth/resend-verification takes just an email).
(function wireResendVerificationForm() {
  const form = document.getElementById('resend-verification-form');
  if (!form) return;
  const submitBtn = form.querySelector(SUBMIT_BUTTON_SELECTOR);

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    submitBtn.disabled = true;
    const email = form.querySelector(EMAIL_INPUT_SELECTOR).value.trim();
    await wireResendVerification(email)();
    submitBtn.disabled = false;
  });
})();
