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

// ── Review panel ─────────────────────────────────────────────────────────────
const btnSave      = document.getElementById('btn-save');
const colName      = document.getElementById('col-name');
const saveStatus   = document.getElementById('save-status');
const simSlider    = document.getElementById('sim-slider');
const sliderVal    = document.getElementById('slider-val');
const selectedSpan = document.getElementById('selected-count');
const reviewItems  = document.querySelectorAll('.review-item');

function updateCount() {
  if (!selectedSpan) return;
  let n = 0;
  reviewItems.forEach(item => {
    if (!item.classList.contains('filtered') && item.querySelector('.art-check').checked) n++;
  });
  selectedSpan.textContent = n;
}

simSlider?.addEventListener('input', function () {
  const threshold = Number.parseInt(this.value);
  sliderVal.textContent = threshold + '%';
  reviewItems.forEach(item => {
    const filtered = Number.parseInt(item.dataset.sim) < threshold;
    item.classList.toggle('filtered', filtered);
    item.querySelector('.art-check').checked = !filtered;
  });
  updateCount();
});

document.getElementById('btn-select-all')?.addEventListener('click', () => {
  reviewItems.forEach(item => {
    if (!item.classList.contains('filtered'))
      item.querySelector('.art-check').checked = true;
  });
  updateCount();
});

document.getElementById('btn-deselect-all')?.addEventListener('click', () => {
  reviewItems.forEach(item => item.querySelector('.art-check').checked = false);
  updateCount();
});

reviewItems.forEach(item =>
  item.querySelector('.art-check')?.addEventListener('change', updateCount)
);

// ── Save ──────────────────────────────────────────────────────────────────────
const embedProgress     = document.getElementById('embed-progress');
const embedProgressBar  = document.getElementById('embed-progress-bar');
const embedProgressText = document.getElementById('embed-progress-text');

function _getSelectedArticles() {
  const selectedPmids = new Set(
    [...reviewItems]
      .filter(item => !item.classList.contains('filtered') && item.querySelector('.art-check').checked)
      .map(item => item.dataset.pmid)
  );
  return ARTICLES.filter(a => selectedPmids.has(a.pmid));
}

function _buildSourceSummary(evt) {
  const parts = [];
  if (evt.full_text) parts.push(`${evt.full_text} full text`);
  if (evt.abstract)  parts.push(`${evt.abstract} abstract`);
  if (evt.fallback)  parts.push(`${evt.fallback} fallback`);
  return parts.join(' · ');
}

function _initSaveUI() {
  btnSave.disabled = true;
  btnSave.textContent = 'Saving…';
  saveStatus.className = '';
  saveStatus.textContent = '';
  embedProgress.style.display = 'block';
  embedProgressBar.classList.remove('complete');
  embedProgressBar.style.width = '0%';
  embedProgressText.textContent = 'Starting…';
  bar.style.display = 'block';
  bar.style.transition = 'none';
  bar.style.width = '0%';
  requestAnimationFrame(() => Progress.set(10, 300));
}

function _onFetchEvent(evt) {
  const pct = Math.round((evt.done / evt.total) * 65) + 5;
  Progress.set(pct, 150);
  embedProgressBar.style.width = `${Math.round(evt.done / evt.total * 80)}%`;
  const summary = _buildSourceSummary(evt);
  embedProgressText.textContent = `Fetching ${evt.done} / ${evt.total}${summary ? ` · ${summary}` : ''}`;
  btnSave.textContent = `Fetching… ${evt.done} / ${evt.total}`;
}

function _onEmbeddingEvent(evt) {
  Progress.set(78, 200);
  embedProgressBar.style.width = '88%';
  embedProgressText.textContent =
    `Embedding ${evt.count} articles (${evt.full_text} full text · ${evt.abstract} abstract)…`;
  btnSave.textContent = 'Embedding…';
}

function _onDoneEvent(evt) {
  Progress.set(95, 200);
  setTimeout(() => Progress.done(), 300);
  embedProgressBar.style.width = '100%';
  embedProgressBar.classList.add('complete');
  const summary = _buildSourceSummary(evt);
  const summaryText = summary ? ` (${summary})` : '';
  embedProgressText.textContent = `Done — ${evt.count} articles embedded${summaryText}`;
  saveStatus.className = 'ok';
  saveStatus.innerHTML =
    `Saved! ${evt.count} articles embedded${summaryText}. ` +
    `<a href="/collections/${evt.id}">Open collection &rarr;</a>`;
  btnSave.textContent = 'Saved';
}

async function* _readSaveLines(res) {
  const reader  = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer    = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop();
    for (const line of lines) {
      if (line.startsWith('data: ')) yield line;
    }
  }
}

async function _dispatchSaveEvents(res) {
  for await (const line of _readSaveLines(res)) {
    const evt = JSON.parse(line.slice(6));
    if (evt.type === 'fetch')     { _onFetchEvent(evt); continue; }
    if (evt.type === 'embedding') { _onEmbeddingEvent(evt); continue; }
    if (evt.type === 'done')      { _onDoneEvent(evt); continue; }
    if (evt.type === 'error')     throw new Error(evt.message);
  }
}

// ── Save ──────────────────────────────────────────────────────────────────────────────
async function saveCollection() {
  const name = colName.value.trim();
  if (!name) { colName.focus(); return; }

  const selectedArticles = _getSelectedArticles();
  if (!selectedArticles.length) {
    saveStatus.className = 'err';
    saveStatus.textContent = 'Select at least one article.';
    return;
  }

  _initSaveUI();

  try {
    const res = await fetch('/collections/save-stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, user_query: USER_QUERY, pubmed_query: PUBMED_QUERY, articles: selectedArticles }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error || `Server error ${res.status}`);
    }
    await _dispatchSaveEvents(res);
  } catch (err) {
    Progress.error();
    embedProgress.style.display = 'none';
    saveStatus.className = 'err';
    saveStatus.textContent = 'Error: ' + err.message;
    btnSave.disabled = false;
    btnSave.textContent = 'Save as RAG Collection';
  }
}

if (btnSave) btnSave.addEventListener('click', saveCollection);

// ── Search form progress ──────────────────────────────────────────────────────
const form    = document.querySelector('form');
const btn     = document.querySelector('.btn-search');
const overlay = document.getElementById('loading-overlay');
const steps   = [
  document.getElementById('step-llm'),
  document.getElementById('step-search'),
  document.getElementById('step-fetch'),
];

const PHASE_DURATIONS = [6000, 2000, 3000];

function setStep(index) {
  steps.forEach((el, i) => {
    el.classList.toggle('done',   i < index);
    el.classList.toggle('active', i === index);
  });
}

form.addEventListener('submit', function () {
  if (!document.getElementById('query').value.trim()) return;

  btn.disabled = true;
  btn.textContent = 'Searching…';
  overlay.classList.add('visible');

  bar.style.display = 'block';
  bar.style.transition = 'none';
  bar.style.width = '0%';
  setStep(0);

  requestAnimationFrame(() => {
    bar.style.transition = `width ${PHASE_DURATIONS[0]}ms ease`;
    bar.style.width = '40%';
  });

  setTimeout(() => {
    setStep(1);
    bar.style.transition = `width ${PHASE_DURATIONS[1]}ms ease`;
    bar.style.width = '65%';
  }, PHASE_DURATIONS[0]);

  setTimeout(() => {
    setStep(2);
    bar.style.transition = `width ${PHASE_DURATIONS[2]}ms ease`;
    bar.style.width = '88%';
  }, PHASE_DURATIONS[0] + PHASE_DURATIONS[1]);
});
