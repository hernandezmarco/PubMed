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
  const threshold = parseInt(this.value);
  sliderVal.textContent = threshold + '%';
  reviewItems.forEach(item => {
    const filtered = parseInt(item.dataset.sim) < threshold;
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
if (btnSave) {
  btnSave.addEventListener('click', async () => {
    const name = colName.value.trim();
    if (!name) { colName.focus(); return; }

    const selectedPmids = new Set();
    reviewItems.forEach(item => {
      if (!item.classList.contains('filtered') && item.querySelector('.art-check').checked)
        selectedPmids.add(item.dataset.pmid);
    });
    const selectedArticles = ARTICLES.filter(a => selectedPmids.has(a.pmid));
    if (!selectedArticles.length) {
      saveStatus.className = 'err';
      saveStatus.textContent = 'Select at least one article.';
      return;
    }

    btnSave.disabled = true;
    btnSave.textContent = 'Saving… (fetching full text)';
    saveStatus.className = '';
    saveStatus.textContent = '';

    bar.style.display = 'block';
    bar.style.transition = 'none';
    bar.style.width = '0%';
    requestAnimationFrame(() => Progress.set(30, 500));

    try {
      const res = await fetch('/collections', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name,
          user_query: USER_QUERY,
          pubmed_query: PUBMED_QUERY,
          articles: selectedArticles,
        }),
      });
      const data = await res.json();
      if (data.id) {
        Progress.set(90, 200);
        setTimeout(() => Progress.done(), 200);
        saveStatus.className = 'ok';
        saveStatus.innerHTML = `Saved! <a href="/collections/${data.id}">Open collection &rarr;</a>`;
        btnSave.textContent = 'Saved';
      } else {
        throw new Error(data.error || 'Unknown error');
      }
    } catch (err) {
      Progress.error();
      saveStatus.className = 'err';
      saveStatus.textContent = 'Error: ' + err.message;
      btnSave.disabled = false;
      btnSave.textContent = 'Save as RAG Collection';
    }
  });
}

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
