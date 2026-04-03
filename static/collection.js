marked.setOptions({ breaks: true });

const chatMessages  = document.getElementById('chat-messages');
const questionInput = document.getElementById('question');
const btnAsk        = document.getElementById('btn-ask');
const askStatus     = document.getElementById('ask-status');
const stepEls       = {
  embed:    document.getElementById('step-embed'),
  search:   document.getElementById('step-search'),
  generate: document.getElementById('step-generate'),
};

// ── Step status ───────────────────────────────────────────────────────────────
function setStep(active) {
  const order = ['embed', 'search', 'generate'];
  const idx   = order.indexOf(active);
  order.forEach((k, i) => {
    stepEls[k].classList.toggle('done',   i < idx);
    stepEls[k].classList.toggle('active', i === idx);
  });
  askStatus.classList.add('visible');
}

function clearSteps() {
  askStatus.classList.remove('visible');
  Object.values(stepEls).forEach(el => el.classList.remove('active', 'done'));
}

// ── Message helpers ───────────────────────────────────────────────────────────
function appendUserMsg(text) {
  document.getElementById('empty-hint')?.remove();
  const el = document.createElement('div');
  el.className = 'msg msg-user';
  el.textContent = text;
  chatMessages.appendChild(el);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function appendAssistantMsg() {
  const el = document.createElement('div');
  el.className = 'msg msg-assistant';
  el.innerHTML = '<span class="answer"></span><span class="typing-cursor"></span>';
  chatMessages.appendChild(el);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return el;
}

function renderSuggestions(questions) {
  if (!questions || !questions.length) return null;
  const row = document.createElement('div');
  row.className = 'suggestions-row';
  questions.forEach(q => {
    const chip = document.createElement('button');
    chip.className = 'suggestion-chip';
    chip.textContent = q;
    chip.addEventListener('click', () => {
      questionInput.value = q;
      questionInput.focus();
    });
    row.appendChild(chip);
  });
  return row;
}

function finalizeMsgWithCitations(msgEl, citations, suggestions) {
  msgEl.querySelector('.typing-cursor')?.remove();

  if (citations.length) {
    const citDiv = document.createElement('div');
    citDiv.className = 'citations';
    citDiv.innerHTML = '<div class="citations-label">Sources retrieved</div>';

    citations.forEach((art, i) => {
      const item = document.createElement('div');
      item.className = 'citation-item';
      item.innerHTML =
        `<span class="citation-num">[${i + 1}]</span>` +
        `<span><a href="${art.url}" target="_blank" rel="noopener">${art.title}</a>` +
        ` <em style="color:#888;font-size:.75rem">${art.journal}${art.year ? ' ' + art.year : ''}</em>` +
        `<span class="sim-badge">${(art.similarity * 100).toFixed(0)}% match</span></span>`;
      citDiv.appendChild(item);
      document.getElementById(`art-${art.pmid}`)?.classList.add('highlighted');
    });

    msgEl.appendChild(citDiv);
  }

  const suggRow = renderSuggestions(suggestions);
  if (suggRow) msgEl.appendChild(suggRow);
}

// ── Ask ───────────────────────────────────────────────────────────────────────
async function ask() {
  const question = questionInput.value.trim();
  if (!question) return;

  questionInput.value = '';
  btnAsk.disabled = true;
  btnAsk.textContent = '…';

  document.querySelectorAll('.art-item.highlighted')
    .forEach(el => el.classList.remove('highlighted'));

  appendUserMsg(question);
  const msgEl    = appendAssistantMsg();
  const answerEl = msgEl.querySelector('.answer');
  let rawAnswer  = '';
  let firstToken = false;

  Progress.start(12, 60);
  setStep('embed');

  try {
    const res = await fetch(`/collections/${COLLECTION_ID}/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });

    setStep('search');
    Progress.set(40, 300);

    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer    = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const payload = JSON.parse(line.slice(6));
        if (payload.text) {
          if (!firstToken) {
            firstToken = true;
            setStep('generate');
            Progress.set(55, 200);
          }
          rawAnswer += payload.text;
          answerEl.innerHTML = marked.parse(rawAnswer);
          chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        if (payload.done) {
          finalizeMsgWithCitations(msgEl, payload.citations || [], payload.suggestions || []);
          Progress.done();
          clearSteps();
        }
      }
    }
  } catch (err) {
    answerEl.textContent = 'Error: ' + err.message;
    msgEl.querySelector('.typing-cursor')?.remove();
    Progress.error();
    clearSteps();
  }

  btnAsk.disabled = false;
  btnAsk.textContent = 'Ask';
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

btnAsk.addEventListener('click', ask);
questionInput.addEventListener('keydown', e => { if (e.key === 'Enter') ask(); });

// Starter suggestion chips
document.querySelectorAll('.starter-chip').forEach(chip => {
  chip.addEventListener('click', () => {
    questionInput.value = chip.textContent.trim();
    questionInput.focus();
  });
});
