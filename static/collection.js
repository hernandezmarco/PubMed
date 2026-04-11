marked.setOptions({ breaks: true });

const chatMessages   = document.getElementById('chat-messages');
const questionInput  = document.getElementById('question');
const btnAsk         = document.getElementById('btn-ask');
const modelSelect    = document.getElementById('model-select');
const convList       = document.getElementById('conv-list');
const btnNewChat     = document.getElementById('btn-new-chat');
const chatToolbar    = document.getElementById('chat-toolbar');
const btnExport      = document.getElementById('btn-export');
const exportDropdown = document.getElementById('export-dropdown');

// ── Conversation state ────────────────────────────────────────────────────────
let currentConversationId = null;

// ── Model persistence ─────────────────────────────────────────────────────────
const MODEL_KEY = 'pubmed_selected_model';
const savedModel = localStorage.getItem(MODEL_KEY);
if (savedModel) {
  const opt = modelSelect.querySelector(`option[value="${savedModel}"]`);
  if (opt) modelSelect.value = savedModel;
}
modelSelect.addEventListener('change', () => {
  localStorage.setItem(MODEL_KEY, modelSelect.value);
});

// ── Export dropdown ───────────────────────────────────────────────────────────
btnExport.addEventListener('click', e => {
  e.stopPropagation();
  exportDropdown.classList.toggle('open');
});

document.addEventListener('click', () => exportDropdown.classList.remove('open'));

document.getElementById('export-docx').addEventListener('click', () => {
  if (currentConversationId) {
    window.location.href = `/conversations/${currentConversationId}/export?format=docx`;
  }
  exportDropdown.classList.remove('open');
});

document.getElementById('export-rtf').addEventListener('click', () => {
  if (currentConversationId) {
    window.location.href = `/conversations/${currentConversationId}/export?format=rtf`;
  }
  exportDropdown.classList.remove('open');
});

const askStatus = document.getElementById('ask-status');
const stepEls   = {
  embed:    document.getElementById('step-embed'),
  search:   document.getElementById('step-search'),
  generate: document.getElementById('step-generate'),
};

// ── Utilities ─────────────────────────────────────────────────────────────────
function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function formatDate(isoStr) {
  const d     = new Date(isoStr);
  const today = new Date();
  const yest  = new Date(today);
  yest.setDate(today.getDate() - 1);
  if (d.toDateString() === today.toDateString()) return 'Today';
  if (d.toDateString() === yest.toDateString())  return 'Yesterday';
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

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

// ── Message rendering ─────────────────────────────────────────────────────────
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
        `<span><a href="${art.url}" target="_blank" rel="noopener">${escapeHtml(art.title)}</a>` +
        ` <em style="color:#888;font-size:.75rem">${escapeHtml(art.journal)}${art.year ? ' ' + art.year : ''}</em>` +
        `<span class="sim-badge">${(art.similarity * 100).toFixed(0)}% match</span></span>`;
      citDiv.appendChild(item);
      document.getElementById(`art-${art.pmid}`)?.classList.add('highlighted');
    });

    msgEl.appendChild(citDiv);
  }

  const suggRow = renderSuggestions(suggestions);
  if (suggRow) msgEl.appendChild(suggRow);
}

// ── Conversation sidebar ──────────────────────────────────────────────────────
async function loadConversations() {
  try {
    const res  = await fetch(`/collections/${COLLECTION_ID}/conversations`);
    const list = await res.json();
    renderConversationList(list);
    if (list.length > 0) {
      await loadConversation(list[0].id);
    }
  } catch (err) {
    console.error('Failed to load conversations:', err);
  }
}

function renderConversationList(list) {
  convList.innerHTML = '';
  if (list.length === 0) {
    convList.innerHTML = '<div class="conv-empty">No conversations yet</div>';
    return;
  }
  list.forEach(c => {
    const item = document.createElement('div');
    item.className = 'conv-item' + (c.id === currentConversationId ? ' active' : '');
    item.dataset.id = c.id;
    item.innerHTML =
      `<div class="conv-item-title">${escapeHtml(c.title)}</div>` +
      `<div class="conv-item-date">${formatDate(c.created_at)}</div>` +
      `<button class="conv-delete" title="Delete conversation" data-id="${c.id}">&times;</button>`;

    item.addEventListener('click', e => {
      if (e.target.classList.contains('conv-delete')) return;
      loadConversation(c.id);
    });

    item.querySelector('.conv-delete').addEventListener('click', async e => {
      e.stopPropagation();
      if (!confirm('Delete this conversation?')) return;
      await fetch(`/conversations/${c.id}/delete`, { method: 'POST' });
      if (currentConversationId === c.id) newConversation();
      await refreshConversationList();
    });

    convList.appendChild(item);
  });
}

async function refreshConversationList() {
  try {
    const res  = await fetch(`/collections/${COLLECTION_ID}/conversations`);
    const list = await res.json();
    renderConversationList(list);
  } catch (err) {
    console.error('Failed to refresh conversations:', err);
  }
}

async function loadConversation(vid) {
  currentConversationId = vid;
  chatToolbar.classList.add('visible');

  // Update active highlight in sidebar
  document.querySelectorAll('.conv-item').forEach(el => {
    el.classList.toggle('active', parseInt(el.dataset.id) === vid);
  });

  try {
    const res      = await fetch(`/conversations/${vid}/messages`);
    const messages = await res.json();

    chatMessages.innerHTML = '';
    document.querySelectorAll('.art-item.highlighted')
      .forEach(el => el.classList.remove('highlighted'));

    messages.forEach(msg => {
      if (msg.role === 'user') {
        const el = document.createElement('div');
        el.className = 'msg msg-user';
        el.textContent = msg.content;
        chatMessages.appendChild(el);
      } else {
        const el = document.createElement('div');
        el.className = 'msg msg-assistant';
        const ansEl = document.createElement('span');
        ansEl.className = 'answer';
        ansEl.innerHTML = marked.parse(msg.content);
        el.appendChild(ansEl);
        chatMessages.appendChild(el);
      }
    });

    chatMessages.scrollTop = chatMessages.scrollHeight;
  } catch (err) {
    console.error('Failed to load conversation:', err);
  }
}

function newConversation() {
  currentConversationId = null;
  chatToolbar.classList.remove('visible');
  document.querySelectorAll('.conv-item').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.art-item.highlighted')
    .forEach(el => el.classList.remove('highlighted'));

  chatMessages.innerHTML = '';
  const hint = document.createElement('div');
  hint.className = 'empty-chat';
  hint.id = 'empty-hint';
  hint.innerHTML =
    '<div class="icon">&#128270;</div>' +
    '<div>Ask a question about this collection</div>' +
    '<div style="font-size:.8rem;color:#bbb">Claude answers using article chunks (full text where available) and cites sources</div>';

  if (STARTER_QUESTIONS && STARTER_QUESTIONS.length) {
    const row = document.createElement('div');
    row.className = 'suggestions-row';
    row.style.cssText = 'justify-content:center;margin-top:1rem;';
    STARTER_QUESTIONS.forEach(q => {
      const chip = document.createElement('button');
      chip.className = 'suggestion-chip starter-chip';
      chip.textContent = q;
      chip.addEventListener('click', () => {
        questionInput.value = q;
        questionInput.focus();
      });
      row.appendChild(chip);
    });
    hint.appendChild(row);
  }

  chatMessages.appendChild(hint);
}

btnNewChat.addEventListener('click', newConversation);

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

  const wasNewConversation = currentConversationId === null;

  try {
    const res = await fetch(`/collections/${COLLECTION_ID}/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        model: modelSelect.value,
        conversation_id: currentConversationId,
      }),
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
          currentConversationId = payload.conversation_id;
          chatToolbar.classList.add('visible');
          finalizeMsgWithCitations(msgEl, payload.citations || [], payload.suggestions || []);
          Progress.done();
          clearSteps();

          // Refresh sidebar when a new conversation was created
          if (wasNewConversation) await refreshConversationList();
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

// Starter suggestion chips (server-rendered, shown before any conversation is loaded)
document.querySelectorAll('.starter-chip').forEach(chip => {
  chip.addEventListener('click', () => {
    questionInput.value = chip.textContent.trim();
    questionInput.focus();
  });
});

// ── Init ──────────────────────────────────────────────────────────────────────
loadConversations();
