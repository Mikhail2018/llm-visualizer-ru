const el = {
  prompt: document.getElementById('prompt'),
  run: document.getElementById('run'),
  mode: document.getElementById('mode'),
  tokens: document.getElementById('tokens'),
  embeddings: document.getElementById('embeddings'),
  attentionMatrix: document.getElementById('attentionMatrix'),
  next: document.getElementById('next'),
  candidates: document.getElementById('candidates')
};

function tokenize(text) {
  return text
    .replace(/[.,!?;:()]/g, ' ')
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 12);
}

function pseudoValue(token, salt = 1) {
  let acc = 0;
  for (let i = 0; i < token.length; i++) acc += token.charCodeAt(i) * (i + salt);
  return (acc % 100) / 100;
}

function renderTokens(tokens) {
  el.tokens.innerHTML = tokens.map(t => `<span class="chip">${t}</span>`).join('');
}

function renderEmbeddings(tokens) {
  el.embeddings.innerHTML = tokens.map(t => {
    const v = Math.max(8, Math.round(pseudoValue(t, 3) * 100));
    return `<div class="embRow"><div class="embLabel">${t}</div><div class="barWrap"><div class="bar" style="width:${v}%"></div></div></div>`;
  }).join('');
}

function attentionWeight(a, b, mode) {
  const base = 0.15 + ((a + 1) / (b + 2));
  const noise = (Math.sin((a + 1) * (b + 2)) + 1) / 10;
  const m = mode === 'advanced' ? 1.25 : 1;
  return Math.min(0.99, +(Math.max(0.01, (base * 0.3 + noise) * m).toFixed(2)));
}

function renderAttention(tokens, mode) {
  const n = tokens.length;
  const rows = [];
  for (let i = 0; i < n; i++) {
    const cells = [];
    for (let j = 0; j < n; j++) {
      const w = attentionWeight(i, j, mode);
      const hue = 210;
      const light = 18 + Math.round(w * 38);
      cells.push(`<div class="cell" style="background:hsl(${hue} 70% ${light}%)">${w}</div>`);
    }
    rows.push(`<div class="matrixRow">${cells.join('')}</div>`);
  }
  el.attentionMatrix.innerHTML = rows.join('');
}

function nextTokenCandidates(tokens, mode) {
  const bank = ['и', 'это', 'потому', 'что', 'модель', 'предсказывает', 'контекст', 'далее'];
  return bank
    .map((w, idx) => ({ w, p: +(0.06 + ((idx + 1) / 100) + (tokens.length % 5) / 50 + (mode === 'advanced' ? 0.01 : 0)).toFixed(2) }))
    .sort((a, b) => b.p - a.p)
    .slice(0, 5);
}

function renderNext(tokens, mode) {
  const cands = nextTokenCandidates(tokens, mode);
  el.next.textContent = `Выбранный токен (top-1): «${cands[0].w}»`;
  el.candidates.innerHTML = cands.map(c => `<div class="cand"><span>${c.w}</span><span>${c.p}</span></div>`).join('');
}

function renderAll() {
  const text = el.prompt.value.trim();
  const tokens = tokenize(text);
  const mode = el.mode.value;
  renderTokens(tokens);
  renderEmbeddings(tokens);
  renderAttention(tokens, mode);
  renderNext(tokens, mode);
}

el.run.addEventListener('click', renderAll);
el.mode.addEventListener('change', renderAll);
renderAll();
