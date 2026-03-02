const el = {
  prompt: document.getElementById('prompt'),
  run: document.getElementById('run'),
  tokens: document.getElementById('tokens'),
  attention: document.getElementById('attention'),
  next: document.getElementById('next')
};

function tokenize(text) {
  return text.split(/\s+/).filter(Boolean);
}

function render(tokens) {
  el.tokens.innerHTML = tokens.map(t => `<span class="chip">${t}</span>`).join('');
  el.attention.textContent = tokens.map((t,i)=>`${t} → вес контекста ~ ${(1/(i+1)).toFixed(2)}`).join('\n');
  el.next.textContent = 'Кандидат next token: "..." (демо-режим)';
}

el.run.addEventListener('click', () => render(tokenize(el.prompt.value.trim())));
render(tokenize(el.prompt.value.trim()));
