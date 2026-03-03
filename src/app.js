const el = {
  prompt: document.getElementById('prompt'),
  run: document.getElementById('run'),
  mode: document.getElementById('mode'),
  tokens: document.getElementById('tokens'),
  embeddings: document.getElementById('embeddings'),
  attentionMatrix: document.getElementById('attentionMatrix'),
  next: document.getElementById('next'),
  candidates: document.getElementById('candidates'),
  qkvToken: document.getElementById('qkvToken'),
  qVec: document.getElementById('qVec'),
  kVec: document.getElementById('kVec'),
  vVec: document.getElementById('vVec'),
  softmaxBars: document.getElementById('softmaxBars')
};

const stepUI = document.createElement('section');
stepUI.className = 'panel controls';
stepUI.innerHTML = `
  <div class="row">
    <button id="play">▶ Play</button>
    <button id="pause">⏸ Pause</button>
    <button id="step">⏭ Шаг</button>
    <span id="stageLabel" class="muted">Стадия: ожидание</span>
  </div>
`;
document.querySelector('main.container').appendChild(stepUI);

const controls = {
  play: document.getElementById('play'),
  pause: document.getElementById('pause'),
  step: document.getElementById('step'),
  stageLabel: document.getElementById('stageLabel')
};

let stageIndex = 0;
let timer = null;
const stages = ['tokens', 'embeddings', 'attention', 'next', 'qkv'];

function tokenize(text) {
  return text.replace(/[.,!?;:()]/g, ' ').split(/\s+/).filter(Boolean).slice(0, 12);
}

function pseudoValue(token, salt = 1) {
  let acc = 0;
  for (let i = 0; i < token.length; i++) acc += token.charCodeAt(i) * (i + salt);
  return (acc % 100) / 100;
}

function vecForToken(token, kind='q') {
  const seed = kind==='q' ? 3 : kind==='k' ? 5 : 7;
  return [0,1,2,3].map(i => +((pseudoValue(token, seed+i)*2-1).toFixed(2)));
}

function dot(a,b){ return a.reduce((s,v,i)=>s+v*b[i],0); }

function softmax(arr){
  const m=Math.max(...arr);
  const ex=arr.map(v=>Math.exp(v-m));
  const sum=ex.reduce((a,b)=>a+b,0);
  return ex.map(v=>v/sum);
}

function highlightStage(name) {
  const map = {
    tokens: el.tokens.closest('.panel'),
    embeddings: el.embeddings.closest('.panel'),
    attention: el.attentionMatrix.closest('.panel'),
    next: el.next.closest('.panel'),
    qkv: el.qVec.closest('section.panel')
  };
  Object.values(map).forEach(p => p && p.classList.remove('activeStage'));
  map[name]?.classList.add('activeStage');
  const labels = { tokens:'токенизация', embeddings:'эмбеддинги', attention:'attention', next:'выбор следующего токена', qkv:'Q/K/V + softmax' };
  controls.stageLabel.textContent = `Стадия: ${labels[name] || 'ожидание'}`;
}

function renderTokens(tokens){ el.tokens.innerHTML=tokens.map(t=>`<span class="chip">${t}</span>`).join(''); }
function renderEmbeddings(tokens){
  el.embeddings.innerHTML=tokens.map(t=>{
    const v=Math.max(8,Math.round(pseudoValue(t,3)*100));
    return `<div class="embRow"><div class="embLabel">${t}</div><div class="barWrap"><div class="bar" style="width:${v}%"></div></div></div>`;
  }).join('');
}
function attentionWeight(a,b,mode){ const base=0.15+((a+1)/(b+2)); const noise=(Math.sin((a+1)*(b+2))+1)/10; const m=mode==='advanced'?1.25:1; return Math.min(0.99, +(Math.max(0.01,(base*0.3+noise)*m).toFixed(2))); }
function renderAttention(tokens,mode){
  const n=tokens.length, rows=[];
  for(let i=0;i<n;i++){ const cells=[]; for(let j=0;j<n;j++){ const w=attentionWeight(i,j,mode); const light=18+Math.round(w*38); cells.push(`<div class="cell" style="background:hsl(210 70% ${light}%)">${w}</div>`);} rows.push(`<div class="matrixRow">${cells.join('')}</div>`);} el.attentionMatrix.innerHTML=rows.join('');
}
function nextTokenCandidates(tokens,mode){ const bank=['и','это','потому','что','модель','предсказывает','контекст','далее']; return bank.map((w,idx)=>({w,p:+(0.06+((idx+1)/100)+(tokens.length%5)/50+(mode==='advanced'?0.01:0)).toFixed(2)})).sort((a,b)=>b.p-a.p).slice(0,5); }
function renderNext(tokens,mode){ const cands=nextTokenCandidates(tokens,mode); el.next.textContent=`Выбранный токен (top-1): «${cands[0].w}»`; el.candidates.innerHTML=cands.map(c=>`<div class="cand"><span>${c.w}</span><span>${c.p}</span></div>`).join(''); }

function updateTokenSelector(tokens){
  const prev=el.qkvToken.value;
  el.qkvToken.innerHTML=tokens.map((t,i)=>`<option value="${i}">${t}</option>`).join('');
  if(prev && +prev<tokens.length) el.qkvToken.value=prev;
}

function renderQKV(tokens){
  if(!tokens.length){ el.qVec.textContent=''; el.kVec.textContent=''; el.vVec.textContent=''; el.softmaxBars.innerHTML=''; return; }
  updateTokenSelector(tokens);
  const qi=+el.qkvToken.value || 0;
  const qTok=tokens[qi];
  const q=vecForToken(qTok,'q');
  const kAll=tokens.map(t=>vecForToken(t,'k'));
  const vAll=tokens.map(t=>vecForToken(t,'v'));
  const scores=kAll.map(k=>dot(q,k));
  const probs=softmax(scores);

  el.qVec.textContent=`${qTok}\n[${q.join(', ')}]`;
  el.kVec.textContent=tokens.map((t,i)=>`${t}: [${kAll[i].join(', ')}]`).join('\n');
  el.vVec.textContent=tokens.map((t,i)=>`${t}: [${vAll[i].join(', ')}]`).join('\n');

  el.softmaxBars.innerHTML=probs.map((p,i)=>{
    const pct=Math.round(p*100);
    return `<div class="sbar"><div class="lbl">${tokens[i]} — ${pct}%</div><div class="wrap"><div class="fill" style="width:${pct}%"></div></div></div>`;
  }).join('');
}

function clearOutputs(){ el.tokens.innerHTML=''; el.embeddings.innerHTML=''; el.attentionMatrix.innerHTML=''; el.next.textContent=''; el.candidates.innerHTML=''; }

function renderByStage(stageName,tokens,mode){
  if(stageName==='tokens') renderTokens(tokens);
  if(stageName==='embeddings') renderEmbeddings(tokens);
  if(stageName==='attention') renderAttention(tokens,mode);
  if(stageName==='next') renderNext(tokens,mode);
  if(stageName==='qkv') renderQKV(tokens);
  highlightStage(stageName);
}

function runStep(){ const tokens=tokenize(el.prompt.value.trim()); const mode=el.mode.value; if(stageIndex===0) clearOutputs(); const stageName=stages[stageIndex]; renderByStage(stageName,tokens,mode); stageIndex=(stageIndex+1)%stages.length; }
function play(){ if(timer) return; runStep(); timer=setInterval(runStep,900); }
function pause(){ if(timer) clearInterval(timer); timer=null; }

function renderAll(){ pause(); stageIndex=0; const tokens=tokenize(el.prompt.value.trim()); const mode=el.mode.value; renderTokens(tokens); renderEmbeddings(tokens); renderAttention(tokens,mode); renderNext(tokens,mode); renderQKV(tokens); highlightStage('qkv'); }

el.run.addEventListener('click', renderAll);
el.mode.addEventListener('change', renderAll);
el.qkvToken?.addEventListener('change', ()=>renderQKV(tokenize(el.prompt.value.trim())));
controls.play.addEventListener('click', play);
controls.pause.addEventListener('click', pause);
controls.step.addEventListener('click', ()=>{ pause(); runStep(); });

renderAll();
