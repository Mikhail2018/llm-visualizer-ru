const el = {
  prompt: document.getElementById('prompt'),
  stepPrev: document.getElementById('stepPrev'),
  stepNext: document.getElementById('stepNext'),
  play: document.getElementById('play'),
  pause: document.getElementById('pause'),
  stageLabel: document.getElementById('stageLabel'),
  stageTitle: document.getElementById('stageTitle'),
  stageDesc: document.getElementById('stageDesc'),
  stageView: document.getElementById('stageView'),
  pipeline: document.getElementById('pipeline'),
  formula: document.getElementById('formula'),
  why: document.getElementById('why')
};

const steps = [
  { key:'tok', title:'1) Токенизация', formula:'x -> [t1, t2, ... tn]', why:'Модель работает с токенами, а не с цельной строкой.' },
  { key:'emb', title:'2) Embedding + Positional Encoding', formula:'E = TokenEmbedding(tokens) + PosEncoding', why:'Добавляем смысл токена + информацию о его позиции.' },
  { key:'qkv', title:'3) Q / K / V проекции', formula:'Q = E*Wq, K = E*Wk, V = E*Wv', why:'Готовим представления для вычисления внимания.' },
  { key:'attn', title:'4) Self-Attention + Softmax + Causal Mask', formula:'A = softmax((QK^T / sqrt(dk)) + mask)\nO = A*V', why:'Определяем, на какие токены смотреть сильнее.' },
  { key:'addnorm1', title:'5) Add & Norm #1', formula:'H1 = LayerNorm(E + O)', why:'Residual + нормализация стабилизируют поток градиента.' },
  { key:'ffn', title:'6) Feed Forward (MLP)', formula:'F = W2 * GELU(W1*H1 + b1) + b2', why:'Нелинейная обработка каждого токена отдельно.' },
  { key:'addnorm2', title:'7) Add & Norm #2', formula:'H2 = LayerNorm(H1 + F)', why:'Второй residual-блок после MLP.' },
  { key:'logits', title:'8) Logits -> Next token', formula:'logits = H2_last * W_vocab\np = softmax(logits)', why:'Получаем вероятности следующего токена.' }
];

let idx = 0;
let timer = null;

function tokenize(text){
  return text.replace(/[.,!?;:()]/g,' ').split(/\s+/).filter(Boolean).slice(0,10);
}
function v(token,s=1){
  let a=0; for(let i=0;i<token.length;i++) a += token.charCodeAt(i)*(i+s);
  return (a%100)/100;
}
function drawTokens(tokens){
  return `<div class="chips">${tokens.map(t=>`<span class="chip">${t}</span>`).join('')}</div>`;
}
function drawBars(tokens, salt){
  return tokens.map(t=>{const p=Math.round((0.1+v(t,salt)*0.9)*100); return `<div><div class="muted">${t} — ${p}%</div><div class="bar"><i style="width:${p}%"></i></div></div>`;}).join('');
}
function drawAttn(tokens){
  const n=tokens.length||1;
  let rows=[];
  for(let i=0;i<n;i++){
    let cells=[];
    for(let j=0;j<n;j++){
      const masked = j>i;
      const w = masked ? 0 : +(0.05 + ((Math.sin((i+1)*(j+2))+1)/4)).toFixed(2);
      const light = masked ? 12 : 18 + Math.round(w*40);
      cells.push(`<div class="cell" style="background:hsl(210 70% ${light}%);">${masked?'×':w}</div>`);
    }
    rows.push(`<div class="rowm">${cells.join('')}</div>`);
  }
  return `<div class="matrix">${rows.join('')}</div>`;
}

function renderStep(){
  const tokens = tokenize(el.prompt.value.trim() || 'трансформер работает');
  const s = steps[idx];
  el.stageTitle.textContent = s.title;
  el.stageDesc.textContent = 'Пайплайн одного декодер-блока в учебном представлении.';
  el.formula.textContent = s.formula;
  el.why.textContent = s.why;
  el.stageLabel.textContent = `Шаг: ${idx+1} / ${steps.length}`;

  let html='';
  if (s.key==='tok') html = drawTokens(tokens);
  if (s.key==='emb') html = drawBars(tokens,3);
  if (s.key==='qkv') html = `<b>Q</b>${drawBars(tokens,5)}<b>K</b>${drawBars(tokens,7)}<b>V</b>${drawBars(tokens,9)}`;
  if (s.key==='attn') html = drawAttn(tokens);
  if (s.key==='addnorm1' || s.key==='addnorm2') html = drawBars(tokens,11+idx);
  if (s.key==='ffn') html = drawBars(tokens,21);
  if (s.key==='logits') html = drawBars(['и','что','это','модель','контекст'],31);
  el.stageView.innerHTML = html;

  el.pipeline.innerHTML = steps.map((x,i)=>`<li class="${i===idx?'active':''}">${x.title}</li>`).join('');
}

function next(){ idx = (idx+1)%steps.length; renderStep(); }
function prev(){ idx = (idx-1+steps.length)%steps.length; renderStep(); }
function play(){ if(timer) return; timer=setInterval(next,1200); }
function pause(){ if(timer) clearInterval(timer); timer=null; }

el.stepNext.addEventListener('click', ()=>{ pause(); next(); });
el.stepPrev.addEventListener('click', ()=>{ pause(); prev(); });
el.play.addEventListener('click', play);
el.pause.addEventListener('click', pause);
el.prompt.addEventListener('input', ()=>{ pause(); renderStep(); });

renderStep();
