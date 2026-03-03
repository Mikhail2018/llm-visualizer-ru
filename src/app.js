import * as THREE from 'https://unpkg.com/three@0.160.0/build/three.module.js';

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
  headSelect: document.getElementById('headSelect'),
  qVec: document.getElementById('qVec'),
  kVec: document.getElementById('kVec'),
  vVec: document.getElementById('vVec'),
  softmaxBars: document.getElementById('softmaxBars'),
  headsCompare: document.getElementById('headsCompare'),
  llm3d: document.getElementById('llm3d'),
  dataMode: document.getElementById('dataMode'),
  loadReal: document.getElementById('loadReal'),
  realStatus: document.getElementById('realStatus'),
  realTopK: document.getElementById('realTopK'),
  realModelId: document.getElementById('realModelId')
};

const stepUI = document.createElement('section');
stepUI.className = 'panel controls';
stepUI.innerHTML = `<div class="row"><button id="play">▶ Play</button><button id="pause">⏸ Pause</button><button id="step">⏭ Шаг</button><span id="stageLabel" class="muted">Стадия: ожидание</span></div>`;
document.querySelector('main.container').appendChild(stepUI);
const controls = { play: document.getElementById('play'), pause: document.getElementById('pause'), step: document.getElementById('step'), stageLabel: document.getElementById('stageLabel') };

let stageIndex = 0, timer = null;
const stages = ['tokens', 'embeddings', 'attention', 'next', 'qkv'];

function tokenize(text){ return text.replace(/[.,!?;:()]/g,' ').split(/\s+/).filter(Boolean).slice(0,12); }
function pseudoValue(token,s=1){ let a=0; for(let i=0;i<token.length;i++) a+=token.charCodeAt(i)*(i+s); return (a%100)/100; }
function vecForToken(token,kind='q',head=1){ const ho=head*11, b=kind==='q'?3:kind==='k'?5:7; return [0,1,2,3].map(i=>+((pseudoValue(token,b+i+ho)*2-1).toFixed(2))); }
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
function softmax(arr){ const m=Math.max(...arr), ex=arr.map(v=>Math.exp(v-m)), sum=ex.reduce((a,b)=>a+b,0); return ex.map(v=>v/sum); }

function highlightStage(name){
  const map={ tokens:el.tokens.closest('.panel'), embeddings:el.embeddings.closest('.panel'), attention:el.attentionMatrix.closest('.panel'), next:el.next.closest('.panel'), qkv:el.qVec.closest('section.panel') };
  Object.values(map).forEach(p=>p&&p.classList.remove('activeStage')); map[name]?.classList.add('activeStage');
  const labels={tokens:'токенизация',embeddings:'эмбеддинги',attention:'attention',next:'выбор следующего токена',qkv:'Q/K/V + softmax'};
  controls.stageLabel.textContent=`Стадия: ${labels[name]||'ожидание'}`;
}

function renderTokens(tokens){ el.tokens.innerHTML=tokens.map(t=>`<span class="chip">${t}</span>`).join(''); }
function renderEmbeddings(tokens){ el.embeddings.innerHTML=tokens.map(t=>{const v=Math.max(8,Math.round(pseudoValue(t,3)*100)); return `<div class="embRow"><div class="embLabel">${t}</div><div class="barWrap"><div class="bar" style="width:${v}%"></div></div></div>`;}).join(''); }
function attentionWeight(a,b,mode){ const base=0.15+((a+1)/(b+2)), noise=(Math.sin((a+1)*(b+2))+1)/10, m=mode==='advanced'?1.25:1; return Math.min(0.99, +(Math.max(0.01,(base*0.3+noise)*m).toFixed(2))); }
function renderAttention(tokens,mode){ const n=tokens.length,rows=[]; for(let i=0;i<n;i++){ const cells=[]; for(let j=0;j<n;j++){ const w=attentionWeight(i,j,mode), light=18+Math.round(w*38); cells.push(`<div class="cell" style="background:hsl(210 70% ${light}%)">${w}</div>`);} rows.push(`<div class="matrixRow">${cells.join('')}</div>`);} el.attentionMatrix.innerHTML=rows.join(''); }
function nextTokenCandidates(tokens,mode){ const bank=['и','это','потому','что','модель','предсказывает','контекст','далее']; return bank.map((w,idx)=>({w,p:+(0.06+((idx+1)/100)+(tokens.length%5)/50+(mode==='advanced'?0.01:0)).toFixed(2)})).sort((a,b)=>b.p-a.p).slice(0,5); }
function renderNext(tokens,mode){ const c=nextTokenCandidates(tokens,mode); el.next.textContent=`Выбранный токен (top-1): «${c[0].w}»`; el.candidates.innerHTML=c.map(x=>`<div class=\"cand\"><span>${x.w}</span><span>${x.p}</span></div>`).join(''); if (el.dataMode?.value === 'real') renderRealTopK(el.prompt.value.trim()); }
function updateTokenSelector(tokens){ const prev=el.qkvToken.value; el.qkvToken.innerHTML=tokens.map((t,i)=>`<option value="${i}">${t}</option>`).join(''); if(prev && +prev<tokens.length) el.qkvToken.value=prev; }
function renderHeadsCompare(tokens,qi){ if(!el.headsCompare) return; if(!tokens.length){el.headsCompare.innerHTML='';return;} const cards=[1,2,3].map(head=>{ const q=vecForToken(tokens[qi],'q',head), kAll=tokens.map(t=>vecForToken(t,'k',head)), probs=softmax(kAll.map(k=>dot(q,k))); const top=probs.map((p,i)=>({t:tokens[i],p})).sort((a,b)=>b.p-a.p).slice(0,3); const rows=top.map(x=>{const pct=Math.round(x.p*100); return `<div><div class="miniLbl"><span>${x.t}</span><span>${pct}%</span></div><div class="miniBar"><i style="width:${pct}%"></i></div></div>`;}).join(''); return `<div class="headCard"><h4>Head ${head}</h4>${rows}</div>`; }); el.headsCompare.innerHTML=cards.join(''); }
function renderQKV(tokens){ if(!tokens.length){ el.qVec.textContent=el.kVec.textContent=el.vVec.textContent=''; el.softmaxBars.innerHTML=''; return; } updateTokenSelector(tokens); const qi=+el.qkvToken.value||0, head=+(el.headSelect?.value||1), qTok=tokens[qi], q=vecForToken(qTok,'q',head), kAll=tokens.map(t=>vecForToken(t,'k',head)), vAll=tokens.map(t=>vecForToken(t,'v',head)); const probs=softmax(kAll.map(k=>dot(q,k))); el.qVec.textContent=`Head ${head}\n${qTok}\n[${q.join(', ')}]`; el.kVec.textContent=tokens.map((t,i)=>`${t}: [${kAll[i].join(', ')}]`).join('\n'); el.vVec.textContent=tokens.map((t,i)=>`${t}: [${vAll[i].join(', ')}]`).join('\n'); el.softmaxBars.innerHTML=probs.map((p,i)=>{const pct=Math.round(p*100); return `<div class="sbar"><div class="lbl">${tokens[i]} — ${pct}%</div><div class="wrap"><div class="fill" style="width:${pct}%"></div></div></div>`;}).join(''); renderHeadsCompare(tokens,qi); }


// -------- Real model (beta) --------
let realModel = null;
let realTokenizer = null;
let realLoaded = false;
let loadedModelId = null;

async function ensureRealModel() {
  const wanted = el.realModelId?.value || 'Xenova/tiny-random-gpt2';
  if (realLoaded && loadedModelId === wanted) return true;
  try {
    el.realStatus.textContent = `загрузка модели (${wanted})...`;
    realLoaded = false;
    const tr = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2');
    const { AutoTokenizer, AutoModelForCausalLM } = tr;
    realTokenizer = await AutoTokenizer.from_pretrained(wanted);
    realModel = await AutoModelForCausalLM.from_pretrained(wanted);
    loadedModelId = wanted;
    realLoaded = true;
    el.realStatus.textContent = `модель загружена: ${wanted}`;
    return true;
  } catch (e) {
    console.error(e);
    el.realStatus.textContent = `ошибка загрузки модели: ${e?.message || e}`;
    return false;
  }
}

function topKFromLogits(logits, k = 5) {
  const arr = Array.from(logits);
  const m = Math.max(...arr);
  const ex = arr.map(v => Math.exp(v - m));
  const sum = ex.reduce((a, b) => a + b, 0);
  const probs = ex.map(v => v / sum);
  const idx = probs.map((p, i) => ({ i, p })).sort((a, b) => b.p - a.p).slice(0, k);
  return idx;
}

async function renderRealTopK(promptText) {
  if (!el.realTopK) return;
  if (!realLoaded) {
    el.realTopK.innerHTML = '<div class="muted">Сначала нажми "Загрузить real model".</div>';
    return;
  }
  try {
    el.realStatus.textContent = 'считаю top-k...';
    const enc = await realTokenizer(promptText || 'Hello');
    const out = await realModel(enc);
    const logits = out.logits;
    const shape = logits.dims || logits.shape;
    const vocab = shape[2];
    const seq = shape[1];
    const data = await logits.data;
    const offset = (seq - 1) * vocab;
    const last = data.slice(offset, offset + vocab);
    const top = topKFromLogits(last, 5);
    const ids = top.map(x => x.i);
    const toks = await realTokenizer.batch_decode(ids, { skip_special_tokens: false });
    el.realTopK.innerHTML = top.map((x, n) => `<div class="cand"><span>${n + 1}. ${String(toks[n]).replace(/</g,'&lt;')}</span><span>${(x.p * 100).toFixed(2)}%</span></div>`).join('');
    el.realStatus.textContent = 'готово';
  } catch (e) {
    console.error(e);
    el.realStatus.textContent = 'ошибка inference';
  }
}

function clearOutputs(){ el.tokens.innerHTML=el.embeddings.innerHTML=el.attentionMatrix.innerHTML=el.candidates.innerHTML=''; el.next.textContent=''; }
function renderByStage(stage,tokens,mode){ if(stage==='tokens')renderTokens(tokens); if(stage==='embeddings')renderEmbeddings(tokens); if(stage==='attention')renderAttention(tokens,mode); if(stage==='next')renderNext(tokens,mode); if(stage==='qkv')renderQKV(tokens); highlightStage(stage); }
function runStep(){ const t=tokenize(el.prompt.value.trim()), m=el.mode.value; if(stageIndex===0) clearOutputs(); const st=stages[stageIndex]; renderByStage(st,t,m); stageIndex=(stageIndex+1)%stages.length; update3D(t); }
function play(){ if(timer) return; runStep(); timer=setInterval(runStep,900); }
function pause(){ if(timer) clearInterval(timer); timer=null; }
function renderAll(){ pause(); stageIndex=0; const t=tokenize(el.prompt.value.trim()), m=el.mode.value; renderTokens(t); renderEmbeddings(t); renderAttention(t,m); renderNext(t,m); renderQKV(t); highlightStage('qkv'); update3D(t); }

el.run.addEventListener('click', renderAll);
el.mode.addEventListener('change', renderAll);
el.qkvToken?.addEventListener('change', ()=>{ const t=tokenize(el.prompt.value.trim()); renderQKV(t); update3D(t); });
el.headSelect?.addEventListener('change', ()=>{ const t=tokenize(el.prompt.value.trim()); renderQKV(t); update3D(t); });
el.dataMode?.addEventListener('change', ()=>{ if (el.dataMode.value === 'real') renderRealTopK(el.prompt.value.trim()); else if (el.realTopK) el.realTopK.innerHTML=''; });
el.loadReal?.addEventListener('click', async ()=>{ await ensureRealModel(); if (el.dataMode?.value==='real') await renderRealTopK(el.prompt.value.trim()); });
controls.play.addEventListener('click', play);
controls.pause.addEventListener('click', pause);
controls.step.addEventListener('click', ()=>{ pause(); runStep(); });

// -------- 3D Scene --------
let scene, camera, renderer, group, raf;
function init3D(){
  if(!el.llm3d) return;
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x070b18);
  camera = new THREE.PerspectiveCamera(45, el.llm3d.clientWidth / el.llm3d.clientHeight, 0.1, 100);
  camera.position.set(0,0,8);
  renderer = new THREE.WebGLRenderer({antialias:true});
  renderer.setSize(el.llm3d.clientWidth, el.llm3d.clientHeight);
  el.llm3d.innerHTML=''; el.llm3d.appendChild(renderer.domElement);

  const light = new THREE.PointLight(0x88aaff, 1.2); light.position.set(3,4,6); scene.add(light);
  scene.add(new THREE.AmbientLight(0x5577aa,0.7));
  group = new THREE.Group(); scene.add(group);

  const animate=()=>{ raf=requestAnimationFrame(animate); group.rotation.y += 0.003; renderer.render(scene,camera); };
  animate();
  window.addEventListener('resize', ()=>{
    if(!renderer||!camera||!el.llm3d) return;
    const w=el.llm3d.clientWidth,h=el.llm3d.clientHeight;
    renderer.setSize(w,h); camera.aspect=w/h; camera.updateProjectionMatrix();
  });
}

function clearGroup(){ if(!group) return; while(group.children.length){ const o=group.children.pop(); o.geometry?.dispose?.(); o.material?.dispose?.(); } }
function sphere(x,y,z,color=0x66ccff,r=0.08){ const m=new THREE.Mesh(new THREE.SphereGeometry(r,16,16), new THREE.MeshStandardMaterial({color,emissive:0x112244})); m.position.set(x,y,z); return m; }
function line(a,b,color=0x4466aa,op=0.5){ const g=new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...a),new THREE.Vector3(...b)]); const l=new THREE.Line(g,new THREE.LineBasicMaterial({color,transparent:true,opacity:op})); return l; }

function update3D(tokens){
  if(!group) return;
  clearGroup();
  const n=Math.max(tokens.length,1);
  const ys=[...Array(n)].map((_,i)=> (i-(n-1)/2)*0.35);
  const leftX=-2.2, midX=0, rightX=2.2;
  ys.forEach((y,i)=>{ group.add(sphere(leftX,y,0,0x6ad8ff,0.07)); group.add(sphere(midX,y,0,0x8a8fff,0.07)); group.add(line([leftX,y,0],[midX,y,0],0x3a5f99,0.45)); });
  const nextY=0; group.add(sphere(rightX,nextY,0,0xffc36a,0.1));
  // attention rays from a focus token
  const qi=+((el.qkvToken && el.qkvToken.value) || 0);
  const qy=ys[Math.min(qi,ys.length-1)] ?? 0;
  ys.forEach((y,j)=>{ const w=0.2+0.8*((Math.sin((qi+1)*(j+2))+1)/2); group.add(line([midX,qy,0],[midX,y,0],0x66e0ff,0.2+0.5*w)); group.add(line([midX,y,0],[rightX,nextY,0],0x9aa7ff,0.12+0.28*w)); });
}

init3D();
renderAll();
