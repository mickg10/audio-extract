#!/usr/bin/env python3
"""Blind A/B listening page for paired instrumentals (ab_pairs/).

Scans ``--dir`` (default ../ab_pairs) for ``<track>__A_<label>.m4a`` /
``<track>__B_<label>.m4a`` pairs and serves a single-page player that:

* switches between A/B **at the same playhead position** (one <audio>, src swap);
* supports a **blind** mode (labels hidden, A/B randomly flipped per track,
  revealed only after you vote);
* records votes to ``ab_verdicts.json`` next to the pairs.

Stdlib only, same design rules as server.py (read-only over audio, defensive
re-scan per request, HTTP Range so seeking works).

Run:  cd ~/audio-extract && uv run python web/ab.py   [--port 8766]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import os
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIR_RE = re.compile(r"^(?P<track>.+?)__(?P<side>[A-Z][0-9]?)_(?P<label>.+?)\.(m4a|mp3|flac|wav)$")


def scan_clips(d: str) -> list[str]:
    """Blind-test mini clips: <anything>__clip<N>.<ext>, shuffled order baked into names."""
    try:
        names = sorted(os.listdir(d))
    except OSError:
        return []
    return [n for n in names if re.match(r"^.+__clip\d+\.(m4a|mp3|flac|wav)$", n)]


def scan_pairs(d: str) -> list[dict]:
    by_track: dict[str, dict] = {}
    try:
        names = sorted(os.listdir(d))
    except OSError:
        return []
    for n in names:
        m = PAIR_RE.match(n)
        if not m:
            continue
        t = by_track.setdefault(m.group("track"), {"track": m.group("track")})
        t[m.group("side")] = {"file": n, "label": m.group("label")}
    return [t for t in by_track.values() if "A" in t and "B" in t]


PAGE = """<!doctype html><meta charset=utf-8>
<title>A/B — instrumental pairs</title>
<style>
 body{font:15px/1.5 system-ui;margin:0;background:#10151c;color:#e8edf3;padding:24px}
 h1{font-size:20px;margin:0 0 4px} .sub{color:#8b98a8;margin:0 0 18px;font-size:13px}
 .pair{background:#182029;border:1px solid #26313d;border-radius:12px;padding:16px 18px;margin:0 0 14px;max-width:860px}
 .t{font-weight:600;margin-bottom:10px}
 button{font:inherit;border:1px solid #33414f;background:#202b36;color:#e8edf3;border-radius:8px;padding:8px 16px;margin-right:8px;cursor:pointer}
 button.on{background:#2e6fb7;border-color:#2e6fb7}
 button[data-k=Y].on{background:#22c3e0;border-color:#22c3e0;color:#06232a}
 button.vote{border-color:#3a5;} button.voted{background:#3a5;border-color:#3a5}
 audio{width:100%;margin:10px 0 8px}
 .reveal{color:#f0b429;font-size:13px;margin-left:8px}
 .hint{color:#8b98a8;font-size:12px;margin-top:12px}
 .empty{color:#8b98a8;padding:40px;text-align:center}
</style>
<div style="position:fixed;top:8px;right:12px;z-index:8;font-size:12px;background:#182029;border:1px solid #26313d;border-radius:16px;padding:6px 12px;color:#8b98a8">👤 <b style="color:#e8edf3" id=whon></b>
 <a href=# id=whochange style="color:#2e6fb7;text-decoration:none;margin-left:8px">change</a>
 <a href=# id=logout style="color:#c33;text-decoration:none;margin-left:6px">logout</a></div>
<h1>Blind A/B — champion vs fine-tuned ensemble</h1>
<p class=sub>One player per track; X/Y switch at the same position. Labels are hidden and randomly flipped until you vote. Space = play/pause on the last-touched player.</p>
<div id=blind></div>
<div id=list></div>
<p class=hint>Votes are saved to ab_verdicts.json. Refresh to pick up newly delivered pairs.</p>
<script>
let state={};
document.getElementById('whon').textContent=localStorage.rater||'mickg (default)';
document.getElementById('whochange').onclick=e=>{e.preventDefault();const v=prompt('Who is voting?');if(v&&v.trim()){localStorage.rater=v.trim();location.reload();}};
document.getElementById('logout').onclick=e=>{e.preventDefault();localStorage.removeItem('rater');location.reload();};
async function loadClips(){
  const r=await fetch('/api/clips'); const clips=await r.json();
  const votes=await (await fetch('/api/votes?user='+encodeURIComponent(localStorage.rater||'mickg'))).json().catch(()=>({}));
  const el=document.getElementById('blind');
  if(!clips.length){el.innerHTML='';return}
  el.innerHTML='<div class=pair><div class=t>🎯 BLIND WINDOW TEST — judge each clip cold: is there VOICE? can you hear WORDS?</div><div id=clist></div></div>';
  const cl=el.querySelector('#clist');
  for(const f of clips){
    const short=f.replace(/\.(m4a|mp3|flac|wav)$/,'').split('__').pop();
    const row=document.createElement('div');
    row.innerHTML=`<div style="margin:10px 0"><b>${short}</b>
      <audio controls preload=none src="/audio/${encodeURIComponent(f)}" style="width:100%"></audio>
      <button class=vote data-v=none>no voice</button>
      <button class=vote data-v=voice>voice (no words)</button>
      <button class=vote data-v=words>WORDS</button>
      &nbsp;|&nbsp; quality:
      <button class="vote q" data-v=quality_ok>acceptable</button>
      <button class="vote q" data-v=quality_borderline>borderline</button>
      <button class="vote q" data-v=quality_unacceptable style="border-color:#c33">UNACCEPTABLE (dropouts/artifacts)</button>
      <button class=clearbtn style="color:#8b98a8">✕ clear</button>
      <span class=reveal></span></div>`;
    row.querySelectorAll('button.vote').forEach(b=>b.onclick=async()=>{
      await fetch('/api/vote',{method:'POST',body:JSON.stringify({track:f,vote:b.dataset.v,user:localStorage.rater||'mickg'})});
      const grp = b.classList.contains('q') ? 'button.vote.q' : 'button.vote:not(.q)';
      row.querySelectorAll(grp).forEach(x=>x.classList.remove('voted'));b.classList.add('voted');
      row.querySelector('.reveal').textContent='recorded: '+b.dataset.v;
    });
    row.querySelector('.clearbtn').onclick=async()=>{
      await fetch('/api/vote',{method:'POST',body:JSON.stringify({track:f,vote:'clear',user:localStorage.rater||'mickg'})});
      row.querySelectorAll('button.vote').forEach(x=>x.classList.remove('voted'));
      row.querySelector('.reveal').textContent='cleared';
    };
    const prev=votes[f]||{};
    const marks=[];
    row.querySelectorAll('button.vote').forEach(x=>{
      if(x.dataset.v===prev.voice||x.dataset.v===prev.quality){x.classList.add('voted');marks.push(x.dataset.v);}
    });
    if(marks.length)row.querySelector('.reveal').textContent='saved: '+marks.join(' + ');
    cl.appendChild(row);
  }
}
loadClips();
async function load(){
  const r=await fetch('/api/pairs'); const pairs=await r.json();
  const lvotes=await (await fetch('/api/votes?user='+encodeURIComponent(localStorage.rater||'mickg'))).json().catch(()=>({}));
  const el=document.getElementById('list');
  if(!pairs.length){el.innerHTML='<div class=empty>No pairs yet — they appear here as they are delivered.</div>';return}
  el.innerHTML='';
  const ladder=pairs.filter(p=>/^L\d/.test(p.track)), legacy=pairs.filter(p=>!/^L\d/.test(p.track));
  if(ladder.length){
    const h=document.createElement('div');h.className='pair';
    h.innerHTML='<div class=t>🪜 LADDER — each pair: one side is the ORIGINAL, the other a candidate (loudness-matched, blind). Switch X/Y at the same spot. Vote relative fullness; flag voice or damage on a side if you hear it.</div>';
    el.appendChild(h);
  }
  for(const p of ladder){
    const flip=Math.random()<0.5;
    const d=document.createElement('div');d.className='pair';
    const side=k=>((k==='X')!==flip)?'B':'A';
    d.innerHTML=`<div class=t>${p.track}</div>
      <audio class=pA controls loop preload=metadata></audio>
      <audio class=pB controls loop preload=metadata style="display:none"></audio><div>
      <button data-k=X class=on>X</button><button data-k=Y>Y</button>
      <button class=vote data-lv=fuller>X fuller</button>
      <button class=vote data-lv=equal>equal</button>
      <button class=vote data-lv=fullerY>Y fuller</button>
      &nbsp;|&nbsp;
      <button class="vote sx" data-lv=trace_X>trace X</button>
      <button class="vote sx" data-lv=voice_X>voice X</button>
      <button class="vote sx" data-lv=words_X>WORDS X</button>
      <button class="vote fl" data-lv=dmg_X style="border-color:#c33">dmg X</button>
      <button class="vote sy" data-lv=trace_Y>trace Y</button>
      <button class="vote sy" data-lv=voice_Y>voice Y</button>
      <button class="vote sy" data-lv=words_Y>WORDS Y</button>
      <button class="vote fl" data-lv=dmg_Y style="border-color:#c33">dmg Y</button>
      <button class=clearbtn style="color:#8b98a8">✕ clear</button>
      <span class=reveal></span></div>`;
    const aA=d.querySelector('.pA'), aB=d.querySelector('.pB');
    aA.src='/audio/'+encodeURIComponent(p.A.file);
    aB.src='/audio/'+encodeURIComponent(p.B.file);
    // X shows first; both play in lockstep, inactive one muted+hidden -> instant switch
    let act = (side('X')==='A') ? aA : aB;
    const oth = () => act===aA ? aB : aA;
    function show(){
      const o=oth();
      act.muted=false; act.style.display='';
      o.muted=true; o.style.display='none';
    }
    act.muted=false; oth().muted=true;
    if(act===aB){ aA.style.display='none'; aB.style.display=''; }
    [[aA,aB],[aB,aA]].forEach(([m,s])=>{
      m.addEventListener('play',  ()=>{ if(m===act){ if(Math.abs(s.currentTime-m.currentTime)>0.08)s.currentTime=m.currentTime; s.play().catch(()=>{});} });
      m.addEventListener('pause', ()=>{ if(m===act) s.pause(); });
      m.addEventListener('seeked',()=>{ if(m===act && Math.abs(s.currentTime-m.currentTime)>0.08) s.currentTime=m.currentTime; });
    });
    setInterval(()=>{ const o=oth(); if(!act.paused && Math.abs(o.currentTime-act.currentTime)>0.08) o.currentTime=act.currentTime; }, 1200);
    d.querySelectorAll('button[data-k]').forEach(b=>b.onclick=()=>{
      const tgt = (side(b.dataset.k)==='A') ? aA : aB;
      if(tgt!==act){
        const playing=!act.paused;
        if(Math.abs(tgt.currentTime-act.currentTime)>0.08) tgt.currentTime=act.currentTime;
        act=tgt; show();
        if(playing) act.play().catch(()=>{}); else act.pause();
      }
      d.querySelectorAll('button[data-k]').forEach(x=>x.classList.toggle('on',x.dataset.k===b.dataset.k));
    });
    d.querySelectorAll('button.vote').forEach(b=>b.onclick=async()=>{
      const lv=b.dataset.lv;
      const vote = lv==='equal' ? 'equal'
        : lv==='fuller'  ? 'fuller_'+side('X')
        : lv==='fullerY' ? 'fuller_'+side('Y')
        : lv.replace('_X','_'+side('X')).replace('_Y','_'+side('Y'));
      await fetch('/api/vote',{method:'POST',body:JSON.stringify({track:p.track,vote,user:localStorage.rater||'mickg'})});
      if(b.classList.contains('sx'))d.querySelectorAll('button.vote.sx').forEach(x=>x.classList.remove('voted'));
      else if(b.classList.contains('sy'))d.querySelectorAll('button.vote.sy').forEach(x=>x.classList.remove('voted'));
      else if(!b.classList.contains('fl'))d.querySelectorAll('button.vote:not(.fl):not(.sx):not(.sy)').forEach(x=>x.classList.remove('voted'));
      b.classList.add('voted');
      d.querySelector('.reveal').textContent='recorded';
    });
    d.querySelector('.clearbtn').onclick=async()=>{
      await fetch('/api/vote',{method:'POST',body:JSON.stringify({track:p.track,vote:'clear',user:localStorage.rater||'mickg'})});
      d.querySelectorAll('button.vote').forEach(x=>x.classList.remove('voted'));
      d.querySelector('.reveal').textContent='cleared';
    };
    const prev=lvotes[p.track]||{};
    if(prev.full){
      const lv=prev.full==='equal'?'equal':(side('X')===prev.full.slice(-1)?'fuller':'fullerY');
      const b=d.querySelector(`button.vote[data-lv=${lv}]`); if(b)b.classList.add('voted');
    }
    (prev.flags||[]).forEach(f=>{
      const xy=(side('X')===f.slice(-1))?'X':'Y';
      const b=d.querySelector(`button.vote[data-lv=dmg_${xy}]`);
      if(b)b.classList.add('voted');
    });
    for(const sd of ['A','B']){
      const sev=prev['v'+sd]; if(!sev)continue;
      const xy=(side('X')===sd)?'X':'Y';
      const b=d.querySelector(`button.vote[data-lv=${sev}_${xy}]`);
      if(b)b.classList.add('voted');
    }
    if(prev.full)d.querySelector('.reveal').textContent='saved';
    el.appendChild(d);
  }
  for(const p of legacy){
    const flip=Math.random()<0.5;                       // blind: X/Y randomly maps to A/B
    const s=state[p.track]={flip,cur:'X',revealed:false,el:null};
    const d=document.createElement('div');d.className='pair';
    const extras=Object.keys(p).filter(k=>!['track','A','B'].includes(k)).sort();
    const lab=k=>k==='O'?'Orig':k==='C'?'Cascade':k==='M'?'Max':(p[k].label||k);
    d.innerHTML=`<div class=t>${p.track}</div>
      <audio controls preload=none></audio><div>
      <button data-k=X class=on>X</button><button data-k=Y>Y</button>
      ${extras.map(k=>`<button data-k=${k}>${lab(k)}</button>`).join('')}
      <button class=vote data-v=X>X is better</button>
      <button class=vote data-v=Y>Y is better</button>
      <button class=vote data-v=tie>tie</button>
      <span class=reveal></span></div>`;
    const audio=d.querySelector('audio'); s.el=d;
    const src=k=> (k==='X'||k==='Y') ? '/audio/'+encodeURIComponent(((k==='X')!==flip)?p.B.file:p.A.file)
                                     : '/audio/'+encodeURIComponent(p[k].file);
    audio.src=src('X');
    d.querySelectorAll('button[data-k]').forEach(b=>b.onclick=()=>{
      const k=b.dataset.k; if(k===s.cur)return;
      const t=audio.currentTime, playing=!audio.paused;
      s.cur=k; audio.src=src(k); audio.currentTime=t; if(playing)audio.play();
      d.querySelectorAll('button[data-k]').forEach(x=>x.classList.toggle('on',x.dataset.k===k));
    });
    d.querySelectorAll('button.vote').forEach(b=>b.onclick=async()=>{
      const v=b.dataset.v;
      const actual=v==='tie'?'tie':(((v==='X')!==flip)?'B_melband_ft':'A_champion');
      await fetch('/api/vote',{method:'POST',body:JSON.stringify({track:p.track,vote:actual,user:localStorage.rater||'mickg'})});
      d.querySelectorAll('button.vote').forEach(x=>x.classList.remove('voted'));b.classList.add('voted');
      const map=k=>((k==='X')!==flip)?p.B.label:p.A.label;
      d.querySelector('.reveal').textContent=`revealed: X=${map('X')} · Y=${map('Y')} · you chose ${actual}`;
    });
    el.appendChild(d);
  }
}
load();
document.addEventListener('play',e=>{
  const box=e.target.closest('.pair')||e.target.parentElement;
  document.querySelectorAll('audio').forEach(a=>{
    if(a!==e.target && !(box && box.contains(a))) a.pause();
  });
},true);
</script>"""


MOBILE_PAGE = """<!doctype html><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Listening Lab</title>
<style>
 :root{--bg:#10151c;--card:#182029;--line:#26313d;--fg:#e8edf3;--dim:#8b98a8;--acc:#2e6fb7;--bx:#2e6fb7;--by:#22c3e0;--ok:#3a5;--warn:#c33}
 *{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
 body{font:16px/1.45 system-ui;margin:0;background:var(--bg);color:var(--fg);padding:12px 12px 80px}
 h1{font-size:18px;margin:4px 0 2px}.sub{color:var(--dim);font-size:13px;margin:0 0 12px}
 .card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px;margin:0 0 12px}
 .t{font-weight:600;margin-bottom:8px;font-size:15px;display:flex;justify-content:space-between;align-items:center}
 .done{color:var(--ok);font-size:13px}
 button{font:inherit;border:1px solid #33414f;background:#202b36;color:var(--fg);border-radius:10px;padding:12px 10px;cursor:pointer}
 button:disabled{opacity:.35}
 .xy{display:grid;grid-template-columns:64px 1fr 1fr;gap:8px;margin:6px 0}
 .xy button{font-size:20px;font-weight:700;padding:16px 0}
 .xy button[data-k=X].on{background:var(--bx);border-color:var(--bx)}
 .xy button[data-k=Y].on{background:var(--by);border-color:var(--by);color:#06232a}
 .pbar{height:8px;background:#243040;border-radius:4px;margin:8px 0 4px;overflow:hidden}
 .pfill{height:100%;width:0;background:var(--bx);border-radius:4px}
 .votes{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin:6px 0}
 .flags{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:6px}
 .flags button{font-size:12px;padding:9px 4px}
 .voted{background:var(--ok)!important;border-color:var(--ok)!important}
 .fv{border-color:var(--ok)} .fw{border-color:var(--warn)}
 #gate{position:fixed;inset:0;background:var(--bg);z-index:9;display:flex;flex-direction:column;justify-content:center;padding:32px;gap:12px}
 #gate input{font:20px system-ui;padding:14px;border-radius:10px;border:1px solid var(--line);background:var(--card);color:var(--fg)}
 #gate button{font-size:18px;padding:14px;background:var(--acc);border-color:var(--acc)}
 #bar{position:fixed;left:0;right:0;bottom:0;background:#0c1117;border-top:1px solid var(--line);padding:10px 16px;font-size:13px;color:var(--dim)}
 #prog{height:4px;background:#243040;border-radius:2px;margin-top:6px}#progi{height:100%;width:0;background:var(--acc);border-radius:2px}
</style>
<div id=who style="position:fixed;top:8px;right:10px;z-index:8;font-size:12px;background:var(--card);border:1px solid var(--line);border-radius:16px;padding:6px 12px;display:none;color:var(--dim)">👤 <b id=whon style="color:var(--fg)"></b> <a href=# id=logout style="color:#c33;text-decoration:none;margin-left:8px">logout</a></div>
<div id=gate><div style="font-size:22px;font-weight:700">Who's listening?</div>
 <div style="color:var(--dim)">Your name is attached to every vote.</div>
 <input id=nm placeholder="name" autocapitalize=none autocomplete=name>
 <button onclick="setName()">Start</button></div>
<h1>🪜 Listening Lab</h1>
<p class=sub>Each pair: press play, flip X/Y (instant, looped). Which is fuller? Flag voice or damage if you hear it.</p>
<div id=cards></div>
<div id=bar><span id=st>loading…</span><div id=prog><div id=progi></div></div></div>
<script>
const $=q=>document.querySelector(q);
let rater=localStorage.rater||'';
function showWho(){const w=document.getElementById('who');if(rater&&w){w.style.display='';document.getElementById('whon').textContent=rater;}}
document.addEventListener('click',e=>{if(e.target&&e.target.id==='logout'){e.preventDefault();localStorage.removeItem('rater');location.reload();}});
if(rater){$('#gate').style.display='none';showWho();}
function setName(){const v=$('#nm').value.trim();if(!v)return;rater=v;localStorage.rater=v;$('#gate').style.display='none';showWho();main();}
let ctx=null, playing=null;           // playing = {stop(),update()}
(function tick(){ if(playing&&playing.update)playing.update(); requestAnimationFrame(tick); })();
const blobs={}, decoded={}, order=[];
async function main(){
  const pairs=(await (await fetch('api/pairs')).json()).filter(p=>/^L\\d/.test(p.track));
  const votes=await (await fetch('api/votes?user='+encodeURIComponent(rater||'anon'))).json().catch(()=>({}));
  const cards=$('#cards');
  let loadedN=0, total=pairs.length*2;
  for(const p of pairs){
    const flip=Math.random()<0.5;
    const side=k=>((k==='X')!==flip)?'B':'A';
    const c=document.createElement('div');c.className='card';
    c.innerHTML=`<div class=t><span>${p.track}</span><span class=done></span></div>
      <div class=xy><button class=pp disabled>▶</button><button data-k=X class=on disabled>X</button><button data-k=Y disabled>Y</button></div>
      <div class=pbar><div class=pfill></div></div>
      <div class=votes><button class="v fv" data-lv=fuller disabled>X fuller</button>
        <button class="v fv" data-lv=equal disabled>equal</button>
        <button class="v fv" data-lv=fullerY disabled>Y fuller</button></div>
      <div class=flags><button class="v sx" data-lv=trace_X disabled>trace X</button>
        <button class="v sx" data-lv=voice_X disabled>voice X</button>
        <button class="v sx" data-lv=words_X disabled>WORDS X</button>
        <button class="v fl fw" data-lv=dmg_X disabled>dmg X</button>
        <button class="v sy" data-lv=trace_Y disabled>trace Y</button>
        <button class="v sy" data-lv=voice_Y disabled>voice Y</button>
        <button class="v sy" data-lv=words_Y disabled>WORDS Y</button>
        <button class="v fl fw" data-lv=dmg_Y disabled>dmg Y</button></div>
      <div style="text-align:right;margin-top:6px"><button class=clr disabled style="font-size:12px;padding:7px 14px;color:var(--dim)">✕ clear my votes</button></div>`;
    cards.appendChild(c);
    const files={A:p.A.file,B:p.B.file};
    (async()=>{ // preload both sides as bytes
      for(const s of ['A','B']){
        const r=await fetch('audio/'+encodeURIComponent(files[s]));
        blobs[p.track+s]=await r.arrayBuffer();
        loadedN++; $('#st').textContent=`preloaded ${loadedN}/${total} clips`;
        $('#progi').style.width=(100*loadedN/total)+'%';
      }
      c.querySelectorAll('button').forEach(b=>b.disabled=false);
      if(loadedN===total)$('#st').textContent=`all ${total} clips preloaded — ready (rater: ${rater||'?'})`;
    })();
    let g={}, srcs=null, cur='X';
    const pp=c.querySelector('.pp');
    async function getBuf(s){
      const k=p.track+s;
      if(!decoded[k]){decoded[k]=await ctx.decodeAudioData(blobs[k].slice(0));order.push(k);
        while(order.length>10){delete decoded[order.shift()];}}
      return decoded[k];
    }
    async function start(){
      if(!ctx)ctx=new (window.AudioContext||window.webkitAudioContext)();
      await ctx.resume();
      if(playing)playing.stop();
      const bA=await getBuf('A'), bB=await getBuf('B');
      const loopEnd=Math.min(bA.duration,bB.duration);
      srcs={};
      for(const s of ['A','B']){
        const src=ctx.createBufferSource(), gn=ctx.createGain();
        src.buffer=(s==='A')?bA:bB; src.loop=true; src.loopEnd=loopEnd;
        gn.gain.value=(side(cur)===s)?1:0;
        src.connect(gn).connect(ctx.destination);
        srcs[s]=src; g[s]=gn;
      }
      const t=ctx.currentTime+0.05;
      srcs.A.start(t); srcs.B.start(t);
      pp.textContent='■';
      const fill=c.querySelector('.pfill');
      playing={
        stop(){try{srcs.A.stop();srcs.B.stop();}catch(e){} srcs=null; pp.textContent='▶'; fill.style.width='0';},
        update(){
          if(!srcs)return;
          const pos=Math.max(0,(ctx.currentTime-t))%loopEnd/loopEnd;
          fill.style.width=(pos*100)+'%';
          fill.style.background=(cur==='X')?'var(--bx)':'var(--by)';
        }
      };
    }
    pp.onclick=()=>{ if(srcs){playing.stop();playing=null;} else start(); };
    c.querySelectorAll('button[data-k]').forEach(b=>b.onclick=()=>{
      cur=b.dataset.k;
      c.querySelectorAll('button[data-k]').forEach(x=>x.classList.toggle('on',x===b));
      if(srcs){const t=ctx.currentTime;
        g.A.gain.setTargetAtTime(side(cur)==='A'?1:0,t,0.006);
        g.B.gain.setTargetAtTime(side(cur)==='B'?1:0,t,0.006);}
    });
    c.querySelectorAll('button.v').forEach(b=>b.onclick=async()=>{
      const lv=b.dataset.lv;
      const vote = lv==='equal' ? 'equal'
        : lv==='fuller' ? 'fuller_'+side('X')
        : lv==='fullerY' ? 'fuller_'+side('Y')
        : lv.replace('_X','_'+side('X')).replace('_Y','_'+side('Y'));
      await fetch('api/vote',{method:'POST',body:JSON.stringify({track:p.track,vote,user:rater||'anon'})});
      if(b.classList.contains('sx'))c.querySelectorAll('button.v.sx').forEach(x=>x.classList.remove('voted'));
      else if(b.classList.contains('sy'))c.querySelectorAll('button.v.sy').forEach(x=>x.classList.remove('voted'));
      else if(!b.classList.contains('fl'))c.querySelectorAll('button.v:not(.fl):not(.sx):not(.sy)').forEach(x=>x.classList.remove('voted'));
      b.classList.add('voted');
      c.querySelector('.done').textContent='✓ '+(rater||'');
    });
    c.querySelector('.clr').onclick=async()=>{
      await fetch('api/vote',{method:'POST',body:JSON.stringify({track:p.track,vote:'clear',user:rater||'anon'})});
      c.querySelectorAll('button.v').forEach(x=>x.classList.remove('voted'));
      c.querySelector('.done').textContent='';
    };
    const prev=votes[p.track]||{};
    if(prev.full){
      const lv=prev.full==='equal'?'equal':(side('X')===prev.full.slice(-1)?'fuller':'fullerY');
      const b=c.querySelector(`button.v[data-lv=${lv}]`); if(b)b.classList.add('voted');
      c.querySelector('.done').textContent='✓ '+(rater||'');
    }
    (prev.flags||[]).forEach(f=>{
      const xy=(side('X')===f.slice(-1))?'X':'Y';
      const b=c.querySelector(`button.v[data-lv=dmg_${xy}]`);
      if(b)b.classList.add('voted');
    });
    for(const sd of ['A','B']){
      const sev=prev['v'+sd]; if(!sev)continue;
      const xy=(side('X')===sd)?'X':'Y';
      const b=c.querySelector(`button.v[data-lv=${sev}_${xy}]`);
      if(b)b.classList.add('voted');
    }
  }
}
if(rater)main();
</script>"""



TRACKS_PAGE = """<!doctype html><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Track picks</title>
<style>
 :root{--bg:#10151c;--card:#182029;--line:#26313d;--fg:#e8edf3;--dim:#8b98a8;--acc:#2e6fb7;--ok:#3a5;--warn:#c33}
 *{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
 body{font:16px/1.45 system-ui;margin:0;background:var(--bg);color:var(--fg);padding:12px}
 h1{font-size:18px;margin:4px 0 2px}.sub{color:var(--dim);font-size:13px;margin:0 0 12px}
 .card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px;margin:0 0 12px}
 .t{font-weight:600;margin-bottom:8px;font-size:15px;word-break:break-all;display:flex;justify-content:space-between;gap:8px}
 .done{color:var(--ok);font-size:13px;white-space:nowrap}
 audio{width:100%;margin:4px 0 10px}
 .votes5{display:grid;grid-template-columns:repeat(5,1fr);gap:6px;margin-bottom:8px}
 .tags{display:flex;flex-wrap:wrap;gap:6px}
 .tags button{font-size:13px;padding:9px 10px;border-radius:16px}
 button{font:inherit;font-size:15px;border:1px solid #33414f;background:#202b36;color:var(--fg);border-radius:10px;padding:12px 4px;cursor:pointer}
 .voted{background:var(--ok)!important;border-color:var(--ok)!important;color:#fff!important}
 .voted.bad{background:var(--warn)!important;border-color:var(--warn)!important}
 #gate{position:fixed;inset:0;background:var(--bg);z-index:9;display:flex;flex-direction:column;justify-content:center;padding:32px;gap:12px}
 #gate input{font:20px system-ui;padding:14px;border-radius:10px;border:1px solid var(--line);background:var(--card);color:var(--fg)}
 #gate button{font-size:18px;background:var(--acc);border-color:var(--acc)}
</style>
<div id=who style="position:fixed;top:8px;right:10px;z-index:8;font-size:12px;background:var(--card);border:1px solid var(--line);border-radius:16px;padding:6px 12px;display:none;color:var(--dim)">👤 <b id=whon style="color:var(--fg)"></b> <a href=# id=logout style="color:#c33;text-decoration:none;margin-left:8px">logout</a></div>
<div id=gate><div style="font-size:22px;font-weight:700">Who's listening?</div>
 <input id=nm placeholder="name" autocapitalize=none>
 <button onclick="setName()">Start</button></div>
<h1>🎵 Track picks & expert critique</h1>
<p class=sub>Top section: the ship candidates that most need your ears. Below: the current delivered versions. Rate, tag, and write anything you hear in the comment box.</p>
<div id=cards></div>
<script>
const $=q=>document.querySelector(q);
let rater=localStorage.rater||'';
function showWho(){const w=document.getElementById('who');if(rater&&w){w.style.display='';document.getElementById('whon').textContent=rater;}}
document.addEventListener('click',e=>{if(e.target&&e.target.id==='logout'){e.preventDefault();localStorage.removeItem('rater');location.reload();}});
if(rater){$('#gate').style.display='none';showWho();}
function setName(){const v=$('#nm').value.trim();if(!v)return;rater=v;localStorage.rater=v;$('#gate').style.display='none';showWho();main();}
async function main(){
  const tracks=await (await fetch('api/tracks')).json();
  const votes=await (await fetch('api/votes?user='+encodeURIComponent(rater||'anon'))).json().catch(()=>({}));
  const el=$('#cards');
  let curatedHdr=false, baseHdr=false;
  for(const it of tracks){
    const f=it.f, nice=it.label;
    if(it.src==='audio'&&!curatedHdr){curatedHdr=true;const h=document.createElement('div');h.innerHTML='<h2 style="font-size:15px;color:var(--acc);margin:6px 0">�standby FOR YOUR CRITIQUE — ship candidates</h2>'.replace('�standby','🎯');el.appendChild(h);}
    if(it.src==='daudio'&&!baseHdr){baseHdr=true;const h=document.createElement('div');h.innerHTML='<h2 style="font-size:15px;color:var(--dim);margin:14px 0 6px">📦 current deliveries (baseline)</h2>';el.appendChild(h);}
    const c=document.createElement('div');c.className='card';
    c.innerHTML=`<div class=t><span>${nice}</span><span class=done></span></div>
      ${it.note?`<div style="color:var(--dim);font-size:12px;margin-bottom:4px">${it.note}</div>`:''}
      <audio controls preload=none src="${it.src}/${encodeURIComponent(f)}"></audio>
      <div class=votes5>
        <button data-v=pick5_1 class=b5>1<br>unusable</button>
        <button data-v=pick5_2 class=b5>2<br>poor</button>
        <button data-v=pick5_3 class=b5>3<br>so-so</button>
        <button data-v=pick5_4 class=b5>4<br>good</button>
        <button data-v=pick5_5 class=b5>5<br>clean instr.</button></div>
      <div class=tags>
        <button data-t=voice class=bad>🎤 voice audible</button>
        <button data-t=dirty class=bad>🧹 dirty / artifacts</button>
        <button data-t=dull_treble class=bad>😶 treble dull where voice was</button>
        <button data-t=tinny_bass class=bad>🔉 bass tinny</button>
        <button data-t=wrong_silence class=bad>🕳 silence where music should be</button>
        <button data-t=crisp>✨ crisp where voice removed</button></div>
      <textarea class=cmt placeholder="comments — anything you hear (voice? where? what kind of damage?)" style="width:100%;min-height:54px;margin-top:8px;font:14px system-ui;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:8px;box-sizing:border-box"></textarea>
      <div style="display:flex;justify-content:space-between;align-items:center;margin-top:6px">
        <span class=csaved style="font-size:12px;color:var(--dim)"></span>
        <button class=rst style="font-size:12px;padding:8px 14px;color:var(--dim)">↺ reset</button></div>`;
    {
      const cmt=c.querySelector('.cmt'), lab=c.querySelector('.csaved');
      let timer=null, last=null;
      const saveC=async()=>{
        const txt=cmt.value.trim();
        if(txt===last)return; last=txt;
        await fetch('api/vote',{method:'POST',body:JSON.stringify({track:'pick:'+f,vote:'comment',text:txt,user:rater||'anon'})});
        lab.textContent='💬 saved'; setTimeout(()=>lab.textContent='',1200);
      };
      cmt.addEventListener('input',()=>{clearTimeout(timer);timer=setTimeout(saveC,900);});
      cmt.addEventListener('blur',()=>{clearTimeout(timer);saveC();});
    }
    c.querySelector('.rst').onclick=async()=>{
      await fetch('api/vote',{method:'POST',body:JSON.stringify({track:'pick:'+f,vote:'clear',user:rater||'anon'})});
      c.querySelectorAll('button.b5,button[data-t]').forEach(x=>x.classList.remove('voted'));
      c.querySelector('.cmt').value='';
      c.querySelector('.done').textContent='';
    };
    c.querySelectorAll('button.b5').forEach(b=>b.onclick=async()=>{
      await fetch('api/vote',{method:'POST',body:JSON.stringify({track:'pick:'+f,vote:b.dataset.v,user:rater||'anon'})});
      c.querySelectorAll('button.b5').forEach(x=>x.classList.remove('voted'));b.classList.add('voted');
      c.querySelector('.done').textContent='✓ '+(rater||'');
    });
    c.querySelectorAll('button[data-t]').forEach(b=>b.onclick=async()=>{
      const on=!b.classList.contains('voted');
      await fetch('api/vote',{method:'POST',body:JSON.stringify({track:'pick:'+f,vote:(on?'tag_':'untag_')+b.dataset.t,user:rater||'anon'})});
      b.classList.toggle('voted',on);
      c.querySelector('.done').textContent='✓ '+(rater||'');
    });
    const prev=votes['pick:'+f]||{};
    if(prev.stars){const b=c.querySelector(`button[data-v=pick5_${prev.stars}]`);if(b)b.classList.add('voted');c.querySelector('.done').textContent='✓ '+(rater||'');}
    if(prev.pick&&!prev.stars){const m={like:5,meh:3,dislike:1}[prev.pick];const b=c.querySelector(`button[data-v=pick5_${m}]`);if(b)b.classList.add('voted');}
    (prev.tags||[]).forEach(t=>{const b=c.querySelector(`button[data-t=${t}]`);if(b)b.classList.add('voted');});
    if(prev.comment)c.querySelector('.cmt').value=prev.comment;
    el.appendChild(c);
  }
  loadActions();
}
if(rater)main();
document.addEventListener('click',e=>{
  const b=e.target.closest('.dlbtn');
  document.querySelectorAll('.dlmenu').forEach(m=>{if(!b||m!==b.nextElementSibling)m.style.display='none';});
  if(b){e.preventDefault();const m=b.nextElementSibling;m.style.display=(m.style.display==='none')?'':'none';}
});
document.addEventListener('play',e=>{
  document.querySelectorAll('audio').forEach(a=>{if(a!==e.target)a.pause();});
},true);
</script>"""



VARIANTS_PAGE = """<!doctype html><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Tobacco variants A–E</title>
<style>
 :root{--bg:#10151c;--card:#182029;--line:#26313d;--fg:#e8edf3;--dim:#8b98a8;--acc:#2e6fb7;--ok:#3a5;
  --cA:#2e6fb7;--cB:#22c3e0;--cC:#3aa563;--cD:#d98c2b;--cE:#9b6dd6;--cF:#e0567f}
 *{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
 body{font:16px/1.45 system-ui;margin:0;background:var(--bg);color:var(--fg);padding:12px}
 h1{font-size:18px;margin:4px 0 2px}.sub{color:var(--dim);font-size:13px;margin:0 0 12px}
 .card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px;margin:0 0 12px}
 .t{font-weight:600;margin-bottom:8px;font-size:15px}
 audio{width:100%;margin:4px 0 8px}
 .vrow{display:grid;grid-template-columns:repeat(5,1fr);gap:6px;margin:6px 0}
 button{font:inherit;font-size:15px;font-weight:700;border:1px solid #33414f;background:#202b36;color:var(--fg);border-radius:10px;padding:12px 4px;cursor:pointer}
 button:disabled{opacity:.3}
 .lab{font-size:10px;font-weight:400;display:block;color:var(--dim)}
 button.on .lab{color:#eee}
 .vA.on{background:var(--cA);border-color:var(--cA)} .vB.on{background:var(--cB);border-color:var(--cB);color:#06232a}
 .vC.on{background:var(--cC);border-color:var(--cC)} .vD.on{background:var(--cD);border-color:var(--cD);color:#241503}
 .vE.on{background:var(--cE);border-color:var(--cE)}
 .vF.on{background:var(--cF);border-color:var(--cF)}
 .drow{display:grid;grid-template-columns:repeat(5,1fr);gap:6px}
 .dlwrap{position:relative;display:block}
 .dlbtn{width:100%;font-size:12px;padding:9px 2px;color:var(--dim);background:transparent;border:1px solid #33414f;border-radius:8px;font-weight:400}
 .dlmenu{position:absolute;top:105%;left:0;right:0;background:var(--card);border:1px solid var(--line);border-radius:8px;z-index:6;overflow:hidden}
 .dlmenu a{display:block;text-align:center;padding:9px 4px;font-size:13px;color:var(--fg);text-decoration:none;border:none}
 .dlmenu a:hover{background:var(--acc)}
 button.voted{background:var(--ok)!important;border-color:var(--ok)!important;color:#fff!important;font-weight:800}
 .drow a{display:block;text-align:center;font-size:12px;padding:9px 2px;border:1px solid #33414f;border-radius:8px;color:var(--dim);text-decoration:none}
 textarea{width:100%;min-height:48px;margin-top:8px;font:14px system-ui;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:8px}
 #gate{position:fixed;inset:0;background:var(--bg);z-index:9;display:flex;flex-direction:column;justify-content:center;padding:32px;gap:12px}
 #gate input{font:20px system-ui;padding:14px;border-radius:10px;border:1px solid var(--line);background:var(--card);color:var(--fg)}
 #gate button{font-size:18px;background:var(--acc);border-color:var(--acc)}
 .csaved{font-size:12px;color:var(--dim)}
</style>
<div id=who style="position:fixed;top:8px;right:10px;z-index:8;font-size:12px;background:var(--card);border:1px solid var(--line);border-radius:16px;padding:6px 12px;display:none;color:var(--dim)">👤 <b id=whon style="color:var(--fg)"></b> <a href=# id=logout style="color:#c33;text-decoration:none;margin-left:8px">logout</a></div>
<div id=gate><div style="font-size:22px;font-weight:700">Who's listening?</div>
 <input id=nm placeholder="name" autocapitalize=none>
 <button onclick="setName()">Start</button></div>
<h1>🚬 Tobacco tracks — variants A–E</h1>
<p class=sub>Switch A–E seamlessly at the same spot. ⬇ downloads that variant with the right filename.</p>
<div id=cards></div>
<script>
const $=q=>document.querySelector(q);
let rater=localStorage.rater||'';
function showWho(){const w=document.getElementById('who');if(rater&&w){w.style.display='';document.getElementById('whon').textContent=rater;}}
document.addEventListener('click',e=>{if(e.target&&e.target.id==='logout'){e.preventDefault();localStorage.removeItem('rater');location.reload();}});
if(rater){$('#gate').style.display='none';showWho();}
function setName(){const v=$('#nm').value.trim();if(!v)return;rater=v;localStorage.rater=v;$('#gate').style.display='none';showWho();main();}
let activeCard=null;
let VADTL={};
const esc=t=>t.replace(/&/g,'&amp;').replace(/</g,'&lt;');
async function loadActions(){
  const A=await (await fetch('api/actions')).json().catch(()=>({}));
  for(const t in (window._actlogs||{})){
    const log=window._actlogs[t]; const items=A[t]||[];
    log.innerHTML=items.slice(-6).map(x=>x.kind==='action'
      ?`<div>🔧 <b>${esc(x.user)}</b>: ${esc(x.text)} <span style="color:var(--dim)">(${x.ts.slice(5,16)})</span></div>`
      :`<div style="color:var(--ok)">↳ <b>engineer</b>: ${esc(x.text)}</div>`).join('') || '';
  }
}
setInterval(loadActions,30000);
async function main(){
  const groups=await (await fetch('api/variants')).json();
  VADTL=await (await fetch('api/vadtl')).json().catch(()=>({}));
  const votes=await (await fetch('api/votes?user='+encodeURIComponent(rater||'anon'))).json().catch(()=>({}));
  const el=$('#cards');
  {
    const uc=document.createElement('div');uc.className='card';
    uc.innerHTML=`<div class=t>⬆️ Upload a new track</div>
      <input type=file id=upf accept=".mp3,.wav,.m4a,.flac,.aiff" style="display:none">
      <button id=upb style="width:100%;padding:16px;font-size:17px;font-weight:700;background:var(--acc);border-color:var(--acc)">⬆️ Upload & analyze</button>
      <div id=upbox style="display:none;margin-top:10px;padding:12px;border:1px solid var(--line);border-radius:10px;background:var(--bg)">
        <div id=upname style="font-weight:600"></div>
        <div id=upmeta style="font-size:13px;color:var(--dim);margin-top:3px"></div>
        <div style="height:7px;background:#243040;border-radius:4px;margin-top:9px"><div id=uppr style="height:100%;width:0;background:var(--acc);border-radius:4px"></div></div>
        <div id=upst style="font-size:13.5px;margin-top:9px"></div>
      </div>`;
    el.appendChild(uc);
    const inp=uc.querySelector('#upf'), box=uc.querySelector('#upbox'), st=uc.querySelector('#upst'),
          nm=uc.querySelector('#upname'), meta=uc.querySelector('#upmeta'), pr=uc.querySelector('#uppr');
    const fmtDur=d=>isFinite(d)?Math.floor(d/60)+':'+String(Math.round(d%60)).padStart(2,'0'):'?';
    uc.querySelector('#upb').onclick=()=>{ inp.click(); };
    inp.onchange=()=>{
      const f=inp.files[0]; if(!f)return;
      box.style.display=''; pr.style.width='0';
      nm.textContent='🎵 '+f.name;
      meta.textContent=(f.size/1e6).toFixed(1)+' MB · reading duration…';
      const au=new Audio(); au.preload='metadata'; au.src=URL.createObjectURL(f);
      au.onloadedmetadata=()=>{ meta.textContent=(f.size/1e6).toFixed(1)+' MB · '+fmtDur(au.duration)+' long'; URL.revokeObjectURL(au.src); };
      au.onerror=()=>{ meta.textContent=(f.size/1e6).toFixed(1)+' MB'; };
      st.textContent='uploading…';
      const xhr=new XMLHttpRequest();
      xhr.open('POST','api/upload?name='+encodeURIComponent(f.name));
      xhr.upload.onprogress=e=>{ if(e.lengthComputable) pr.style.width=Math.round(100*e.loaded/e.total)+'%'; };
      xhr.onload=()=>{
        let j=null; try{ j=JSON.parse(xhr.responseText); }catch(e){}
        if(!j||!j.ok){ st.textContent='❌ '+(j&&j.err?j.err:('server said HTTP '+xhr.status)); return; }
        pr.style.width='100%';
        st.innerHTML='✅ uploaded as <b>'+j.file+'</b> — pipeline started. <a href="w?t='+encodeURIComponent(j.track)+'" style="color:var(--acc);font-weight:700">Watch the analysis live →</a>';
        inp.value='';
      };
      xhr.onerror=()=>{ st.textContent='❌ network error during upload'; };
      xhr.send(f);
    };
  }
  for(const g of groups){
    if(!g.variants.length){
      const c=document.createElement('div');c.className='card';
      c.innerHTML=`<div class=t>${g.track.replace(/_/g,' ')} <span style="color:var(--dim);font-size:12px">⏳ analyzing…</span></div>
        <a href="w?t=${encodeURIComponent(g.track)}" style="display:block;text-align:center;padding:13px;font-size:15px;border-radius:12px;background:#202b36;border:1px solid var(--line);color:var(--fg);text-decoration:none">watch the analysis live →</a>`;
      el.appendChild(c);
      continue;
    }
    const c=document.createElement('div');c.className='card';
    const letters=g.variants.map(v=>v.letter);
    c.innerHTML=`<div class=t>${g.track.replace(/_/g,' ')}</div>
      <div class=players></div>
      <canvas class=vadc height=30 style="width:100%;height:30px;display:block;margin:6px 0 2px;border-radius:4px"></canvas>
      <div style="font-size:10px;color:var(--dim);margin-bottom:4px">top: detector confidence (green = sure, yellow = unsure) · bottom: <span style="color:#c33">red = voice heard</span> / <span style="color:#5c9fe0">blue = clean</span> · tap bar to seek</div>
      <div class=vrow style="grid-template-columns:repeat(6,1fr)">${['A','B','C','D','E','F'].map(L=>{
        const v=g.variants.find(x=>x.letter===L);
        return `<button class="v${L} sw" data-l=${L} ${v?'':'disabled'}>${L}<span class=lab>${v?v.label:'—'}</span></button>`;}).join('')}</div>
      <div style="display:flex;align-items:center;gap:10px;margin-top:10px">
        <span style="font-size:14px;color:var(--dim);white-space:nowrap">👍 I like:</span>
        <select class=bestsel style="flex:1;font:16px system-ui;padding:11px 10px;border-radius:10px;background:#202b36;color:var(--fg);border:1px solid #33414f">
          <option value="">— choose —</option>
          ${g.variants.map(v=>`<option value="${v.letter}">${v.letter} — ${v.label}</option>`).join('')}
          <option value="none">none of them</option>
        </select><span class=bestok style="color:var(--ok);font-size:15px;min-width:20px"></span></div>
      <div class=drow style="grid-template-columns:repeat(6,1fr)">${['A','B','C','D','E','F'].map(L=>{
        const v=g.variants.find(x=>x.letter===L);
        if(!v)return `<span class=dlwrap><a style="opacity:.25;display:block;text-align:center;padding:9px 2px">—</a></span>`;
        const u=encodeURIComponent(v.file);
        return `<span class=dlwrap><button class=dlbtn>⬇ ${L}</button><span class=dlmenu style="display:none">
          <a href="vaudio/${u}" download="${g.track}_${L}.m4a">m4a</a>
          <a href="wav/${u}" download="${g.track}_${L}.wav">wav</a></span></span>`;}).join('')}</div>
      <div style="display:flex;justify-content:flex-start;margin-top:4px"><span class=csaved></span></div>
      <textarea class=cmt placeholder="💬 comments (which variant, what you hear, where)"></textarea>
      <a href="w?t=${encodeURIComponent(g.track)}" style="display:block;text-align:center;margin-top:10px;padding:15px;font-size:17px;font-weight:700;border-radius:12px;background:var(--acc);color:#fff;text-decoration:none">🎙 Talk to the AI about this piece</a>`;
    const players=c.querySelector('.players');
    const els={};
    for(const v of g.variants){
      const a=document.createElement('audio');
      a.controls=true; a.preload='metadata';
      a.src='vaudio/'+encodeURIComponent(v.file);
      if(v.letter!==letters[0])a.style.display='none';
      players.appendChild(a); els[v.letter]=a;
    }
    let act=letters[0];
    c.querySelector('.v'+act).classList.add('on');
    const cv=c.querySelector('.vadc'), cx=cv.getContext('2d');
    function drawTL(){
      const v=g.variants.find(x=>x.letter===act);
      const d=v?VADTL[v.file]:null;
      if(!d){cv.style.display='none';return;}
      cv.style.display='';
      const W=cv.clientWidth||600; if(cv.width!==W)cv.width=W;
      const P=d.p, bw=W/P.length;
      cx.clearRect(0,0,W,30);
      for(let i=0;i<P.length;i++){
        const pp=P[i], conf=Math.min(1,Math.abs(pp-0.5)*2);
        const r=Math.round(224-(224-47)*conf), gg=Math.round(192-(192-125)*conf), b=Math.round(74-(74-79)*conf);
        cx.fillStyle=`rgb(${r},${gg},${b})`; cx.fillRect(i*bw,0,bw+1,12);
        cx.fillStyle = pp>=d.thr ? '#c33' : '#2e6fb7'; cx.fillRect(i*bw,16,bw+1,12);
      }
      const a=els[act];
      if(a&&a.duration){const x=(a.currentTime/a.duration)*W; cx.fillStyle='#fff'; cx.fillRect(x-1,0,2,30);}
    }
    setInterval(drawTL,150); setTimeout(drawTL,300);
    cv.onclick=e=>{const a=els[act]; if(!a||!a.duration)return;
      const r=cv.getBoundingClientRect(); a.currentTime=((e.clientX-r.left)/r.width)*a.duration;};
    const sync=()=>{
      for(const L of letters){const a=els[L];
        if(L!==act && !a.paused) a.pause(); } };
    setInterval(sync,1200);
    els[letters[0]].addEventListener('play',()=>{
      if(activeCard&&activeCard!==c){activeCard.querySelectorAll('audio').forEach(a=>a.pause());}
      activeCard=c; sync();
    });
    for(const L of letters){
      els[L].addEventListener('play',()=>{ if(L===act){ if(activeCard&&activeCard!==c){activeCard.querySelectorAll('audio').forEach(a=>a.pause());} activeCard=c; sync(); } });
      els[L].addEventListener('seeked',()=>{ if(L===act) sync(); });
    }
    c.querySelectorAll('button.sw').forEach(b=>b.onclick=()=>{
      const L=b.dataset.l; if(!els[L]||L===act)return;
      const prev=els[act], playing=!prev.paused;
      if(Math.abs(els[L].currentTime-prev.currentTime)>0.08)els[L].currentTime=prev.currentTime;
      prev.muted=true; prev.style.display='none';
      els[L].muted=false; els[L].style.display='';
      act=L;
      c.querySelectorAll('button.sw').forEach(x=>x.classList.toggle('on',x.dataset.l===L));
      if(playing){els[L].play().catch(()=>{});}else{els[L].pause();}
      drawTL();
    });
    {
      const sel=c.querySelector('.bestsel'), ok=c.querySelector('.bestok');
      sel.onchange=async()=>{
        if(!sel.value)return;
        await fetch('api/vote',{method:'POST',body:JSON.stringify({track:'var:'+g.track,vote:'best_'+sel.value,user:rater||'anon'})});
        ok.textContent='✓'; setTimeout(()=>ok.textContent='',1500);
      };
    }
    {
      const cmt=c.querySelector('.cmt'), lab=c.querySelector('.csaved');
      let timer=null,last=null;
      const saveC=async()=>{const txt=cmt.value.trim(); if(txt===last)return; last=txt;
        await fetch('api/vote',{method:'POST',body:JSON.stringify({track:'var:'+g.track,vote:'comment',text:txt,user:rater||'anon'})});
        lab.textContent='💬 saved'; setTimeout(()=>lab.textContent='',1200);};
      cmt.addEventListener('input',()=>{clearTimeout(timer);timer=setTimeout(saveC,900);});
      cmt.addEventListener('blur',()=>{clearTimeout(timer);saveC();});
      const prev=votes['var:'+g.track]||{};
      if(prev.comment)cmt.value=prev.comment;
      if(prev.best){const sel=c.querySelector('.bestsel'); sel.value=prev.best; c.querySelector('.bestok').textContent='✓';}
    }
    el.appendChild(c);
  }
}
document.addEventListener('click',function(e){
  var b=e.target.closest && e.target.closest('.dlbtn');
  document.querySelectorAll('.dlmenu').forEach(function(m){ if(!b||m!==b.nextElementSibling) m.style.display='none'; });
  if(b){ e.preventDefault(); var m=b.nextElementSibling; m.style.display=(m.style.display==='none'||!m.style.display)?'block':'none'; }
});
if(rater)main();
</script>"""



WORKSPACE_PAGE = """<!doctype html><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Track workspace</title>
<style>
 :root{--bg:#10151c;--card:#182029;--line:#26313d;--fg:#e8edf3;--dim:#8b98a8;--acc:#2e6fb7;--ok:#3a5;--warn:#a55;
  --cA:#2e6fb7;--cB:#22c3e0;--cC:#3aa563;--cD:#d98c2b;--cE:#9b6dd6;--cF:#e0567f}
 *{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
 body{font:16px/1.5 system-ui;margin:0;background:var(--bg);color:var(--fg);padding:12px;max-width:860px;margin:0 auto}
 h1{font-size:18px;margin:8px 0 2px} .sub{color:var(--dim);font-size:13px;margin:0 0 10px}
 .card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px;margin:0 0 12px}
 audio{width:100%;margin:4px 0 8px}
 .vrow{display:grid;grid-template-columns:repeat(6,1fr);gap:6px;margin:6px 0}
 .vrow button{font-size:15px;font-weight:700;padding:12px 4px}
 button{font:inherit;border:1px solid #33414f;background:#202b36;color:var(--fg);border-radius:10px;cursor:pointer}
 button:disabled{opacity:.3}
 .lab{font-size:10px;font-weight:400;display:block;color:var(--dim)}
 .vA.on{background:var(--cA);border-color:var(--cA)} .vB.on{background:var(--cB);border-color:var(--cB);color:#06232a}
 .vC.on{background:var(--cC);border-color:var(--cC)} .vD.on{background:var(--cD);border-color:var(--cD);color:#241503}
 .vE.on{background:var(--cE);border-color:var(--cE)} .vF.on{background:var(--cF);border-color:var(--cF)}
 .drow{display:grid;grid-template-columns:repeat(6,1fr);gap:6px;margin-top:6px}
 .dlwrap{position:relative;display:block}
 .dlbtn{width:100%;font-size:12px;padding:9px 2px;color:var(--dim);background:transparent;border:1px solid #33414f;border-radius:8px;font-weight:400}
 .dlmenu{position:absolute;top:105%;left:0;right:0;background:var(--card);border:1px solid var(--line);border-radius:8px;z-index:6;overflow:hidden}
 .dlmenu a{display:block;text-align:center;padding:9px 4px;font-size:13px;color:var(--fg);text-decoration:none;border:none}
 .dlmenu a:hover{background:var(--acc)}

 #thread{display:flex;flex-direction:column;gap:8px;margin:10px 0}
 .msg{max-width:85%;padding:10px 13px;border-radius:14px;font-size:14.5px;white-space:pre-wrap;word-break:break-word}
 .m-comment{background:#202b36;align-self:flex-start;border:1px solid var(--line)}
 .m-action{background:#3a2226;align-self:flex-start;border:1px solid #a55}
 .m-reply{background:#1d2f24;align-self:flex-end;border:1px solid #3a5}
 .m-best{background:transparent;align-self:center;color:var(--dim);font-size:12.5px;border:none;padding:2px}
 .who{font-size:11px;color:var(--dim);margin-bottom:3px}
 .msg table{border-collapse:collapse;margin:6px 0;font-size:13px;background:var(--bg)}
 .msg th,.msg td{border:1px solid var(--line);padding:4px 9px;text-align:left}
 .msg th{color:var(--dim);font-size:11px;text-transform:uppercase}
 .msg code{background:var(--bg);border:1px solid var(--line);border-radius:4px;padding:0 4px}
 .att{margin-top:8px}.att a{color:var(--acc);text-decoration:none;font-size:13px}
 textarea{width:100%;min-height:56px;font:15px system-ui;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:10px;padding:10px}
 .send{display:flex;gap:8px;margin-top:8px}
 .send button{flex:1;padding:12px 8px;font-size:15px}
 #gate{position:fixed;inset:0;background:var(--bg);z-index:9;display:flex;flex-direction:column;justify-content:center;padding:32px;gap:12px}
 #gate input{font:20px system-ui;padding:14px;border-radius:10px;border:1px solid var(--line);background:var(--card);color:var(--fg)}
 #gate button{font-size:18px;background:var(--acc);border-color:var(--acc);padding:12px}
</style>
<div id=who style="position:fixed;top:8px;right:10px;z-index:8;font-size:12px;background:var(--card);border:1px solid var(--line);border-radius:16px;padding:6px 12px;display:none;color:var(--dim)">👤 <b id=whon style="color:var(--fg)"></b> <a href=# id=logout style="color:#c33;text-decoration:none;margin-left:8px">logout</a></div>
<div id=gate><div style="font-size:22px;font-weight:700">Who's listening?</div>
 <input id=nm placeholder="name" autocapitalize=none><button onclick="setName()">Start</button></div>
<a href="v" style="color:var(--dim);font-size:13px;text-decoration:none">← all tracks</a>
<h1 id=title>…</h1>
<p class=sub>One page per master: audition, chat, downloads. 🔧 messages are work orders — the engineer replies here and re-renders.</p>
<div class=card id=player style="display:none">
 <div class=players></div>
 <canvas class=vadc height=30 style="width:100%;height:30px;display:block;margin:6px 0 2px;border-radius:4px"></canvas>
 <div style="font-size:10px;color:var(--dim)">top: detector confidence · bottom: <span style="color:#c33">red = voice</span>/<span style="color:#5c9fe0">blue = clean</span> · tap to seek</div>
 <div class=vrow></div>
 <div class=drow></div>
</div>
<div class=card>
 <div style="display:flex;justify-content:space-between;align-items:center">
   <b style="font-size:15px">Conversation</b>
   <button id=dl style="font-size:12px;padding:7px 12px;color:var(--dim)">⬇ transcript</button></div>
 <div id=thread></div>
 <textarea id=box placeholder="write here… (💬 = note it, 🔧 = ask the engineer to fix/re-render)"></textarea>
 <div class=send>
   <button id=sc>💬 comment</button>
   <button id=sa style="border-color:var(--warn)">🔧 action / re-render</button></div>
</div>
<script>
const $=q=>document.querySelector(q);
const TRACK=new URLSearchParams(location.search).get('t')||'';
document.title=TRACK+' — workspace';
$('#title').textContent=TRACK.replace(/_/g,' ');
let rater=localStorage.rater||'';
function showWho(){const w=document.getElementById('who');if(rater&&w){w.style.display='';document.getElementById('whon').textContent=rater;}}
document.addEventListener('click',e=>{if(e.target&&e.target.id==='logout'){e.preventDefault();localStorage.removeItem('rater');location.reload();}});
if(rater){$('#gate').style.display='none';showWho();}
function setName(){const v=$('#nm').value.trim();if(!v)return;rater=v;localStorage.rater=v;$('#gate').style.display='none';showWho();boot();}
const esc=t=>String(t).replace(/&/g,'&amp;').replace(/</g,'&lt;');
function md(raw){
  let t=esc(raw);
  const lines=t.split(String.fromCharCode(10)); const out=[]; let i=0;
  while(i<lines.length){
    if(lines[i].includes('|') && i+1<lines.length && /^\s*\|?[\s:|-]+$/.test(lines[i+1]) && lines[i+1].includes('-')){
      const cells=l=>l.split('|').map(c=>c.trim()).filter((c,k,arr)=>!(c===''&&(k===0||k===arr.length-1)));
      const hdr=cells(lines[i]); let j=i+2; const rows=[];
      while(j<lines.length && lines[j].includes('|')){rows.push(cells(lines[j]));j++;}
      out.push('<table><tr>'+hdr.map(h=>'<th>'+h+'</th>').join('')+'</tr>'+
        rows.map(r=>'<tr>'+r.map(c=>'<td>'+c+'</td>').join('')+'</tr>').join('')+'</table>');
      i=j; continue;
    }
    out.push(lines[i]); i++;
  }
  t=out.join(String.fromCharCode(10));
  t=t.replace(/\*\*([^*]+)\*\*/g,'<b>$1</b>').replace(/`([^`]+)`/g,'<code>$1</code>');
  t=t.replace(/^### (.*)$/gm,'<b>$1</b>').replace(/^## (.*)$/gm,'<b style="font-size:16px">$1</b>');
  t=t.replace(/^- /gm,'• ');
  return t.replace(new RegExp(String.fromCharCode(10),'g'),'<br>').replace(/<\/table><br>/g,'</table>');
}
let els={}, act=null, letters=[], VADTL={}, g=null;
async function boot(){
  const groups=await (await fetch('api/variants')).json();
  VADTL=await (await fetch('api/vadtl')).json().catch(()=>({}));
  g=groups.find(x=>x.track===TRACK);
  if(g&&!g.variants.length){
    document.querySelector('.sub').textContent='⏳ pipeline is analyzing this upload — steps appear below as they happen (auto-refreshes).';
    g=null;
  }
  if(g){
    $('#player').style.display='';
    letters=g.variants.map(v=>v.letter);
    const players=$('#player .players'), vrow=$('#player .vrow'), drow=$('#player .drow');
    for(const v of g.variants){
      const a=document.createElement('audio');
      a.controls=true;a.preload='metadata';a.src='vaudio/'+encodeURIComponent(v.file);
      if(v.letter!==letters[0])a.style.display='none';
      players.appendChild(a);els[v.letter]=a;
    }
    act=letters[0];
    vrow.innerHTML=['A','B','C','D','E','F'].map(L=>{
      const v=g.variants.find(x=>x.letter===L);
      return `<button class="v${L} sw" data-l=${L} ${v?'':'disabled'}>${L}<span class=lab>${v?v.label:'—'}</span></button>`;}).join('');
    vrow.querySelector('.v'+act).classList.add('on');
    drow.innerHTML=['A','B','C','D','E','F'].map(L=>{
      const v=g.variants.find(x=>x.letter===L);
      if(!v)return `<span class=dlwrap><a style="opacity:.25;display:block;text-align:center;padding:9px 2px">—</a></span>`;
      const u=encodeURIComponent(v.file);
      return `<span class=dlwrap><button class=dlbtn>⬇ ${L}</button><span class=dlmenu style="display:none">
        <a href="vaudio/${u}" download="${TRACK}_${L}.m4a">m4a</a>
        <a href="wav/${u}" download="${TRACK}_${L}.wav">wav</a></span></span>`;}).join('');
    const sync=()=>{
      for(const L of letters){const a=els[L];
        if(L!==act && !a.paused) a.pause(); } };
    setInterval(sync,1200);
    for(const L of letters){
      els[L].addEventListener('seeked',()=>{if(L===act)sync();});
    }
    vrow.querySelectorAll('button.sw').forEach(b=>b.onclick=()=>{
      const L=b.dataset.l;if(!els[L]||L===act)return;
      const prev=els[act],playing=!prev.paused;
      if(Math.abs(els[L].currentTime-prev.currentTime)>0.08)els[L].currentTime=prev.currentTime;
      prev.muted=true;prev.style.display='none';
      els[L].muted=false;els[L].style.display='';act=L;
      vrow.querySelectorAll('button.sw').forEach(x=>x.classList.toggle('on',x.dataset.l===L));
      if(playing)els[L].play().catch(()=>{});else els[L].pause();
      drawTL();
    });
    const cv=$('#player .vadc'),cx=cv.getContext('2d');
    function drawTL(){
      const v=g.variants.find(x=>x.letter===act);
      const d=v?VADTL[v.file]:null;
      if(!d){cv.style.display='none';return;}
      cv.style.display='';
      const W=cv.clientWidth||600;if(cv.width!==W)cv.width=W;
      const P=d.p,bw=W/P.length;cx.clearRect(0,0,W,30);
      for(let i=0;i<P.length;i++){
        const pp=P[i],conf=Math.min(1,Math.abs(pp-0.5)*2);
        const r=Math.round(224-(224-47)*conf),gg=Math.round(192-(192-125)*conf),b=Math.round(74-(74-79)*conf);
        cx.fillStyle=`rgb(${r},${gg},${b})`;cx.fillRect(i*bw,0,bw+1,12);
        cx.fillStyle=pp>=d.thr?'#c33':'#2e6fb7';cx.fillRect(i*bw,16,bw+1,12);
      }
      const a=els[act];
      if(a&&a.duration){const x=(a.currentTime/a.duration)*W;cx.fillStyle='#fff';cx.fillRect(x-1,0,2,30);}
    }
    window._drawTL=drawTL;setInterval(drawTL,150);setTimeout(drawTL,300);
    cv.onclick=e=>{const a=els[act];if(!a||!a.duration)return;
      const r=cv.getBoundingClientRect();a.currentTime=((e.clientX-r.left)/r.width)*a.duration;};
  }
  loadThread();
}
let _lastThreadJson='';
async function loadThread(){
  const ev=await (await fetch('api/thread?t='+encodeURIComponent(TRACK))).json().catch(()=>[]);
  window._thread=ev;
  const j=JSON.stringify(ev);
  const playing=[...document.querySelectorAll('#thread audio')].some(a=>!a.paused);
  if(j===_lastThreadJson || playing) return;   // never rebuild while listening / nothing new
  _lastThreadJson=j;
  $('#thread').innerHTML=ev.map(x=>{
    if(x.kind==='best')return `<div class="msg m-best">👍 ${esc(x.user)} picked ${esc(x.text)} · ${x.ts.replace('T',' ')}</div>`;
    const cls=x.kind==='reply'?'m-reply':(x.kind==='action'?'m-action':'m-comment');
    const tag=x.kind==='action'?'🔧 ':'';
    const body=x.kind==='reply'?md(x.text):esc(x.text);
    let att='';
    const files=x.files||(x.file?[x.file]:[]);
    for(const fn of files){
      const u='afile/'+encodeURIComponent(fn);
      const isAudio=/\.(m4a|mp3|wav|flac)$/i.test(fn);
      const isImg=/\.(png|jpe?g|webp|gif|svg)$/i.test(fn);
      const isVid=/\.(mp4|webm)$/i.test(fn);
      att+=`<div class=att>${isImg?`<a href="${u}" target=_blank><img src="${u}" style="max-width:100%;border-radius:8px;border:1px solid var(--line)"></a>`:''}${isVid?`<video controls preload=metadata src="${u}" style="width:100%;border-radius:8px;border:1px solid var(--line)"></video>`:''}${isAudio?`<audio controls preload=none src="${u}" style="width:100%"></audio>`:''}<a href="${u}" download>⬇ ${esc(fn)}</a></div>`;
    }
    return `<div class="msg ${cls}"><div class=who>${tag}${esc(x.user)} · ${x.ts.replace('T',' ')}</div>${body}${att}</div>`;
  }).join('');
}
setInterval(loadThread,20000);
$('#sc').onclick=async()=>{const t=$('#box').value.trim();if(!t)return;
  await fetch('api/vote',{method:'POST',body:JSON.stringify({track:'var:'+TRACK,vote:'comment',text:t,user:rater||'anon'})});
  $('#box').value='';loadThread();};
$('#sa').onclick=async()=>{const t=$('#box').value.trim();if(!t)return;
  await fetch('api/vote',{method:'POST',body:JSON.stringify({track:'var:'+TRACK,vote:'action',text:t,user:rater||'anon'})});
  $('#box').value='';loadThread();};
$('#dl').onclick=()=>{
  const ev=window._thread||[];
  const txt=ev.map(x=>`[${x.ts}] ${x.kind.toUpperCase()} ${x.user}: ${x.text}`).join(String.fromCharCode(10));
  const b=new Blob([txt],{type:'text/plain'});
  const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download=TRACK+'_thread.txt';a.click();};
document.addEventListener('click',function(e){
  var b=e.target.closest && e.target.closest('.dlbtn');
  document.querySelectorAll('.dlmenu').forEach(function(m){ if(!b||m!==b.nextElementSibling) m.style.display='none'; });
  if(b){ e.preventDefault(); var m=b.nextElementSibling; m.style.display=(m.style.display==='none'||!m.style.display)?'block':'none'; }
});
if(rater)boot();
</script>"""


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    dir: str = "."

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/w" or self.path.startswith("/w?") or self.path.startswith("/w/"):
            b = WORKSPACE_PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
            return
        if self.path in ("/v", "/v/"):
            b = VARIANTS_PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
            return
        if self.path.startswith("/api/thread"):
            from urllib.parse import parse_qs, urlparse
            track = (parse_qs(urlparse(self.path).query).get("t") or [""])[0]
            vf = os.path.join(self.dir, "ab_verdicts.json")
            ev: list[dict] = []
            try:
                for v in json.load(open(vf)):
                    if str(v.get("track", "")) != "var:" + track:
                        continue
                    vote = str(v.get("vote", ""))
                    u = v.get("user", "?")
                    if vote == "comment":
                        if ev and ev[-1]["kind"] == "comment" and ev[-1]["user"] == u:
                            ev[-1].update(text=v.get("text", ""), ts=v.get("ts", ""))
                            continue
                        ev.append({"kind": "comment", "user": u, "text": v.get("text", ""), "ts": v.get("ts", "")})
                    elif vote == "action":
                        ev.append({"kind": "action", "user": u, "text": v.get("text", ""), "ts": v.get("ts", "")})
                    elif vote == "action_reply":
                        rec = {"kind": "reply", "user": v.get("user", "engineer"), "text": v.get("text", ""), "ts": v.get("ts", "")}
                        files = v.get("files") or ([v["file"]] if v.get("file") else [])
                        if files:
                            rec["files"] = files
                        ev.append(rec)
                    elif vote.startswith("best_"):
                        ev.append({"kind": "best", "user": u, "text": vote[5:], "ts": v.get("ts", "")})
            except Exception:
                pass
            self._json(ev)
            return
        if self.path == "/api/actions":
            vf = os.path.join(self.dir, "ab_verdicts.json")
            out: dict[str, list] = {}
            try:
                for v in json.load(open(vf)):
                    vote = str(v.get("vote", ""))
                    if vote not in ("action", "action_reply"):
                        continue
                    t = str(v.get("track", "")).replace("var:", "")
                    out.setdefault(t, []).append({
                        "kind": "action" if vote == "action" else "reply",
                        "user": v.get("user", "?"), "text": v.get("text", ""),
                        "ts": v.get("ts", "")})
            except Exception:
                pass
            self._json(out)
            return
        if self.path == "/api/vadtl":
            fp = os.path.join(ROOT, "variants", ".vad_timelines.json")
            try:
                b = open(fp, "rb").read()
            except OSError:
                b = b"{}"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
            return
        if self.path == "/api/variants":
            vd = os.path.join(ROOT, "variants")
            groups: dict[str, list] = {}
            try:
                for n in sorted(os.listdir(vd)):
                    m = re.match(r"^(?P<track>.+?)__(?P<letter>[A-F])_(?P<label>.+?)\.m4a$", n)
                    if m:
                        groups.setdefault(m.group("track"), []).append(
                            {"letter": m.group("letter"), "label": m.group("label"), "file": n})
            except OSError:
                pass
            out = [{"track": t, "variants": sorted(v, key=lambda x: x["letter"])}
                   for t, v in sorted(groups.items())]
            try:
                ups = json.load(open(os.path.join(ROOT, "variants", ".uploads.json")))
                have = {o["track"] for o in out}
                for t, meta in ups.items():
                    if t not in have and meta.get("status") == "processing":
                        out.append({"track": t, "variants": [], "processing": True})
            except Exception:
                pass
            self._json(out)
            return
        if self.path in ("/t", "/t/"):
            b = TRACKS_PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
            return
        if self.path == "/api/tracks":
            curated = [
                ("verdi_slow_voices__V5_ensmax.m4a", "VERDI — V5 ensmax (provisional ship leader)", ""),
                ("verdi_slow_voices__V2_mdx23c.m4a", "VERDI — V2 mdx23c (fullest clean candidate)", ""),
                ("nessun_dorma__V2_mdx23c.m4a", "NESSUN — V2 (passed your spot-checks)", "diagnostic lineage — re-render pending"),
                ("donna__V2_mdx23c.m4a", "DONNA — V2 (your trace flag @34.5s)", "diagnostic lineage — re-render pending"),
                ("barber_agnus_dei__V2_mdx23c.m4a", "BARBER — V2 (passed your spot-checks)", ""),
            ]
            out = [{"f": f, "src": "audio", "label": lab, "note": note}
                   for f, lab, note in curated
                   if os.path.isfile(os.path.join(self.dir, f))]
            dd = os.path.join(ROOT, "deliverables")
            try:
                names = [n for n in os.listdir(dd) if n.endswith(".m4a")]
                names.sort(key=lambda n: os.path.getmtime(os.path.join(dd, n)), reverse=True)
            except OSError:
                names = []
            out += [{"f": n, "src": "daudio",
                     "label": n.replace("_instrumental.m4a", "").replace("_", " "),
                     "note": "current delivery (baseline)"} for n in names[:10]]
            self._json(out)
            return
        if self.path == "/m/":
            self.send_response(301); self.send_header("Location", "/m"); self.end_headers(); return
        if self.path in ("/", "/index.html", "/m"):
            b = (MOBILE_PAGE if self.path == "/m" else PAGE).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        elif self.path == "/api/pairs":
            self._json(scan_pairs(self.dir))
        elif self.path == "/api/clips":
            self._json(scan_clips(self.dir))
        elif self.path.startswith("/api/votes"):
            from urllib.parse import parse_qs, urlparse
            want = (parse_qs(urlparse(self.path).query).get("user") or [None])[0]
            norm = lambda x: {"lena": "elena"}.get(str(x).strip().lower(), str(x).strip().lower())
            if want:
                want = norm(want)
            vf = os.path.join(self.dir, "ab_verdicts.json")
            latest: dict[str, dict] = {}
            try:
                for v in json.load(open(vf)):
                    t, vote = v.get("track"), str(v.get("vote", ""))
                    if not t:
                        continue
                    u = norm(v.get("user") or "mickg")  # untagged history = owner
                    if want and u != want:
                        continue
                    if vote == "clear":
                        latest.pop(t, None)
                        continue
                    d = latest.setdefault(t, {})
                    if vote.startswith("quality_"):
                        d["quality"] = vote
                    elif vote in ("fuller_A", "fuller_B", "equal"):
                        d["full"] = vote
                    elif vote.startswith(("trace_", "voice_", "words_")) and vote[-1] in "AB":
                        d["v" + vote[-1]] = vote.split("_")[0]
                    elif vote in ("like", "meh", "dislike"):
                        d["pick"] = vote
                    elif vote == "comment":
                        d["comment"] = v.get("text", "")
                    elif vote.startswith("best_"):
                        d["best"] = vote[5:]
                    elif vote.startswith("rank_") and len(vote.split("_")) == 3:
                        _, L, P = vote.split("_")
                        if L in "ABCDEF" and P in "12345":
                            d["rank" + L] = int(P)
                    elif vote.startswith("pick5_"):
                        d["stars"] = int(vote[-1])
                    elif vote.startswith("tag_"):
                        d.setdefault("tags", [])
                        if vote[4:] not in d["tags"]:
                            d["tags"].append(vote[4:])
                    elif vote.startswith("untag_"):
                        if vote[6:] in d.get("tags", []):
                            d["tags"].remove(vote[6:])
                    elif vote in ("dmg_A", "dmg_B"):
                        d.setdefault("flags", [])
                        if vote not in d["flags"]:
                            d["flags"].append(vote)
                    else:
                        d["voice"] = vote
            except Exception:
                pass
            self._json(latest)
        elif self.path.startswith("/wav/"):
            name = os.path.basename(unquote(self.path[len("/wav/"):]))
            src = os.path.join(ROOT, "variants", name)
            if not (name.endswith(".m4a") and os.path.isfile(src)):
                self.send_error(404)
                return
            cache = os.path.join(ROOT, "variants", ".wav_cache")
            os.makedirs(cache, exist_ok=True)
            out = os.path.join(cache, name[:-4] + ".wav")
            if not os.path.isfile(out):
                subprocess.run(["ffmpeg", "-nostdin", "-y", "-i", src, "-ar", "44100",
                                "-ac", "2", "-c:a", "pcm_s24le", out], capture_output=True)
            if not os.path.isfile(out):
                self.send_error(500)
                return
            size = os.path.getsize(out)
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(size))
            self.end_headers()
            with open(out, "rb") as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        return
            return
        elif self.path.startswith(("/audio/", "/daudio/", "/vaudio/", "/afile/")):
            pref = next(p for p in ("/daudio/", "/vaudio/", "/afile/", "/audio/") if self.path.startswith(p))
            name = os.path.basename(unquote(self.path[len(pref):]))
            base = {"/daudio/": os.path.join(ROOT, "deliverables"),
                    "/vaudio/": os.path.join(ROOT, "variants"),
                    "/afile/": os.path.join(ROOT, "attachments")}.get(pref, self.dir)
            fp = os.path.join(base, name)
            if not os.path.isfile(fp):
                self.send_error(404)
                return
            size = os.path.getsize(fp)
            rng = self.headers.get("Range")
            start, end = 0, size - 1
            if rng and rng.startswith("bytes="):
                a, _, b2 = rng[6:].partition("-")
                start = int(a) if a else max(0, size - int(b2 or 0))
                end = int(b2) if b2 and a else end
            length = end - start + 1
            self.send_response(206 if rng else 200)
            ext = os.path.splitext(fp)[1].lower()
            ctype = {".m4a": "audio/mp4", ".mp3": "audio/mpeg", ".flac": "audio/flac",
                     ".wav": "audio/wav", ".png": "image/png", ".jpg": "image/jpeg",
                     ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif",
                     ".svg": "image/svg+xml", ".mp4": "video/mp4", ".webm": "video/webm", ".txt": "text/plain; charset=utf-8",
                     ".json": "application/json"}.get(ext, "application/octet-stream")
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            if rng:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(length))
            self.end_headers()
            with open(fp, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    remaining -= len(chunk)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path.startswith("/api/upload"):
            from urllib.parse import parse_qs, urlparse
            import re as _re
            q = parse_qs(urlparse(self.path).query)
            raw = os.path.basename((q.get("name") or ["upload.mp3"])[0])
            raw = _re.sub(r"[^A-Za-z0-9._() -]", "_", raw)
            stem, ext = os.path.splitext(raw)
            n = int(self.headers.get("Content-Length", 0))

            def _drain():
                left = n
                while left > 0:
                    c = self.rfile.read(min(1 << 20, left))
                    if not c:
                        break
                    left -= len(c)

            if ext.lower() not in (".mp3", ".wav", ".m4a", ".flac", ".aiff"):
                _drain()
                self._json({"ok": False, "err": "unsupported extension (mp3/wav/m4a/flac/aiff)"}, 400)
                return
            if n <= 0 or n > 400 * 1024 * 1024:
                self.close_connection = True
                self._json({"ok": False, "err": "bad size (max 400 MB)"}, 400)
                return
            rawdir = os.path.join(ROOT, "raw_input")
            dest = os.path.join(rawdir, raw)
            k = 2
            while os.path.exists(dest):
                dest = os.path.join(rawdir, f"{stem}_{k}{ext}")
                k += 1
            remaining = n
            with open(dest, "wb") as f:
                while remaining > 0:
                    chunk = self.rfile.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    f.write(chunk)
                    remaining -= len(chunk)
            try:
                import shutil
                shutil.copy2(dest, os.path.join(ROOT, "input", os.path.basename(dest)))
            except Exception:
                pass
            track = _re.sub(r"[^a-z0-9]+", "_", os.path.splitext(os.path.basename(dest))[0].lower()).strip("_")[:60]
            up = os.path.join(ROOT, "variants", ".uploads.json")
            try:
                d = json.load(open(up))
            except Exception:
                d = {}
            d[track] = {"file": os.path.basename(dest), "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "status": "processing"}
            json.dump(d, open(up, "w"), indent=1)
            logf = open(f"/tmp/pipeline_{track}.log", "ab")
            subprocess.Popen([sys.executable, os.path.join(ROOT, "tools", "upload_pipeline.py"), dest, track],
                             stdout=logf, stderr=logf, start_new_session=True)
            self._json({"ok": True, "track": track, "file": os.path.basename(dest)})
            return
        if self.path == "/api/vote":
            n = int(self.headers.get("Content-Length", 0))
            try:
                v = json.loads(self.rfile.read(n))
            except Exception:
                self._json({"ok": False}, 400)
                return
            vf = os.path.join(self.dir, "ab_verdicts.json")
            votes = []
            if os.path.exists(vf):
                try:
                    votes = json.load(open(vf))
                except Exception:
                    votes = []
            rec = {"track": v.get("track"), "vote": v.get("vote"),
                   "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
            if v.get("user"):
                rec["user"] = str(v["user"])[:40]
            if v.get("text"):
                rec["text"] = str(v["text"])[:4000]
            if isinstance(v.get("files"), list):
                rec["files"] = [os.path.basename(str(x))[:200] for x in v["files"]][:12]
            votes.append(rec)
            json.dump(votes, open(vf, "w"), indent=1)
            self._json({"ok": True})
        else:
            self.send_error(404)

    def log_message(self, *a):  # quiet
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "ab_pairs"))
    ap.add_argument("--port", type=int, default=8766)
    a = ap.parse_args()
    os.makedirs(a.dir, exist_ok=True)
    H.dir = a.dir
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), H)
    print(f"A/B page: http://localhost:{a.port}  (pairs dir: {a.dir})")
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
