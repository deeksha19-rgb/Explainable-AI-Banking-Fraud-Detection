"""Build a single-file, dependency-free HTML dashboard from ./outputs.

Run:  python -m src.build_static_dashboard
Opens anywhere (double-click the file, or view it on a phone) - no server needed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .features import FEATURE_GROUPS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fraud Detection Console</title>
<style>
:root{--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;
--blue:#2a78d6;--orange:#eb6834;--red:#d03b3b;--amber:#b7791f;--hair:rgba(11,11,11,.10)}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;
--grid:#2c2c2a;--axis:#383835;--blue:#3987e5;--orange:#d95926;--amber:#e0a43a;--hair:rgba(255,255,255,.10)}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:22px;margin:4px 0}h2{font-size:16px;margin:0 0 8px}p.sub{color:var(--ink2);margin:0 0 14px;font-size:13px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:12px}
.card{background:var(--surface);border:1px solid var(--hair);border-radius:12px;padding:14px}
.kpi .v{font-size:26px;font-weight:650}.kpi .l{color:var(--ink2);font-size:12px}.kpi .d{color:var(--muted);font-size:12px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px;margin-bottom:12px}
svg{width:100%;height:auto;display:block}svg text{fill:var(--muted);font-size:11px}
.legend{display:flex;gap:14px;font-size:12px;color:var(--ink2);margin-bottom:4px;flex-wrap:wrap}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px}
.controls{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px}
select,input{background:var(--surface);color:var(--ink);border:1px solid var(--hair);border-radius:8px;padding:7px 9px;font:inherit}
.tw{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;color:var(--ink2);font-weight:600;padding:6px 8px;border-bottom:1px solid var(--axis);white-space:nowrap}
td{padding:7px 8px;border-bottom:1px solid var(--grid);vertical-align:top}
tr.row{cursor:pointer}tr.row:hover td{background:rgba(137,135,129,.08)}
.chip{padding:2px 8px;border-radius:99px;font-size:12px;font-weight:600;white-space:nowrap}
.High{background:rgba(208,59,59,.14);color:var(--red)}.Medium{background:rgba(183,121,31,.16);color:var(--amber)}
.bar{height:6px;border-radius:3px;background:var(--grid);min-width:60px}.bar b{display:block;height:6px;border-radius:3px;background:var(--blue)}
.detail td{background:rgba(137,135,129,.06)}.dgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}
.cb{display:grid;grid-template-columns:150px 1fr 56px;align-items:center;gap:8px;font-size:12px;margin:3px 0}
.cb .t{position:relative;height:12px}.cb .t span{position:absolute;top:0;height:12px;border-radius:3px}
.cm{display:grid;grid-template-columns:auto 1fr 1fr;gap:4px;font-size:13px}.cm div{padding:12px 8px;border-radius:8px;text-align:center}
.note{font-size:13px;color:var(--ink2);border-left:3px solid var(--orange);padding:4px 10px;margin-top:8px}
ul{margin:4px 0 0;padding-left:18px}
</style></head><body><div class="wrap">
<h1>🛡️ Bank Transaction Fraud Detection</h1>
<p class="sub" id="sub"></p>
<div class="kpis" id="kpis"></div>
<div class="grid2">
 <div class="card"><h2>Alerts vs confirmed fraud, by day</h2><div class="legend"><span><i style="background:var(--blue)"></i>Alerts raised</span><span><i style="background:var(--orange)"></i>Confirmed fraud</span></div><div id="daily"></div></div>
 <div class="card"><h2>Global drivers (permutation importance)</h2><div id="imp"></div></div>
</div>
<div class="grid2">
 <div class="card"><h2>ROC curve</h2><div id="roc"></div></div>
 <div class="card"><h2>Precision–recall curve</h2><div class="legend"><span><i style="background:var(--blue)"></i>Curve</span><span><i style="background:var(--orange)"></i>Operating point</span></div><div id="pr"></div></div>
</div>
<div class="grid2">
 <div class="card"><h2>Confusion matrix (test)</h2><div id="cm"></div></div>
 <div class="card"><h2>Recall by fraud pattern</h2><div id="bytype"></div>
 <div class="note">The "subtle" pattern (mildly elevated, otherwise normal behaviour) is mostly missed - catching it needs richer signals.</div></div>
</div>
<div class="card" style="margin-bottom:12px"><h2>Model comparison</h2><div class="tw"><table id="cmp"></table></div></div>
<div class="card"><h2>Suspicious transactions <span style="font-weight:400;color:var(--muted);font-size:13px">- tap a row to see why</span></h2>
<div class="controls"><select id="band"><option value="">All flagged</option><option>High</option><option>Medium</option></select>
<select id="chan"><option value="">All channels</option></select><input id="q" placeholder="Search ID / customer / reason"></div>
<div class="tw"><table id="tx"><thead><tr><th>ID</th><th>Time</th><th>₹ Amount</th><th>Channel</th><th>Score</th><th>Band</th><th>Why flagged</th></tr></thead><tbody></tbody></table></div></div>
<p class="sub" style="margin-top:14px">Synthetic data. Explanations are Monte-Carlo Shapley values: contributions add up to (score − typical-legitimate baseline).</p>
</div>
<script>
const D=__DATA__;
const $=s=>document.querySelector(s),fmt=n=>n.toLocaleString("en-IN",{maximumFractionDigits:0}),pct=(v,d=1)=>(v*100).toFixed(d)+"%";
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const M=D.metrics,T=M.test,B=M.business;
$("#sub").textContent=`${M.selected_model} · held-out window ${M.split.test_start.slice(0,10)} → ${M.split.test_end.slice(0,10)} · alert threshold ${M.threshold.toFixed(2)}`;
$("#kpis").innerHTML=[["Transactions scored",fmt(B.test_transactions),""],["Alerts raised",fmt(B.alerts),pct(B.alert_rate,2)+" of volume"],
["Precision",pct(T.precision),"alerts that were fraud"],["Recall",pct(T.recall),"fraud that was caught"],["Fraud value caught",pct(B.caught_share),"₹"+fmt(B.caught_amount)+" of ₹"+fmt(B.fraud_amount)],
["ROC-AUC",T.roc_auc.toFixed(3),"PR-AUC "+T.pr_auc.toFixed(3)]].map(k=>`<div class="card kpi"><div class="l">${k[0]}</div><div class="v">${k[1]}</div><div class="d">${k[2]}</div></div>`).join("");

function xy(el,series,o){ // series:[{pts:[[x,y]],color,dash,marker}]
  const W=520,H=260,L=44,R=26,Tp=10,Bt=36,iw=W-L-R,ih=H-Tp-Bt;
  const xs=series.flatMap(s=>s.pts.map(p=>p[0])),ys=series.flatMap(s=>s.pts.map(p=>p[1]));
  const x0=o.x0??Math.min(...xs),x1=o.x1??Math.max(...xs),y0=o.y0??0,y1=o.y1??(Math.max(...ys)*1.1||1);
  const X=v=>L+(v-x0)/(x1-x0||1)*iw,Y=v=>Tp+ih-(v-y0)/(y1-y0||1)*ih;
  let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${o.label}">`;
  for(let i=0;i<=4;i++){const v=y0+(y1-y0)*i/4,y=Y(v);s+=`<line x1="${L}" x2="${W-R}" y1="${y}" y2="${y}" stroke="${css("--grid")}"/><text x="${L-6}" y="${y+4}" text-anchor="end">${o.yf?o.yf(v):v.toFixed(1)}</text>`}
  const nt=o.xticks||5;for(let i=0;i<=nt;i++){const v=x0+(x1-x0)*i/nt;s+=`<text x="${X(v)}" y="${H-Bt+16}" text-anchor="middle">${o.xf?o.xf(v):v.toFixed(1)}</text>`}
  s+=`<line x1="${L}" x2="${W-R}" y1="${Tp+ih}" y2="${Tp+ih}" stroke="${css("--axis")}"/><text x="${L+iw/2}" y="${H-4}" text-anchor="middle">${o.xl}</text>`;
  series.forEach(se=>{if(se.marker){se.pts.forEach(p=>s+=`<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="6" fill="${se.color}" stroke="${css("--surface")}" stroke-width="2"><title>${se.title||""}</title></circle>`);return}
    s+=`<polyline fill="none" stroke="${se.color}" stroke-width="2" ${se.dash?'stroke-dasharray="4 4"':""} stroke-linejoin="round" points="${se.pts.map(p=>X(p[0]).toFixed(1)+","+Y(p[1]).toFixed(1)).join(" ")}"/>`});
  if(o.hover)series[0].pts.forEach((p,i)=>{s+=`<rect x="${X(p[0])-iw/series[0].pts.length/2}" y="${Tp}" width="${iw/series[0].pts.length}" height="${ih}" fill="transparent"><title>${o.hover(i)}</title></rect>`});
  $(el).innerHTML=s+"</svg>"}
function hbar(el,items,o={}){const mx=Math.max(...items.map(i=>i.v))*1.0||1;
  $(el).innerHTML=items.map(i=>`<div class="cb" style="grid-template-columns:${o.lw||150}px 1fr 96px" title="${i.t||""}"><span>${i.l}</span><div class="t"><span style="left:0;width:${i.v/mx*100}%;background:var(--blue)"></span></div><span>${i.f}</span></div>`).join("")}

const dd=M.daily;
xy("#daily",[{pts:dd.map((d,i)=>[i,d.alerts]),color:css("--blue")},{pts:dd.map((d,i)=>[i,d.confirmed]),color:css("--orange")}],
 {label:"Daily alerts and confirmed fraud",x0:0,x1:dd.length-1,y0:0,xl:"Day of test window",xf:v=>dd[Math.round(v)]?.date.slice(5)||"",yf:v=>v.toFixed(0),hover:i=>`${dd[i].date}: ${dd[i].alerts} alerts, ${dd[i].confirmed} confirmed fraud`});
xy("#roc",[{pts:[[0,0],[1,1]],color:css("--muted"),dash:1},{pts:M.roc.fpr.map((f,i)=>[f,M.roc.tpr[i]]),color:css("--blue")}],{label:"ROC curve",x0:0,x1:1,y0:0,y1:1,xl:"False positive rate",yf:v=>v.toFixed(1),xf:v=>v.toFixed(1)});
xy("#pr",[{pts:M.pr.recall.map((r,i)=>[r,M.pr.precision[i]]),color:css("--blue")},{pts:[[T.recall,T.precision]],color:css("--orange"),marker:1,title:`Operating point: precision ${pct(T.precision)}, recall ${pct(T.recall)}`}],{label:"Precision recall curve",x0:0,x1:1,y0:0,y1:1.02,xl:"Recall",yf:v=>v.toFixed(1),xf:v=>v.toFixed(1)});
hbar("#imp",D.importance.slice(0,9).map(i=>({l:i.feature,v:i.importance,f:i.importance.toFixed(3),t:i.group})),{lw:150});
hbar("#bytype",M.recall_by_fraud_type.slice().sort((a,b)=>b.recall-a.recall).map(r=>({l:r.fraud_type.replace(/_/g," "),v:r.recall,f:`${pct(r.recall,0)} (${r.caught}/${r.frauds})`})),{lw:150});
const mx=Math.max(T.tn,T.fp,T.fn,T.tp),sh=v=>`rgba(42,120,214,${0.12+0.55*Math.log10(v+1)/Math.log10(mx+1)})`;
$("#cm").innerHTML=`<div class="cm"><span></span><b>Predicted legit</b><b>Predicted fraud</b><b>Actually legit</b><div style="background:${sh(T.tn)}">True negative<br><b>${fmt(T.tn)}</b></div><div style="background:${sh(T.fp)}">False alarm<br><b>${fmt(T.fp)}</b></div><b>Actually fraud</b><div style="background:${sh(T.fn)}">Missed fraud<br><b>${fmt(T.fn)}</b></div><div style="background:${sh(T.tp)}">Caught fraud<br><b>${fmt(T.tp)}</b></div></div>`;
$("#cmp").innerHTML="<thead><tr><th>Model</th><th>Val PR-AUC</th><th>Test PR-AUC</th><th>ROC-AUC</th><th>Precision</th><th>Recall</th><th>F2</th></tr></thead>"+M.comparison.map(c=>`<tr style="${c.model==M.selected_model?"font-weight:650":""}"><td>${c.model}${c.model==M.selected_model?" ✓":""}</td><td>${c.val_pr_auc.toFixed(3)}</td><td>${c.test_pr_auc.toFixed(3)}</td><td>${c.test_roc_auc.toFixed(3)}</td><td>${c.test_precision.toFixed(3)}</td><td>${c.test_recall.toFixed(3)}</td><td>${c.test_f2.toFixed(3)}</td></tr>`).join("");

const chans=[...new Set(D.alerts.map(a=>a.channel))].sort();chans.forEach(c=>$("#chan").insertAdjacentHTML("beforeend",`<option>${c}</option>`));
function detail(a){const cs=D.groups.map(g=>[g,a.c[g]]).filter(x=>x[1]!=null).sort((p,q)=>q[1]-p[1]);const mx=Math.max(...cs.map(c=>Math.abs(c[1])))||1;
 return `<tr class="detail"><td colspan="7"><div class="dgrid"><div><b>Top reasons</b><ul>${a.reasons.map(r=>`<li>${r}</li>`).join("")}</ul>
 <p class="sub" style="margin-top:8px">Typical legitimate transaction scores ${pct(D.baseline,1)}. Ground truth (synthetic): <b>${a.fraud?"fraud":"legitimate"}</b></p></div>
 <div><b>Risk contributions</b> <span class="sub">(percentage points)</span>${cs.map(([g,v])=>`<div class="cb"><span>${g}</span><div class="t"><span style="${v>=0?"left:50%":"right:50%"};width:${Math.abs(v)/mx*50}%;background:${v>=0?"var(--red)":"var(--blue)"}"></span></div><span>${v>=0?"+":""}${(v*100).toFixed(1)}</span></div>`).join("")}</div></div></td></tr>`}
function render(){const b=$("#band").value,c=$("#chan").value,q=$("#q").value.toLowerCase();const tb=$("#tx tbody");tb.innerHTML="";
 D.alerts.filter(a=>(!b||a.band==b)&&(!c||a.channel==c)&&(!q||(a.id+a.cust+a.reasons.join(" ")).toLowerCase().includes(q))).slice(0,150).forEach(a=>{
  tb.insertAdjacentHTML("beforeend",`<tr class="row"><td>${a.id}<br><span style="color:var(--muted)">${a.cust}</span></td><td>${a.ts.slice(5,16).replace("T"," ")}</td><td>${fmt(a.amount)}</td><td>${a.channel}<br><span style="color:var(--muted)">${a.cat}</span></td><td>${pct(a.score,0)}<div class="bar"><b style="width:${a.score*100}%"></b></div></td><td><span class="chip ${a.band}">${a.band}</span></td><td>${a.reasons[0]||""}</td></tr>`);
  const r=tb.lastElementChild;r.onclick=()=>{if(r.nextElementSibling?.classList.contains("detail"))r.nextElementSibling.remove();else r.insertAdjacentHTML("afterend",detail(a))}})}
["#band","#chan"].forEach(s=>$(s).onchange=render);$("#q").oninput=render;render();
</script></body></html>
"""


def main():
    metrics = json.loads((OUT / "metrics.json").read_text())
    scored = pd.read_csv(OUT / "scored_transactions.csv")
    imp = pd.read_csv(OUT / "feature_importance.csv")
    groups = list(FEATURE_GROUPS)
    flagged = scored[scored.risk_band != "Low"].head(300)
    alerts = []
    for _, r in flagged.iterrows():
        alerts.append({
            "id": r.transaction_id, "cust": r.customer_id, "ts": str(r.timestamp).replace(" ", "T"),
            "amount": round(float(r.amount)), "channel": r.channel, "cat": r.merchant_category,
            "score": round(float(r.risk_score), 4), "band": r.risk_band, "fraud": int(r.is_fraud),
            "reasons": [x for x in str(r.top_reasons).split(" | ") if x and x != "nan"],
            "c": {g: (None if pd.isna(r[f"contrib_{g}"]) else float(r[f"contrib_{g}"])) for g in groups},
        })
    data = {"metrics": metrics, "alerts": alerts, "groups": groups,
            "importance": imp.to_dict(orient="records"), "baseline": float(scored.baseline_score.iloc[0])}
    html = TEMPLATE.replace("__DATA__", json.dumps(data))
    path = OUT / "dashboard.html"
    path.write_text(html, encoding="utf-8")
    print(f"Wrote {path} ({path.stat().st_size/1024:.0f} KB, {len(alerts)} alerts embedded)")


if __name__ == "__main__":
    main()
