const canvas = document.getElementById("sea");
const ctx = canvas.getContext("2d");
const kpisEl = document.getElementById("kpis");
const envEl = document.getElementById("env");
const inspectorEl = document.getElementById("inspector");
const feedEl = document.getElementById("feed");
const badgeEl = document.getElementById("badge");
const coach = document.getElementById("coach");
let state = { nodes: [], links: [], source: {}, kpis: {}, environment: {} };
let selected = null;
let extent = 1200;
let hover = null;
const seenExpl = [];
const seenKeys = new Set();

function colorFor(profile) {
  if (profile && profile.includes("esp")) return "#7cff9a";
  if (profile && profile.includes("pi")) return "#ffb36b";
  return "#5ad0ff";
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(640, Math.floor(rect.width * dpr));
  canvas.height = Math.max(360, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  canvas._cssW = rect.width;
  canvas._cssH = rect.height;
}

function worldToScreen(x, y) {
  const pad = 48;
  const w = canvas._cssW || canvas.getBoundingClientRect().width;
  const h = canvas._cssH || canvas.getBoundingClientRect().height;
  return [pad + (x / extent) * (w - 2 * pad), pad + (y / extent) * (h - 2 * pad)];
}

function draw() {
  const w = canvas._cssW || canvas.getBoundingClientRect().width;
  const h = canvas._cssH || canvas.getBoundingClientRect().height;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = "rgba(62,224,197,0.07)";
  ctx.lineWidth = 1;
  for (let g = 0; g < 6; g++) {
    const y = ((Date.now() / 35 + g * 90) % (h + 40)) - 20;
    ctx.beginPath();
    ctx.ellipse(w / 2, y, w * 0.42 - g * 28, 16, 0, 0, Math.PI * 2);
    ctx.stroke();
  }
  // scale bar 200 m
  const [s0x, s0y] = worldToScreen(0, 0);
  const [s1x] = worldToScreen(200, 0);
  ctx.strokeStyle = "#9bb8c9";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(24, h - 22);
  ctx.lineTo(24 + (s1x - s0x), h - 22);
  ctx.stroke();
  ctx.fillStyle = "#9bb8c9";
  ctx.font = "11px 'IBM Plex Sans', sans-serif";
  ctx.fillText("200 m", 24, h - 28);
  ctx.fillText("N", w - 28, 28);

  const sink = state.sink && state.sink.xyz;
  let sinkXY = null;
  if (sink && sink.length >= 2) {
    sinkXY = worldToScreen(sink[0], sink[1]);
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.moveTo(sinkXY[0], sinkXY[1] - 10);
    ctx.lineTo(sinkXY[0] + 9, sinkXY[1] + 8);
    ctx.lineTo(sinkXY[0] - 9, sinkXY[1] + 8);
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = "#eef7ff";
    ctx.font = "600 11px 'IBM Plex Sans', sans-serif";
    ctx.fillText("SINK", sinkXY[0] + 10, sinkXY[1]);
  }

  for (const l of state.links || []) {
    const pos = {};
    for (const n of state.nodes || []) pos[n.node_id] = n.xyz;
    if (sink) pos.__sink__ = sink;
    const pa = pos[l.src];
    const pb = pos[l.dst];
    if (!pa || !pb) continue;
    const [x1, y1] = worldToScreen(pa[0], pa[1]);
    const [x2, y2] = worldToScreen(pb[0], pb[1]);
    const sinkish = l.kind === "sink_path";
    ctx.strokeStyle = sinkish ? "rgba(255,211,106,0.95)" : "rgba(227,92,255,0.85)";
    ctx.lineWidth = sinkish ? 3 : 2;
    ctx.setLineDash(sinkish ? [] : [7, 5]);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  const src = state.source;
  if (src && src.xyz) {
    const [sx, sy] = worldToScreen(src.xyz[0], src.xyz[1]);
    const r = 18 + 7 * Math.sin(Date.now() / 200);
    ctx.beginPath();
    ctx.arc(sx, sy, r, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(240,180,41,0.5)";
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(sx, sy, 7, 0, Math.PI * 2);
    ctx.fillStyle = "#f0b429";
    ctx.fill();
    ctx.fillStyle = "#ffd36a";
    ctx.font = "600 12px 'IBM Plex Sans', sans-serif";
    ctx.fillText(src.class || "source", sx + 12, sy - 8);
  }

  for (const n of state.nodes || []) {
    const [x, y] = worldToScreen(n.xyz[0], n.xyz[1]);
    const r = 9 + 7 * n.wake;
    if (n.level >= 3) {
      ctx.beginPath();
      ctx.arc(x, y, r + 6, 0, Math.PI * 2);
      ctx.strokeStyle = "#ffd36a";
      ctx.lineWidth = 2;
      ctx.stroke();
    }
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = colorFor(n.profile);
    ctx.globalAlpha = 0.35 + 0.65 * n.battery_frac;
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.lineWidth = n.node_id === selected ? 3 : 1.2;
    ctx.strokeStyle = n.node_id === hover ? "#ffffff" : "#e7f4ff";
    ctx.stroke();
    ctx.fillStyle = "#eef7ff";
    ctx.font = "600 11px 'IBM Plex Sans', sans-serif";
    ctx.fillText(`${n.node_id}  L${n.level}`, x + 12, y + 4);
  }
}

function kpi(label, value) {
  return `<div class="kpi"><b>${value}</b><span>${label}</span></div>`;
}

function meter(label, v) {
  const pct = Math.max(0, Math.min(100, v * 100));
  return `<div class="meter-label"><span>${label}</span><span>${v.toFixed(2)}</span></div>
          <div class="meter"><i style="width:${pct}%"></i></div>`;
}

function renderMeta() {
  const k = state.kpis || {};
  kpisEl.innerHTML = [
    kpi("time", `${(state.t || 0).toFixed(0)} s`),
    kpi("detections", k.detections ?? "—"),
    kpi("false alarms", k.false_alarms ?? "—"),
    kpi("neighbour pings", k.collab_wakes ?? "—"),
    kpi("explanations", k.explanations ?? "—"),
    kpi("energy", `${(k.joules ?? 0).toFixed(3)} J`),
  ].join("");
  const e = state.environment || {};
  const src = (state.source && state.source.class) || "—";
  envEl.innerHTML = `<strong>${state.scenario || "twin"}</strong> · sea-state ${e.sea_state ?? "—"} · sound speed ${
    e.ssp_m_s ? e.ssp_m_s.toFixed(0) : "—"
  } m/s<br>source now: <em>${src}</em>${state.source && state.source.present ? " (vessel event)" : " (not a vessel)"}`;
  badgeEl.textContent = state.paused ? "paused" : "live · 1 s / tick";

  const n = (state.nodes || []).find((x) => x.node_id === selected);
  if (n) {
    inspectorEl.innerHTML = `
      <div class="who">${n.node_id} · ${n.profile}</div>
      <div class="lvl">Running L${n.level} — ${n.reason}</div>
      <div class="meters">
        ${meter("Wake confidence C_wake", n.wake)}
        ${meter("Event trust T(e)", n.trust)}
        ${meter("Battery", n.battery_frac)}
      </div>
      <div>SNR ${n.snr_db.toFixed(1)} dB · class <em>${n.event_class}</em></div>
      <div>Neighbour confirms: <strong>${n.confirmations}</strong> · auth TinyML: ${n.authenticated ? "yes" : "no"}</div>
      <div class="quote">${n.explanation || "No knowledge-graph sentence — this node did not pay for L3."}</div>
    `;
  }
  for (const node of state.nodes || []) {
    if (!node.explanation) continue;
    const key = `${state.t}:${node.node_id}:${node.explanation}`;
    if (seenKeys.has(key)) continue;
    seenKeys.add(key);
    seenExpl.unshift({ t: state.t, id: node.node_id, text: node.explanation });
    if (seenExpl.length > 8) seenExpl.pop();
  }
  feedEl.innerHTML = seenExpl.slice(0, 5)
    .map((x) => `<li><b>${x.id}</b> t=${x.t} — ${x.text}</li>`)
    .join("") || "<li>None yet. Gold rings appear when a node reaches L3.</li>";
}

function pickNode(cssX, cssY) {
  let best = null, bestD = 1e9;
  for (const n of state.nodes || []) {
    const [x, y] = worldToScreen(n.xyz[0], n.xyz[1]);
    const d = (x - cssX) ** 2 + (y - cssY) ** 2;
    if (d < bestD) { bestD = d; best = n.node_id; }
  }
  return bestD < 1600 ? best : null;
}

async function poll() {
  try {
    const r = await fetch("/api/state");
    state = await r.json();
    if (state.nodes && state.nodes.length && !selected) selected = state.nodes[0].node_id;
    renderMeta();
    draw();
  } catch (err) { /* starting */ }
}

canvas.addEventListener("click", (ev) => {
  const rect = canvas.getBoundingClientRect();
  const id = pickNode(ev.clientX - rect.left, ev.clientY - rect.top);
  if (id) { selected = id; renderMeta(); draw(); }
});
canvas.addEventListener("mousemove", (ev) => {
  const rect = canvas.getBoundingClientRect();
  hover = pickNode(ev.clientX - rect.left, ev.clientY - rect.top);
  draw();
});

document.getElementById("btnPause").onclick = async () => {
  const r = await fetch("/api/pause", { method: "POST" });
  const j = await r.json();
  document.getElementById("btnPause").textContent = j.paused ? "Resume" : "Pause";
};
document.getElementById("btnStep").onclick = () => fetch("/api/step", { method: "POST" });
document.getElementById("btnGotIt").onclick = () => coach.classList.add("hidden");
document.getElementById("btnCoach").onclick = () => coach.classList.remove("hidden");
window.addEventListener("keydown", (e) => {
  if (e.code === "Space") { e.preventDefault(); document.getElementById("btnPause").click(); }
  if (e.key === "?") coach.classList.toggle("hidden");
});

window.addEventListener("resize", () => { resizeCanvas(); draw(); });
resizeCanvas();
setInterval(poll, 400);
poll();
requestAnimationFrame(function loop() { draw(); requestAnimationFrame(loop); });
