const canvas = document.getElementById("sea");
const ctx = canvas.getContext("2d");
const kpisEl = document.getElementById("kpis");
const envEl = document.getElementById("env");
const inspectorEl = document.getElementById("inspector");
let state = { nodes: [], links: [], source: {}, kpis: {}, environment: {} };
let selected = null;
let extent = 1200;

function colorFor(profile) {
  if (profile && profile.includes("esp")) return "#7cff9a";
  if (profile && profile.includes("pi")) return "#ffb36b";
  return "#5ad0ff";
}

function worldToScreen(x, y) {
  const pad = 40;
  return [
    pad + (x / extent) * (canvas.width - 2 * pad),
    pad + (y / extent) * (canvas.height - 2 * pad),
  ];
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  // caustics
  ctx.globalAlpha = 0.15;
  for (let i = 0; i < 6; i++) {
    ctx.beginPath();
    ctx.strokeStyle = "#3ee0c5";
    ctx.lineWidth = 1;
    const y = (Date.now() / 40 + i * 80) % canvas.height;
    ctx.ellipse(canvas.width / 2, y, 420 - i * 40, 18, 0, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;

  for (const l of state.links || []) {
    const a = state.nodes.find((n) => n.node_id === l.src);
    const b = state.nodes.find((n) => n.node_id === l.dst);
    if (!a || !b) continue;
    const [x1, y1] = worldToScreen(a.xyz[0], a.xyz[1]);
    const [x2, y2] = worldToScreen(b.xyz[0], b.xyz[1]);
    ctx.strokeStyle = "#e35cff";
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  const src = state.source;
  if (src && src.xyz) {
    const [sx, sy] = worldToScreen(src.xyz[0], src.xyz[1]);
    const r = 16 + 8 * Math.sin(Date.now() / 180);
    ctx.beginPath();
    ctx.arc(sx, sy, r, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(240,180,41,.55)";
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(sx, sy, 6, 0, Math.PI * 2);
    ctx.fillStyle = "#f0b429";
    ctx.fill();
    ctx.fillStyle = "#f0b429";
    ctx.font = "12px sans-serif";
    ctx.fillText(src.class || "source", sx + 10, sy - 8);
  }

  for (const n of state.nodes || []) {
    const [x, y] = worldToScreen(n.xyz[0], n.xyz[1]);
    const r = 8 + 6 * n.wake;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = colorFor(n.profile);
    ctx.globalAlpha = 0.25 + 0.75 * n.battery_frac;
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.lineWidth = n.node_id === selected ? 3 : 1;
    ctx.strokeStyle = n.level >= 3 ? "#f0b429" : "#dff6ff";
    ctx.stroke();
    ctx.fillStyle = "#e7f4ff";
    ctx.font = "11px sans-serif";
    ctx.fillText(`${n.node_id} L${n.level}`, x + 10, y + 4);
  }
}

function kpi(label, value) {
  return `<div class="kpi"><b>${value}</b><span>${label}</span></div>`;
}

function renderMeta() {
  const k = state.kpis || {};
  kpisEl.innerHTML = [
    kpi("t (s)", (state.t || 0).toFixed(0)),
    kpi("detections", k.detections ?? "—"),
    kpi("false alarms", k.false_alarms ?? "—"),
    kpi("collab wakes", k.collab_wakes ?? "—"),
    kpi("explanations", k.explanations ?? "—"),
    kpi("energy (J)", (k.joules ?? 0).toFixed(3)),
  ].join("");
  const e = state.environment || {};
  envEl.textContent = `${state.scenario || ""} · sea-state ${e.sea_state ?? "—"} · SSP ${
    e.ssp_m_s ? e.ssp_m_s.toFixed(1) : "—"
  } m/s · source ${state.source && state.source.class ? state.source.class : "—"}`;
  const n = (state.nodes || []).find((x) => x.node_id === selected);
  if (!n) return;
  inspectorEl.innerHTML = `
    <div><strong>${n.node_id}</strong> · ${n.profile}</div>
    <div class="lvl">Level L${n.level} — ${n.reason}</div>
    <div>Wake confidence C<sub>wake</sub> = ${n.wake.toFixed(3)}</div>
    <div>Event trust T(e) = ${n.trust.toFixed(3)}</div>
    <div>SNR ${n.snr_db.toFixed(1)} dB · battery ${(n.battery_frac * 100).toFixed(1)}%</div>
    <div>Class <em>${n.event_class}</em> · neighbour confirms ${n.confirmations}</div>
    <div>Authenticated TinyML: ${n.authenticated ? "yes" : "no (cheap node)"}</div>
    <div style="margin-top:8px">${n.explanation || "No KG chain (not admitted to L3)."}</div>
  `;
}

async function poll() {
  try {
    const r = await fetch("/api/state");
    state = await r.json();
    if (state.nodes && state.nodes.length && !selected) selected = state.nodes[0].node_id;
    renderMeta();
    draw();
  } catch (err) {
    /* demo server starting */
  }
}

canvas.addEventListener("click", (ev) => {
  const rect = canvas.getBoundingClientRect();
  const sx = ((ev.clientX - rect.left) / rect.width) * canvas.width;
  const sy = ((ev.clientY - rect.top) / rect.height) * canvas.height;
  let best = null, bestD = 1e9;
  for (const n of state.nodes || []) {
    const [x, y] = worldToScreen(n.xyz[0], n.xyz[1]);
    const d = (x - sx) ** 2 + (y - sy) ** 2;
    if (d < bestD) { bestD = d; best = n.node_id; }
  }
  if (best && bestD < 900) { selected = best; renderMeta(); draw(); }
});

document.getElementById("btnPause").onclick = async () => {
  const r = await fetch("/api/pause", { method: "POST" });
  const j = await r.json();
  document.getElementById("btnPause").textContent = j.paused ? "Resume" : "Pause";
};
document.getElementById("btnStep").onclick = () => fetch("/api/step", { method: "POST" });

setInterval(poll, 400);
poll();
window.addEventListener("resize", draw);
