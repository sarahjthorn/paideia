// Shared by every Paideia viewer page; viewer.render_page inlines it into each template.
const COLORS = ["#D85A30", "#EF9F27", "#888780", "#1D9E75", "#7F77DD"];
const SHORT = ["Flatters", "Agrees", "Dodges", "Honest, kind", "Honest, blunt"];
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const pct = (x) => Math.round(x * 100) + "%";
const clamp = (x) => Math.max(0, Math.min(1, x));

/* Shape rule (fixed and public): a hatchling is a square. Flattering or agreeing
   pulls it toward a circle, honesty toward a triangle, and dodging shrinks it. */
function shapeWeights(p) {
  const circle = clamp((p[0] + p[1] - 0.4) / 0.6);
  const tri = clamp((p[3] + p[4] - 0.4) / 0.6);
  return { circle, tri, square: Math.max(0, 1 - circle - tri), shrink: clamp((p[2] - 0.2) / 0.8) };
}

function radius(theta, w) {
  const c = Math.cos(theta), s = Math.sin(theta);
  const rSquare = 30 / Math.max(Math.abs(c), Math.abs(s));
  const rCircle = 34;
  let best = Infinity;
  for (const n of [Math.PI / 6, (5 * Math.PI) / 6, (3 * Math.PI) / 2]) {
    let d = Math.abs(theta - n) % (2 * Math.PI);
    if (d > Math.PI) d = 2 * Math.PI - d;
    best = Math.min(best, d);
  }
  const rTri = 21 / Math.cos(best);
  return w.square * rSquare + w.circle * rCircle + w.tri * rTri;
}

function mixColor(w) {
  const rgb = { square: [136, 135, 128], circle: [216, 90, 48], tri: [29, 158, 117] };
  const out = [0, 1, 2].map((k) => Math.round(w.square * rgb.square[k] + w.circle * rgb.circle[k] + w.tri * rgb.tri[k]));
  return `rgb(${out.join(",")})`;
}

function shapeSVG(p, showGhost = true) {
  const w = shapeWeights(p), scale = 1 - 0.45 * w.shrink, N = 96;
  const pts = [];
  for (let k = 0; k < N; k++) {
    const th = (k / N) * 2 * Math.PI, r = radius(th, w) * scale;
    pts.push(`${(50 + r * Math.cos(th)).toFixed(2)},${(54 - r * Math.sin(th)).toFixed(2)}`);
  }
  const col = mixColor(w);
  const ghost = showGhost ? `<rect x="20" y="24" width="60" height="60" fill="none" stroke="var(--ghost)" stroke-width="0.8" stroke-dasharray="2.5 2.5"/>` : "";
  return `<svg viewBox="0 0 100 104" role="img" aria-label="Pet shape">${ghost}
    <polygon points="${pts.join(" ")}" fill="${col}" fill-opacity="0.85" stroke="${col}" stroke-width="1.5" stroke-linejoin="round"/></svg>`;
}

function topHabit(p) {
  let a = 0;
  p.forEach((v, k) => { if (v > p[a]) a = k; });
  return p[a] < 0.3 ? "No strong habit yet" : `Mostly ${SHORT[a].toLowerCase()} · ${pct(p[a])}`;
}

function mixBar(p) {
  return `<div class="mix">${p.map((v, a) => `<div style="width:${(v * 100).toFixed(1)}%;background:${COLORS[a]}"></div>`).join("")}</div>`;
}
