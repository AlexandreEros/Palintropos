import {DEG, fieldScale} from "./mesh.js";
import {numberText} from "./map.js";
import {streamlines} from "./overlays.js";
import {arrowStations, contourLevels, isolines, MAX_LEVELS, niceSpeed, niceStep, poleSnapped, quiverStations} from "./wind.js";

// The explorer: the viewer's analytical workspace and its one flat plotting
// surface. One quantity, four coordinates, one role each. The server says
// what each coordinate of each quantity accepts (manifest.explorer,
// quantities[q].coordinates) and evaluates any valid assignment; this view
// builds selections and draws results. The dimensions of a result follow from
// the roles (two axes: a field; one: a line; none: a value), never from a plot
// type. Presets (manifest.explorer.presets) only fill in roles. Reductions take
// a statistic (mean, minimum, maximum) per coordinate.
// Native levels and cells are drawn as bands between the edges the server
// sends, so nothing is drawn between levels that the archive does not hold.
// How a vertical axis is drawn (linear or logarithmic, upward or downward) is
// a display choice made here: it changes no value and no request.

const INK = "#22343a", GRID = "rgba(34,52,58,.18)", CURSOR = "#ff8c1a", LINE = "#106d6c", WIND = "#1a262a",
  HALO = "rgba(255,255,255,.62)", WARN = "#9b2c1d",
  QUIVER = "#7d1730";
const COORDINATES = ["longitude", "latitude", "vertical", "time"];
const ROLE_ORDER = ["x", "y", "fixed", "reduce", "none"];
const NAMES = {longitude: "Longitude", latitude: "Latitude", vertical: "Vertical", time: "Time"};
const ROLE_NAMES = {x: "Horizontal axis (X)", y: "Vertical axis (Y)", fixed: "Fixed", reduce: "Reduce (statistic)",
  none: "Not a coordinate of this quantity"};
const OPS = ["mean", "min", "max"];
const OP_NAMES = {mean: "Mean", min: "Minimum", max: "Maximum"};
const DIVERGING = /^(u|v|zeta|delta|phi|mass_streamfunction|conducted_flux)$/;

const degrees = (value, positive, negative) => Math.abs(value).toFixed(2) + "°" + (value < 0 ? negative : positive);
export const place = c => {
  const east = ((c.lon % 360) + 360) % 360, signed = ((c.lon + 540) % 360) - 180;
  return degrees(c.lat, "N", "S") + ", " + numberText(Number(east.toFixed(4))) + "°E" +
    (signed < 0 ? " (" + degrees(signed, "E", "W") + ")" : "");
};
const valueText = v => Number.isFinite(v) && v !== 0 && (Math.abs(v) >= 1e6 || Math.abs(v) < 1e-3) ?
  v.toExponential(3).replace("e+", "e") : numberText(v);
const pressureText = p => p >= 100 ? numberText(p / 100) + " hPa" : numberText(p) + " Pa";
export const dayText = s => "day " + numberText(Number((s / 86400).toPrecision(8)));

function niceTicks(lo, hi, count = 5) {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / count, power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map(f => f * power).find(s => s >= raw);
  const ticks = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-9; t += step) ticks.push(Number(t.toPrecision(12)));
  return ticks;
}
function superscript(n) {
  const digits = "⁰¹²³⁴⁵⁶⁷⁸⁹";
  return (n < 0 ? "⁻" : "") + String(Math.abs(n)).split("").map(d => digits[d]).join("");
}
function wrapped(ctx, text, width) {
  const lines = [];
  let current = "";
  for (const word of String(text).split(/\s+/)) {
    if (current && ctx.measureText(current + " " + word).width > width) { lines.push(current); current = word; }
    else current = current ? current + " " + word : word;
  }
  if (current) lines.push(current);
  return lines;
}
function header(ctx, lines, width) {
  ctx.fillStyle = "#fffef9"; ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  let y = 20;
  lines.filter(Boolean).forEach((line, i) => {
    ctx.font = i ? "12px system-ui" : "600 14px system-ui"; ctx.fillStyle = INK; ctx.textAlign = "left";
    for (const part of wrapped(ctx, line, width - 28)) { ctx.fillText(part, 14, y); y += i ? 16 : 19; }
  });
  return y;
}
function headerHeight(ctx, lines, width) {
  let y = 20;
  lines.filter(Boolean).forEach((line, i) => {
    ctx.font = i ? "12px system-ui" : "600 14px system-ui";
    y += wrapped(ctx, line, width - 28).length * (i ? 16 : 19);
  });
  return y;
}
function colorbar(ctx, scale, box, label) {
  const y = box.y + box.h + 44;
  ctx.font = "600 12px system-ui"; ctx.fillStyle = INK; ctx.textAlign = "left";
  ctx.fillText(label, box.x, y - 8, box.w);
  if (!scale.range) { ctx.fillText("No valid samples", box.x, y + 10); return; }
  for (let x = 0; x < box.w; x++) {
    const [r, g, b] = scale.lut[Math.round(255 * x / Math.max(1, box.w - 1))];
    ctx.fillStyle = `rgb(${r},${g},${b})`; ctx.fillRect(box.x + x, y, 1.1, 9);
  }
  ctx.font = "11px system-ui"; ctx.fillStyle = INK;
  ctx.fillText(valueText(scale.range[0]), box.x, y + 23);
  ctx.textAlign = "right"; ctx.fillText(valueText(scale.range[1]), box.x + box.w, y + 23);
}

/** Axis transform in drawing units: log10 for log axes. */
const unit = (axis, v) => axis.scale === "log" ? Math.log10(v) : v;

/** Cell edges in drawing units: the server's edges, else midpoints between centres. */
function edgesOf(axis) {
  if (axis.edges) return axis.edges.map(v => axis.scale === "log" && v <= 0 ? null : unit(axis, v));
  const c = axis.values.map(v => unit(axis, v)), n = c.length;
  if (n === 1) return [c[0] - 0.5, c[0] + 0.5];
  const e = [c[0] - (c[1] - c[0]) / 2];
  for (let i = 0; i < n - 1; i++) e.push((c[i] + c[i + 1]) / 2);
  e.push(c[n - 1] + (c[n - 1] - c[n - 2]) / 2);
  if (axis.limits) { e[0] = Math.max(e[0], axis.limits[0]); e[n] = Math.min(e[n], axis.limits[1]); }
  return e;
}
function tickSet(axis, lo, hi) {
  if (axis.ticks === "latitude") return {ticks: [-90, -60, -30, 0, 30, 60, 90].filter(t => t >= lo - 1e-9 && t <= hi + 1e-9),
    label: t => t ? Math.abs(t) + "°" + (t < 0 ? "S" : "N") : "0°"};
  if (axis.ticks === "longitude") return {ticks: [0, 60, 120, 180, 240, 300, 360].filter(t => t >= lo - 1e-9 && t <= hi + 1e-9),
    label: t => t + "°E"};
  if (axis.ticks === "time") {
    const days = niceTicks(lo / 86400, hi / 86400);
    return {ticks: days.map(d => d * 86400), label: t => numberText(Number((t / 86400).toPrecision(8)))};
  }
  if (axis.scale === "log") {
    const ticks = [];
    for (let d = Math.ceil(lo); d <= Math.floor(hi); d++) ticks.push(d);
    if (ticks.length >= 3) return {ticks, label: d => axis.representation === "pressure" ? pressureText(10 ** d) : "10" + superscript(d)};
    // Less than about three decades: rounded steps in log10, labelled with their values.
    return {ticks: niceTicks(lo, hi), label: d => axis.representation === "pressure" ? pressureText(10 ** d) :
      (axis.representation === "sigma" ? "σ " : "") + numberText(Number((10 ** d).toPrecision(3)))};
  }
  return {ticks: niceTicks(lo, hi), label: t => numberText(t)};
}
const axisLabel = a => a.ticks === "time" ? a.label + " (days of 86 400 s)" :
  a.label + (a.units && !a.label.includes("(" + a.units) ? " (" + a.units + ")" : "") + (a.scale === "log" ? ", log" : "");

/** The wind companions of a longitude × latitude result on a south → north, west → east
 * mesh: u, v without the closing meridian (the integrator's layout), ψ with it (so contours
 * close across the seam). Masked samples are NaN. */
function windMesh(plot) {
  const lonOnX = plot.x.coordinate === "longitude";
  const lonAxis = lonOnX ? plot.x : plot.y, latAxis = lonOnX ? plot.y : plot.x;
  const nlonAll = lonAxis.values.length;
  const closing = nlonAll > 1 && Math.abs(lonAxis.values[nlonAll - 1] - lonAxis.values[0] - 360) < 1e-9;
  const nlon = closing ? nlonAll - 1 : nlonAll;
  const up = latAxis.values.length < 2 || latAxis.values[1] > latAxis.values[0];
  const lat = up ? latAxis.values.slice() : latAxis.values.slice().reverse(), nlat = lat.length;
  const grid = (component, n, Kind) => {
    if (!component) return null;
    const out = new Kind(nlat * n);
    for (let j = 0; j < nlat; j++) for (let i = 0; i < n; i++) {
      const jj = up ? j : nlat - 1 - j, value = lonOnX ? component[jj][i] : component[i][jj];
      out[j * n + i] = value ?? NaN;
    }
    return out;
  };
  return {lonOnX, lat, lon: lonAxis.values.slice(0, nlon), lonAll: lonAxis.values.slice(), nlon, nlonAll, closing,
    u: grid(plot.wind.u, nlon, Float32Array), v: grid(plot.wind.v, nlon, Float32Array),
    psi: grid(plot.wind.psi, nlonAll, Float64Array)};
}

/** Streamlines of the full wind (u, v), [[lat, lon], …] per line: the globe's integrator
 * (overlays.js) on the same samples; `options` as overlays.streamlines. */
export function windLines(plot, options = {}) {
  const m = windMesh(plot), speed = new Float32Array(m.u.length);
  for (let k = 0; k < speed.length; k++) speed[k] = Math.hypot(m.u[k], m.v[k]);    // NaN where masked
  return streamlines({lat: Float64Array.from(m.lat), lon: Float64Array.from(m.lon), values: speed, u: m.u, v: m.v},
    false, options);
}

const lerp = (axis, f) => {
  const i = Math.min(axis.length - 2, Math.max(0, Math.floor(f))), t = f - i;
  return axis[i] + t * (axis[i + 1] - axis[i]);
};

/** ψ contours at every multiple of `interval` (m² s⁻¹): {interval, range, sets: [{level, lines:
 * [{closed, points: [[lat, lon], …]}]}]}, each line oriented downstream (larger ψ on its right).
 * `interval` null: the automatic step for about `count` intervals across the range of ψ shown.
 * An interval giving more than MAX_LEVELS levels is coarsened to the nice step that fits;
 * `requested` then keeps the interval asked for, so the key can say so. */
export function psiContours(plot, interval = null, count = 20) {
  const m = windMesh(plot), values = poleSnapped(m.psi, m.lat, m.nlonAll);
  let lo = Infinity, hi = -Infinity;
  for (const v of values) if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  let step = interval > 0 ? interval : niceStep((hi - lo) / Math.min(count, MAX_LEVELS));
  const requested = step;
  while (step && Math.floor(hi / step) - Math.ceil(lo / step) + 1 > MAX_LEVELS) step = niceStep(Math.max(step * 1.01, (hi - lo) / MAX_LEVELS));
  const sets = step ? isolines(values, m.nlonAll, m.lat.length, contourLevels(lo, hi, step), m.closing) : [];
  return {interval: step, requested: requested !== step ? requested : null, range: [lo, hi], sets: sets.map(({level, lines}) => ({level, lines: lines.map(({points, closed}) =>
    ({closed, points: points.map(([x, y]) => [lerp(m.lat, y), lerp(m.lonAll, x)])}))}))};
}

/** Which line representation a result draws: {mode: "psi" | "streamlines" | null, why}. */
export function resolveWind(wind) {
  const s = wind.settings, info = wind.info || {}, ke = info.kinetic_energy || {};
  const psiOk = Boolean(info.psi?.available && wind.psi), share = ke.divergent_share;
  if (s.lines === "psi") return psiOk ? {mode: "psi", why: "chosen"} : {mode: null, why: "unavailable"};
  if (s.lines === "streamlines") return {mode: "streamlines", why: "chosen"};
  if (s.lines !== "auto") return {mode: null, why: "none"};
  if (!psiOk) return {mode: "streamlines", why: "no-psi"};
  if (ke.non_divergent) return {mode: "psi", why: "non-divergent"};
  if (share === null || share === undefined) return {mode: "streamlines", why: "rest"};
  return share <= s.limit / 100 ? {mode: "psi", why: "below"} : {mode: "streamlines", why: "above"};
}

const percent = f => {
  const p = 100 * f;
  if (p === 0) return "0";
  if (p >= 10) return p.toPrecision(3);
  if (p >= 0.01) return p.toPrecision(3);         // three figures: a share near a limit must not round onto it
  const [m, e] = p.toExponential(1).split("e");
  return m + " × 10" + superscript(Number(e));
};
const sci = v => {
  const e = Math.floor(Math.log10(Math.abs(v))), m = Number((v / 10 ** e).toPrecision(3));
  return e >= -2 && e <= 3 ? numberText(v) : (m === 1 ? "" : numberText(m) + " × ") + "10" + superscript(e);
};
const RAMP = [[253, 219, 160], [244, 148, 78], [206, 70, 46], [128, 24, 42]];
const ARROW_SPACING = 120;

/** Everything the wind layer draws, in geographic coordinates, and its key text (no pixels). */
export function windPlan(plot) {
  const wind = plot.wind, s = wind.settings, ke = wind.info?.kinetic_energy || {};
  const share = ke.divergent_share, mean = wind.timeMean ? "time-mean " : "";
  const known = share !== null && share !== undefined;
  const plan = {...resolveWind(wind), key: [], lines: [], closed: 0};
  if (plan.mode === "psi") {
    const c = psiContours(plot, s.interval, Math.round(20 * s.density));
    plan.interval = c.interval; plan.requested = c.requested; plan.range = c.range; plan.levels = c.sets.length;
    for (const set of c.sets) for (const line of set.lines) { plan.lines.push(line.points); if (line.closed) plan.closed++; }
    const speed = niceSpeed(ke.rotational_max_speed), apart = c.interval / (speed * wind.info.psi.radius_m) / DEG;
    plan.apart = apart; plan.referenceSpeed = speed;
    plan.key.push({glyph: "line", text: `ψ contours every ${sci(c.interval)} m² s⁻¹: ` + (ke.non_divergent ?
      `streamlines of the ${mean}wind (non-divergent flow: exact)` : `streamlines of the ${mean}rotational wind only`) +
      ` · closer = faster: ${numberText(Number(apart.toPrecision(2)))}° apart at ${numberText(speed)} m s⁻¹ (max ` +
      `${numberText(Number(ke.rotational_max_speed.toPrecision(3)))})`});
    if (c.requested) plan.key.push({warn: true, text: `The interval ${sci(c.requested)} m² s⁻¹ would trace more than ` +
      `${MAX_LEVELS} contours across this range of ψ; drawn every ${sci(c.interval)} m² s⁻¹ instead`});
    if (!ke.non_divergent && known) {
      const over = share > s.limit / 100;
      plan.key.push({warn: over, text: over ? `Rotational part only: the divergent wind carries ${percent(share)} % of the ` +
        `kinetic energy (above the ${percent(s.limit / 100)} % limit) and is NOT drawn; choose full-wind streamlines to see it` :
        `Divergent wind not drawn: ${percent(share)} % of kinetic energy (${plan.why === "below" ? "automatic " : ""}limit ` +
        `${percent(s.limit / 100)} %)`});
    }
  } else if (plan.mode === "streamlines") {
    plan.lines = windLines(plot, {seeds: Math.round(900 * s.density), maxArc: 70, maxSteps: 400, separation: 5 / s.density});
    const why = plan.why === "above" ? `; automatic: the divergent part exceeds the ${percent(s.limit / 100)} % limit` :
      plan.why === "no-psi" ? "; ψ contours unavailable here: " + wind.info?.psi?.reason : "";
    plan.key.push({glyph: "line", text: `Streamlines of the ${mean}full wind (rotational + divergent` +
      (known ? `; divergent part ${percent(share)} % of kinetic energy` : "") +
      `), arrows downstream; not trajectories, and their spacing does not encode speed${why}`});
  } else if (plan.why === "unavailable") {
    plan.key.push({warn: true, text: "ψ contours unavailable: " + (wind.info?.psi?.reason || "no streamfunction for this result")});
  }
  if (s.arrows) {
    const m = windMesh(plot);
    plan.quiver = quiverStations(m.lat, m.lon, m.u, m.v, s.spacing);
    plan.vmax = plan.quiver.reduce((a, p) => Math.max(a, Math.hypot(p.u, p.v)), 0);
    plan.referenceArrow = niceSpeed(plan.vmax);
    const vmax = numberText(Number(plan.vmax.toPrecision(3)));
    // The key arrow's speed is final only once drawWind knows the pixel scale (quiverReference).
    const text = reference => (s.length === "unit" ? `Red arrows: direction of the ${mean}wind, equal length (max ${vmax} m s⁻¹)` :
      `Red arrows: ${mean}wind, length ∝ speed; the key arrow is ${numberText(reference)} m s⁻¹`) +
      ` · native mesh nodes about ${numberText(s.spacing)}° apart, poles omitted` +
      (s.colour ? ` · colour: speed from 0 (pale) to ${vmax} m s⁻¹ (dark)` : "");
    plan.key.push({glyph: "quiver", text: text(plan.referenceArrow), retext: s.length === "unit" ? null : text});
  }
  return plan;
}

function stroke(ctx, path, width, colour = WIND) {
  ctx.lineJoin = "round"; ctx.lineCap = "round";
  ctx.strokeStyle = HALO; ctx.lineWidth = width + 2.2; path(); ctx.stroke();
  ctx.strokeStyle = colour; ctx.lineWidth = width; path(); ctx.stroke();
}
function arrowHead(ctx, x, y, dx, dy, size, colour = WIND) {
  const path = () => { ctx.beginPath(); ctx.moveTo(x + dx * size * 0.35, y + dy * size * 0.35);
    ctx.lineTo(x - dx * size * 0.65 + dy * size * 0.42, y - dy * size * 0.65 - dx * size * 0.42);
    ctx.lineTo(x - dx * size * 0.65 - dy * size * 0.42, y - dy * size * 0.65 + dx * size * 0.42); ctx.closePath(); };
  ctx.strokeStyle = HALO; ctx.lineWidth = 1.6; path(); ctx.stroke();
  ctx.fillStyle = colour; path(); ctx.fill();
}

/** The on-screen unit direction of the wind (u, v) at latitude p.lat: the map's image of the
 * vector, whose eastward part spans 1/cos φ more degrees of longitude. `lonPx`, `latPx`: signed
 * pixels per degree of longitude and of latitude along their screen axes. */
export function quiverDirection(p, lonPx, latPx, lonOnX) {
  const e = p.u / Math.cos(p.lat * DEG) * lonPx, n = p.v * latPx;
  const dx = lonOnX ? e : n, dy = lonOnX ? n : e, norm = Math.hypot(dx, dy);
  return [dx / norm, dy / norm];
}

/** Draw a wind plan on a field box; returns counts and the quiver's pixel scale. */
function drawWind(ctx, plot, plan, box, X, Y) {
  const s = plot.wind.settings, lonOnX = plot.x.coordinate === "longitude";
  const pixel = (lat, lon) => {
    const east = ((lon % 360) + 360) % 360;
    return lonOnX ? [X(unit(plot.x, east)), Y(unit(plot.y, lat)), east] : [X(unit(plot.x, lat)), Y(unit(plot.y, east)), east];
  };
  const jump = Math.max(box.w, box.h) / 2, out = {mode: plan.mode, lines: plan.lines.length, closed: plan.closed,
    interval: plan.interval, requested: plan.requested ?? null, arrows: 0, quiver: 0};
  ctx.save(); ctx.beginPath(); ctx.rect(box.x, box.y, box.w, box.h); ctx.clip();
  const heads = [];
  for (const line of plan.lines) {
    const pts = line.map(([lat, lon]) => pixel(lat, lon));
    stroke(ctx, () => {
      ctx.beginPath();
      pts.forEach(([x, y, east], k) => {
        if (k === 0 || Math.abs(east - pts[k - 1][2]) > 180) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
    }, 1.15);
    heads.push(...arrowStations(pts.map(([x, y]) => [x, y]), plan.mode === "psi" ? 1.25 * ARROW_SPACING : ARROW_SPACING, jump));
  }
  for (const [x, y, dx, dy] of heads) arrowHead(ctx, x, y, dx, dy, plan.mode === "psi" ? 8 : 7);
  out.arrows = heads.length;
  if (plan.quiver) {
    // Pixels per degree of longitude and of latitude, with their signs on screen.
    const lonPx = lonOnX ? X(1) - X(0) : Y(1) - Y(0), latPx = lonOnX ? Y(1) - Y(0) : X(1) - X(0);
    const room = s.spacing * Math.min(Math.abs(lonPx), Math.abs(latPx));
    const scale = plan.vmax > 0 ? room / plan.vmax : 0;
    for (const p of plan.quiver) {
      const speed = Math.hypot(p.u, p.v);
      if (!(speed > 0)) continue;
      const [dx, dy] = quiverDirection(p, lonPx, latPx, lonOnX);
      const length = s.length === "unit" ? 0.7 * room : speed * scale;
      const [x0, y0] = pixel(p.lat, p.lon), x1 = x0 + dx * length, y1 = y0 + dy * length;
      let colour = QUIVER;
      if (s.colour) {
        const q = Math.min(1, speed / plan.vmax) * (RAMP.length - 1), a = Math.min(Math.floor(q), RAMP.length - 2);
        colour = `rgb(${RAMP[a].map((v, i) => Math.round(v + (q - a) * (RAMP[a + 1][i] - v)))})`;
      }
      stroke(ctx, () => { ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); }, 1.1, colour);
      arrowHead(ctx, x1, y1, dx, dy, Math.min(7, 1.2 + length * 0.4), colour);
      out.quiver++;
    }
    const reference = s.length === "unit" ? plan.referenceArrow : quiverReference(plan.vmax, scale);
    out.quiverScale = {vmax: plan.vmax, reference, pixelsPerMs: scale,
      keyPixels: s.length === "unit" ? Math.min(KEY_ARROW, 0.7 * room) : reference * scale};
  }
  ctx.restore();
  return out;
}

/** The longest key arrow, in pixels: it must fit beside the key text. */
const KEY_ARROW = 64;

/** The key arrow's speed: the plotted scale's nice speed whose arrow fits in KEY_ARROW pixels.
 * The key arrow is drawn at the same pixels per m s⁻¹ as the plotted arrows, so its label
 * changes with it rather than the arrow being shortened alone. */
export function quiverReference(vmax, pixelsPerMs) {
  if (!(pixelsPerMs > 0)) return niceSpeed(vmax);
  return niceSpeed(Math.min(vmax, KEY_ARROW / pixelsPerMs));
}

/** Wrapped key rows: [{glyph, warn, lines}] and their height. A row whose text depends on the
 * drawn scale (`retext`) reserves one extra line, since its final text is wrapped again. */
function windKeyLayout(ctx, plan, width) {
  const rows = plan.key.map(row => {
    ctx.font = (row.warn ? "600 " : "") + "11px system-ui";
    return {...row, width: width - (row.glyph ? 76 : 0), lines: wrapped(ctx, row.text, width - (row.glyph ? 76 : 0))};
  });
  return {rows, height: rows.reduce((h, r) => h + 15 * (r.lines.length + (r.retext ? 1 : 0)) + 5, 0)};
}

function drawWindKey(ctx, layout, drawn, box, top) {
  let y = top;
  for (const row of layout.rows) {
    const x = box.x;
    if (row.glyph === "line") {
      stroke(ctx, () => { ctx.beginPath(); ctx.moveTo(x, y - 4); ctx.lineTo(x + 30, y - 4); }, 1.15);
      arrowHead(ctx, x + 20, y - 4, 1, 0, 8);
    } else if (row.glyph === "quiver") {
      const l = drawn.quiverScale?.keyPixels || 30;
      stroke(ctx, () => { ctx.beginPath(); ctx.moveTo(x, y - 4); ctx.lineTo(x + l, y - 4); }, 1.1, QUIVER);
      arrowHead(ctx, x + l, y - 4, 1, 0, Math.min(7, 1.2 + l * 0.4), QUIVER);
    }
    ctx.font = (row.warn ? "600 " : "") + "11px system-ui"; ctx.fillStyle = row.warn ? WARN : INK; ctx.textAlign = "left";
    const lines = row.retext && drawn.quiverScale ? wrapped(ctx, row.retext(drawn.quiverScale.reference), row.width) : row.lines;
    lines.forEach((line, k) => ctx.fillText(line, x + (row.glyph ? 76 : 0), y + 15 * k));
    y += 15 * lines.length + 5;
  }
}

/** A two-dimensional result: cells between edges, an optional surface strip. */
function drawField(canvas, plot, width, ratio) {
  const ctx = canvas.getContext("2d");
  const {x: ax, y: ay} = plot;
  const rows = plot.values.length, cols = ax.values.length;
  const flat = new Float32Array(rows * cols);
  plot.values.forEach((row, k) => flat.set(row.map(v => v ?? NaN), k * cols));
  const strip = plot.surface ? 14 : 0;
  const all = plot.surface ? Float32Array.from([...flat, ...plot.surface]) : flat;
  const scale = fieldScale({values: all}, {scale: plot.diverging ? "diverging" : "sequential", range: plot.range});
  const plotHeight = Math.round(Math.min(440, Math.max(220, width * 0.42)));
  const windPlanned = plot.wind ? windPlan(plot) : null;
  const keyLayout = windPlanned ? windKeyLayout(ctx, windPlanned, width - 74 - 24) : null;
  const height = headerHeight(ctx, plot.title, width) + plotHeight + strip + 132 + (keyLayout ? keyLayout.height : 0);
  canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  const top = header(ctx, plot.title, width);
  const right = ay.secondary ? 96 : 24;
  const box = {x: 74, y: top + 6 + strip, w: width - 74 - right, h: plotHeight};
  const xe = edgesOf(ax), ye = edgesOf(ay);
  if (ye[0] === null) ye[0] = 2 * unit(ay, ay.values[0]) - ye[1];       // a top face at 0 on a log axis
  const [x0, x1] = [Math.min(xe[0], xe[cols]), Math.max(xe[0], xe[cols])];
  const [y0, y1] = [Math.min(ye[0], ye[rows]), Math.max(ye[0], ye[rows])];
  const X = v => box.x + box.w * (v - x0) / (x1 - x0);
  // Y: the first row's side at the top unless the axis grows upward.
  const up = !ay.downward;
  const Y = v => up ? box.y + box.h * (y1 - v) / (y1 - y0) : box.y + box.h * (v - y0) / (y1 - y0);
  for (let k = 0; k < rows; k++) for (let i = 0; i < cols; i++) {
    const v = plot.values[k][i];
    if (!Number.isFinite(v)) continue;
    const [r, g, b] = scale.color(v);
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    const a = X(xe[i]), c = X(xe[i + 1]), p = Y(ye[k]), q = Y(ye[k + 1]);
    ctx.fillRect(Math.min(a, c), Math.min(p, q), Math.abs(c - a) + 0.6, Math.abs(q - p) + 0.6);
  }
  if (plot.surface) {
    // The diagnosed skin is its own native level: a strip above the cells.
    for (let i = 0; i < cols; i++) {
      const v = plot.surface[i];
      if (!Number.isFinite(v)) continue;
      const [r, g, b] = scale.color(v);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      const a = X(xe[i]), c = X(xe[i + 1]);
      ctx.fillRect(Math.min(a, c), box.y - strip, Math.abs(c - a) + 0.6, strip - 3);
    }
    ctx.fillStyle = INK; ctx.font = "10px system-ui"; ctx.textAlign = "right";
    ctx.fillText("skin", box.x - 6, box.y - 5);
  }
  const winds = windPlanned ? drawWind(ctx, plot, windPlanned, box, X, Y) : null;
  ctx.strokeStyle = GRID; ctx.lineWidth = 1; ctx.font = "11px system-ui"; ctx.fillStyle = INK; ctx.beginPath();
  const yt = tickSet(ay, y0, y1), xt = tickSet(ax, x0, x1);
  for (const t of yt.ticks) {
    ctx.moveTo(box.x, Y(t)); ctx.lineTo(box.x + box.w, Y(t));
    ctx.textAlign = "right"; ctx.fillText(yt.label(t), box.x - 6, Y(t) + 4);
    if (ay.secondary) {
      ctx.textAlign = "left";
      ctx.fillText(pressureText(ay.secondary.factor * (ay.scale === "log" ? 10 ** t : t)), box.x + box.w + 6, Y(t) + 4);
    }
  }
  for (const t of xt.ticks) { ctx.moveTo(X(t), box.y); ctx.lineTo(X(t), box.y + box.h); }
  ctx.stroke();
  ctx.textAlign = "center";
  for (const t of xt.ticks) ctx.fillText(xt.label(t), X(t), box.y + box.h + 14);
  ctx.fillText(axisLabel(ax), box.x + box.w / 2, box.y + box.h + 28);
  ctx.save(); ctx.translate(16, box.y + box.h / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(axisLabel(ay), 0, 0); ctx.restore();
  if (ay.secondary) {
    ctx.save(); ctx.translate(box.x + box.w + 84, box.y + box.h / 2); ctx.rotate(Math.PI / 2);
    ctx.fillText(ay.secondary.label, 0, 0); ctx.restore();
  }
  // The linked geographic cursor: a line per geographic axis (a crosshair on a longitude × latitude field).
  ctx.strokeStyle = CURSOR; ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]); ctx.beginPath();
  if (Number.isFinite(plot.cursor)) { ctx.moveTo(X(plot.cursor), box.y); ctx.lineTo(X(plot.cursor), box.y + box.h); }
  if (Number.isFinite(plot.cursorY)) { ctx.moveTo(box.x, Y(plot.cursorY)); ctx.lineTo(box.x + box.w, Y(plot.cursorY)); }
  ctx.stroke(); ctx.setLineDash([]);
  if (Number.isInteger(plot.selected) && plot.selected < rows) {
    ctx.strokeStyle = INK; ctx.lineWidth = 1.2;
    const p = Y(ye[plot.selected]), q = Y(ye[plot.selected + 1]);
    ctx.strokeRect(box.x, Math.min(p, q), box.w, Math.abs(q - p));
  }
  ctx.strokeStyle = INK; ctx.lineWidth = 1; ctx.strokeRect(box.x, box.y, box.w, box.h);
  colorbar(ctx, scale, {...box, y: box.y}, plot.range ? plot.label + " · fixed colour range; values beyond it take the end colours" : plot.label);
  if (keyLayout) drawWindKey(ctx, keyLayout, winds, box, box.y + box.h + 84);
  return {kind: "field", box, width, height, xe, ye, X, Y, x0, x1, y0, y1, rows, cols, up, strip, range: scale.range, winds};
}

/** A line: values against one coordinate, horizontal (time) or vertical (profile). */
function drawLine(canvas, plot, width, ratio) {
  const ctx = canvas.getContext("2d");
  const vertical = plot.orientation === "profile";
  const plotHeight = Math.round(Math.min(480, Math.max(240, width * (vertical ? 0.5 : 0.36))));
  const legend = (plot.markers || []).length * 18;
  const height = headerHeight(ctx, plot.title, width) + legend + plotHeight + 60;
  canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  let top = header(ctx, plot.title, width);
  ctx.font = "12px system-ui";
  for (const m of plot.markers || []) {
    ctx.fillStyle = m.kind === "surface" ? "#e0a03c" : "#6fb7b5"; ctx.strokeStyle = INK;
    if (m.kind === "surface") { ctx.fillRect(20, top + 2, 10, 10); ctx.strokeRect(20, top + 2, 10, 10); }
    else { ctx.beginPath(); ctx.arc(25, top + 7, 5.5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke(); }
    ctx.fillStyle = INK; ctx.textAlign = "left"; ctx.fillText(m.label, 38, top + 11, width - 52); top += 18;
  }
  const box = {x: 96, y: top + 8, w: width - 124, h: plotHeight};
  // c: the coordinate (time, or the profile's vertical); v: the value
  const ca = plot.coordinate, cs = ca.values.map(v => unit(ca, v));
  const vs = plot.values;
  const markerC = (plot.markers || []).map(m => unit(ca, m.position)), markerV = (plot.markers || []).map(m => m.value);
  // Bounds by loop: long records exceed the argument limit of Math.min(...values).
  const bounds = arrays => {
    let lo = Infinity, hi = -Infinity;
    for (const a of arrays) for (const v of a) if (Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
    return [lo, hi];
  };
  let [vlo, vhi] = bounds([vs, markerV]);
  if (plot.diverging) { vlo = Math.min(vlo, 0); vhi = Math.max(vhi, 0); }
  if (vlo === vhi) { vlo -= 1; vhi += 1; }
  const pad = (vhi - vlo) * 0.06; vlo -= pad; vhi += pad;
  let [clo, chi] = bounds([cs, markerC]);
  if (clo === chi) { clo -= 1; chi += 1; }
  const along = c => (c - clo) / (chi - clo), across = v => (v - vlo) / (vhi - vlo);
  // A profile puts the coordinate on Y (downward axes grow downward) and the value on X.
  const P = vertical ? (c, v) => [box.x + box.w * across(v), ca.downward ? box.y + box.h * along(c) : box.y + box.h * (1 - along(c))]
    : (c, v) => [box.x + box.w * along(c), box.y + box.h * (1 - across(v))];
  ctx.strokeStyle = GRID; ctx.fillStyle = INK; ctx.font = "11px system-ui"; ctx.beginPath();
  const ct = tickSet(ca, clo, chi), vt = niceTicks(vlo, vhi);
  for (const t of ct.ticks) {
    const [px, py] = P(t, vlo);
    if (vertical) { ctx.moveTo(box.x, py); ctx.lineTo(box.x + box.w, py); ctx.textAlign = "right"; ctx.fillText(ct.label(t), box.x - 6, py + 4); }
    else { ctx.moveTo(px, box.y); ctx.lineTo(px, box.y + box.h); ctx.textAlign = "center"; ctx.fillText(ct.label(t), px, box.y + box.h + 14); }
  }
  for (const t of vt) {
    const [px, py] = P(clo, t);
    if (vertical) { ctx.moveTo(px, box.y); ctx.lineTo(px, box.y + box.h); ctx.textAlign = "center"; ctx.fillText(valueText(t), px, box.y + box.h + 14); }
    else { ctx.moveTo(box.x, py); ctx.lineTo(box.x + box.w, py); ctx.textAlign = "right"; ctx.fillText(valueText(t), box.x - 6, py + 4); }
  }
  ctx.stroke();
  const valueLabel = plot.valueLabel, coordinateLabel = axisLabel(ca);
  ctx.textAlign = "center";
  ctx.fillText(vertical ? valueLabel : coordinateLabel, box.x + box.w / 2, box.y + box.h + 32);
  ctx.save(); ctx.translate(14, box.y + box.h / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText(vertical ? coordinateLabel : valueLabel, 0, 0); ctx.restore();
  if (plot.surfaceLine && vertical) {
    // The surface of a substrate column: the skin lies on it, the cells below.
    const [, py] = P(unit(ca, 0), vlo);
    ctx.strokeStyle = "rgba(34,52,58,.6)"; ctx.setLineDash([3, 3]); ctx.beginPath();
    ctx.moveTo(box.x, py); ctx.lineTo(box.x + box.w, py); ctx.stroke(); ctx.setLineDash([]);
  }
  ctx.strokeStyle = LINE; ctx.lineWidth = 2; ctx.beginPath();
  let joined = false;
  const points = [];
  vs.forEach((v, k) => {
    if (!Number.isFinite(v) || !Number.isFinite(cs[k])) { joined = false; points.push(null); return; }
    const [px, py] = P(cs[k], v);
    points.push([px, py]);
    if (joined) ctx.lineTo(px, py); else ctx.moveTo(px, py);
    joined = true;
  });
  ctx.stroke();
  if (vs.length <= 400) points.forEach((p, k) => {
    if (!p) return;
    ctx.fillStyle = k === plot.selected ? CURSOR : LINE;
    ctx.beginPath(); ctx.arc(p[0], p[1], k === plot.selected ? 4.5 : 2.4, 0, 2 * Math.PI); ctx.fill();
  });
  (plot.markers || []).forEach((m, i) => {
    const [px, py] = P(markerC[i], m.value);
    ctx.fillStyle = m.kind === "surface" ? "#e0a03c" : "#6fb7b5"; ctx.strokeStyle = INK; ctx.lineWidth = 1;
    if (m.kind === "surface") { ctx.fillRect(px - 5, py - 5, 10, 10); ctx.strokeRect(px - 5, py - 5, 10, 10); }
    else { ctx.beginPath(); ctx.arc(px, py, 5.5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke(); }
  });
  ctx.strokeStyle = INK; ctx.lineWidth = 1; ctx.strokeRect(box.x, box.y, box.w, box.h);
  return {kind: "line", box, width, height, points, vertical};
}

function option(value, text, disabled = false) {
  const o = new Option(text, value);
  o.disabled = disabled;
  return o;
}

/** A quantity the globe can draw: one saved state on one longitude × latitude surface. */
export const onGlobe = q => {
  const c = q.coordinates;
  return c.longitude.roles.includes("x") && c.latitude.roles.includes("y") && c.time.roles.includes("fixed") &&
    (c.vertical.roles.includes("fixed") || c.vertical.roles.includes("none"));
};

const lacks = caps => caps.roles.length === 1 && caps.roles[0] === "none";
const ROLE_TEXT = {x: "the horizontal axis", y: "the vertical axis", fixed: "fixed", reduce: "reduced", none: "absent"};
const ORIGIN = {stored: "stored in the archive", derived: "derived from stored fields", recorded: "archived diagnostic"};

/** The saved sample at or before ``seconds`` (the first otherwise). */
function atOrBefore(values, seconds) {
  const k = values.findIndex(s => s > seconds);
  return k < 0 ? values.length - 1 : Math.max(0, k - 1);
}

export class Explorer {
  /** The analytical workspace and the owner of the selection.
   *
   * ``requested`` holds the roles the person chose, ``ops`` the statistic of
   * each reduction, ``vfixed`` the fixed vertical surface and ``times`` the
   * chosen saved times (as seconds). Changing the quantity changes none of
   * them: a quantity that cannot answer the configured question shows why
   * and offers an explicit adaptation. The globe draws the selection's
   * quantity at its saved time and fixed surface (``globeSelection``).
   *
   * links: manifest, runName, timeText(axis, index), onCursor(c), onSelection(). */
  constructor(node, links) {
    this.node = node;
    this.links = links;
    this.c = Object.fromEntries([...node.querySelectorAll("[data-e]")].map(e => [e.dataset.e, e]));
    this.explorer = links.manifest.explorer;
    this.presets = this.explorer.presets || {};
    this.ranges = new Map();
    this.rows = {};
    this.display = {scale: "auto", direction: "auto"};
    this.requested = {longitude: "none", latitude: "none", vertical: "none", time: "none"};
    this.ops = {longitude: "mean", latitude: "mean", vertical: "max", time: "mean"};
    this.vfixed = {representation: "native", level: null,
      values: {sigma: "0.5", pressure: "50000", geopotential_height: "1000", height_above_surface: "1000"}};
    this.vaxis = "sigma";
    this.times = {index: null, first: null, last: null};
    const c = this.c, keys = Object.keys(this.explorer.quantities);
    this.fillQuantities("");
    c.quantity.value = keys.includes("temperature") ? "temperature" : keys[0];
    this.buildRows();
    c.quantity.onchange = () => { this.update(); };
    c["quantity-search"].oninput = () => this.fillQuantities(c["quantity-search"].value);
    c.preset.onchange = () => { if (c.preset.value) { this.applyPreset(c.preset.value); this.update(); } };
    c.snap.onchange = () => this.update();
    // Wind display settings redraw what is on hand; only turning the layer on or off asks the server.
    // Typed numbers (limit, interval, spacing) apply when committed, not at each keystroke: a
    // half-typed interval such as "1" would otherwise be traced at once.
    for (const id of ["wind-lines", "wind-limit", "wind-interval", "wind-density", "wind-arrows", "wind-spacing", "wind-length", "wind-colour"]) c[id].onchange = () => this.windChanged();
    for (const id of ["wind-density"]) c[id].oninput = () => this.windChanged();
    c["range-mode"].onchange = () => {
      if (c["range-mode"].value === "fixed" && this.layout?.range) {
        if (c["range-min"].value === "") c["range-min"].value = String(Number(this.layout.range[0].toPrecision(6)));
        if (c["range-max"].value === "") c["range-max"].value = String(Number(this.layout.range[1].toPrecision(6)));
      }
      this.storeRange();
    };
    c["range-min"].onchange = c["range-max"].onchange = () => this.storeRange();
    c.png.onclick = () => this.png();
    c.plot.addEventListener("pointermove", e => this.hover(e));
    c.plot.addEventListener("pointerleave", () => { c.readout.textContent = ""; });
    c.plot.addEventListener("click", e => this.click(e));
    new ResizeObserver(() => { if (this.data && this.node.clientWidth !== this.width) this.draw(); }).observe(node);
    const query = new URLSearchParams(location.search), requested = query.get("preset");
    this.applyPreset(requested && this.valid(this.presets[requested]?.roles) ? requested : this.firstPreset());
    // A configuration carried from another archive replaces the preset: the analytical question is kept.
    if (query.get("config")) this.applyConfig(query.get("config"));
    node.hidden = false;
  }

  /** The analytical configuration (roles, statistics, vertical settings) as text: carried across archives. */
  configText() {
    return JSON.stringify({requested: this.requested, ops: this.ops, vaxis: this.vaxis, vfixed: this.vfixed,
      display: this.display});
  }

  applyConfig(text) {
    let config;
    try { config = JSON.parse(text); } catch { return; }
    if (!config || typeof config !== "object") return;
    const roles = config.requested || {};
    if (COORDINATES.every(c => ROLE_ORDER.includes(roles[c]))) this.requested = {...roles};
    for (const c of COORDINATES) if (OPS.includes(config.ops?.[c])) this.ops[c] = config.ops[c];
    if (typeof config.vaxis === "string") this.vaxis = config.vaxis;
    const v = config.vfixed;
    if (v && typeof v.representation === "string") {
      this.vfixed.representation = v.representation;
      if (Number.isInteger(v.level)) this.vfixed.level = v.level;
      for (const [k, value] of Object.entries(v.values || {})) if (k in this.vfixed.values) this.vfixed.values[k] = String(value);
    }
    if (config.display) {
      for (const k of ["scale", "direction"]) if (typeof config.display[k] === "string") {
        this.display[k] = config.display[k];
        this.rows.vertical[k].value = config.display[k];
      }
    }
    this.syncRows();
  }

  get quantity() { return this.explorer.quantities[this.c.quantity.value]; }

  quantityLabel(key) {
    const p = this.explorer.quantities[key].presentation;
    return [p.symbol ? p.name + " (" + p.symbol + ")" : p.name, p.units].filter(Boolean).join(" · ");
  }

  /** The quantity selector: grouped by the registry's catalogue group, sorted by name, filtered by text. */
  fillQuantities(filter) {
    const c = this.c, quantities = this.explorer.quantities, keep = c.quantity.value;
    const words = filter.trim().toLowerCase().split(/\s+/).filter(Boolean);
    const groups = new Map();
    for (const key of Object.keys(quantities)) {
      const p = quantities[key].presentation;
      const text = [key, p.name, p.symbol, p.units, p.group].join(" ").toLowerCase();
      if (words.length && !words.every(w => text.includes(w)) && key !== keep) continue;
      const group = p.group || "Other";
      if (!groups.has(group)) groups.set(group, []);
      groups.get(group).push(key);
    }
    const order = this.explorer.groups || [];
    const names = [...groups.keys()].sort((a, b) => (order.indexOf(a) + 1 || 99) - (order.indexOf(b) + 1 || 99));
    c.quantity.replaceChildren(...names.map(name => {
      const group = document.createElement("optgroup");
      group.label = name;
      group.append(...groups.get(name).sort((a, b) => quantities[a].presentation.name.localeCompare(quantities[b].presentation.name))
        .map(k => option(k, this.quantityLabel(k))));
      return group;
    }));
    if (keep) c.quantity.value = keep;
  }

  /** The role a coordinate takes for this quantity: a fixed vertical coordinate of a quantity
   * without one is absent (nothing is fabricated; the setting is kept for 3-D quantities). */
  effective(coordinate, quantity = this.quantity) {
    const r = this.requested[coordinate], caps = quantity.coordinates[coordinate];
    return coordinate === "vertical" && lacks(caps) && r === "fixed" ? "none" : r;
  }

  roles(quantity = this.quantity) {
    return Object.fromEntries(COORDINATES.map(c => [c, this.effective(c, quantity)]));
  }

  /** Every role accepted by its coordinate and each axis used once: the server's own checks. */
  valid(roles, quantity = this.quantity) {
    if (!roles) return false;
    const caps = quantity.coordinates;
    const eff = c => c === "vertical" && lacks(caps[c]) && roles[c] === "fixed" ? "none" : roles[c];
    if (!COORDINATES.every(c => caps[c].roles.includes(eff(c)))) return false;
    return ["x", "y"].every(a => COORDINATES.filter(c => eff(c) === a).length <= 1);
  }

  firstPreset() {
    return ["profile", "time_series", "recorded_series", "meridional_section", "zonal_mean", "time_section", "map"]
      .find(p => this.valid(this.presets[p]?.roles)) || Object.keys(this.presets).find(p => this.valid(this.presets[p].roles));
  }

  applyPreset(name) {
    const preset = this.presets[name];
    if (preset) {
      this.requested = {...preset.roles};
      // The presets' reductions are means (zonal mean, time mean): their labels say so.
      for (const c of COORDINATES) if (preset.roles[c] === "reduce") this.ops[c] = "mean";
    }
    this.syncRows();
  }

  buildRows() {
    const box = this.c.coords;
    box.replaceChildren(Object.assign(document.createElement("legend"), {textContent: "Coordinates"}));
    for (const name of COORDINATES) {
      const row = document.createElement("div");
      row.className = "coord";
      row.dataset.coordinate = name;
      const label = Object.assign(document.createElement("label"), {textContent: NAMES[name]});
      const role = document.createElement("select");
      role.setAttribute("aria-label", NAMES[name] + " role");
      label.append(role);
      row.append(label);
      const parts = {row, role};
      parts.op = document.createElement("select");
      parts.op.setAttribute("aria-label", NAMES[name] + " statistic");
      parts.op.onchange = () => { this.ops[name] = parts.op.value; this.update(); };
      row.append(parts.op);
      if (name === "longitude" || name === "latitude") {
        parts.value = Object.assign(document.createElement("input"), {type: "number", step: "any"});
        parts.value.setAttribute("aria-label", NAMES[name] + " (degrees " + (name === "latitude" ? "north" : "east") + ")");
        if (name === "latitude") { parts.value.min = "-90"; parts.value.max = "90"; }
        parts.value.onchange = () => {
          const lat = this.rows.latitude.value.valueAsNumber, lon = this.rows.longitude.value.valueAsNumber;
          if (Number.isFinite(lat) && Number.isFinite(lon)) this.links.onCursor({lat, lon});
          else this.update();
        };
        row.append(parts.value);
      } else if (name === "vertical") {
        parts.representation = document.createElement("select");
        parts.representation.setAttribute("aria-label", "Vertical representation");
        parts.level = document.createElement("select");
        parts.level.setAttribute("aria-label", "Native level");
        parts.value = Object.assign(document.createElement("input"), {type: "number", step: "any"});
        parts.value.setAttribute("aria-label", "Vertical value");
        parts.representation.onchange = () => {
          if (this.requested.vertical === "fixed") this.vfixed.representation = parts.representation.value;
          else this.vaxis = parts.representation.value;
          this.update();
        };
        parts.level.onchange = () => { this.vfixed.level = Number(parts.level.value); this.update(); };
        parts.value.onchange = () => { this.vfixed.values[this.vfixed.representation] = parts.value.value; this.update(); };
        // How the vertical axis is drawn: display only, no request.
        parts.display = Object.assign(document.createElement("div"), {className: "pair"});
        parts.scale = document.createElement("select");
        parts.scale.setAttribute("aria-label", "Vertical axis scale");
        parts.scale.append(option("auto", "Scale: default"), option("linear", "Linear"), option("log", "Logarithmic"));
        parts.direction = document.createElement("select");
        parts.direction.setAttribute("aria-label", "Vertical axis direction");
        parts.direction.append(option("auto", "Direction: default"), option("up", "Increasing upward"),
          option("down", "Increasing downward"));
        parts.scale.onchange = parts.direction.onchange = () => {
          this.display = {scale: parts.scale.value, direction: parts.direction.value};
          if (this.data && !this.pending) this.draw();
        };
        parts.display.append(parts.scale, parts.direction);
        row.append(parts.representation, parts.level, parts.value, parts.display);
      } else {
        parts.index = document.createElement("select");
        parts.index.setAttribute("aria-label", "Saved time");
        parts.first = document.createElement("select");
        parts.first.setAttribute("aria-label", "First saved time");
        parts.last = document.createElement("select");
        parts.last.setAttribute("aria-label", "Last saved time");
        for (const k of ["index", "first", "last"]) parts[k].onchange = () => {
          this.times[k] = this.axisValues()[Number(parts[k].value)];
          this.update();
        };
        parts.interval = Object.assign(document.createElement("p"), {className: "hint"});
        row.append(parts.index, parts.first, parts.last, parts.interval);
      }
      role.onchange = () => { this.requested[name] = role.value; this.update(); };
      box.append(row);
      this.rows[name] = parts;
    }
  }

  /** Every role, the accepted ones enabled; the requested one shown even when this quantity refuses it. */
  fillRoles(coordinate) {
    const caps = this.quantity.coordinates[coordinate], select = this.rows[coordinate].role;
    const requested = this.requested[coordinate], effective = this.effective(coordinate);
    const shown = lacks(caps) ? ["none"] : ROLE_ORDER.filter(r => r !== "none");
    if (!shown.includes(requested)) shown.push(requested);
    const name = this.quantity.presentation.name;
    select.replaceChildren(...shown.map(r => {
      let text = coordinate === "time" && r === "fixed" ? "Fixed (one saved time)" : ROLE_NAMES[r];
      let accepted = caps.roles.includes(r);
      if (lacks(caps) && r === requested && effective === "none" && r !== "none") {
        text += " · " + name + " has no vertical coordinate (kept for 3-D quantities)";
        accepted = true;
      } else if (!accepted) text += " · unavailable";
      const o = option(r, text, !accepted);
      if (!caps.roles.includes(r)) o.title = caps.unavailable?.[r] || (lacks(caps) ? name + " has no " + coordinate + " coordinate" :
        "Not offered for this quantity");
      return o;
    }));
    select.value = requested;
  }

  axisValues() { return this.links.manifest.time_axes[this.timeAxis()]?.values || []; }

  /** Time axis of the selection: a fixed level's own axis, else the quantity's. */
  timeAxis() {
    const q = this.quantity;
    if (q.vertical && this.effective("vertical") === "fixed" && this.vfixed.representation === "native") {
      const level = this.level();
      if (level?.time_axis) return level.time_axis;
    }
    return q.time_axis;
  }

  level() {
    const q = this.quantity;
    if (!q.vertical) return null;
    return q.vertical.levels.find(l => l.index === this.vfixed.level) || null;
  }

  discrete(rep) {
    // Native levels, and every representation of a substrate column (native depths only).
    return rep === "native" || this.quantity.vertical?.domain === "substrate";
  }

  /** One sample of every coordinate other than the vertical: no axis and no reduction elsewhere. */
  singleSample() {
    const roles = this.roles();
    return COORDINATES.every(c => c === "vertical" || ["fixed", "none"].includes(roles[c]));
  }

  describe(roles = this.roles()) {
    const axes = ["x", "y"].map(a => COORDINATES.find(c => roles[c] === a)).filter(Boolean);
    const head = axes.map((c, i) => i ? c : NAMES[c]).join(" × ") || "Single value";
    const rest = COORDINATES.filter(c => ["reduce", "fixed"].includes(roles[c]))
      .map(c => c + (roles[c] === "reduce" ? " " + this.ops[c] : " fixed"));
    return [head, ...rest].join(" · ");
  }

  syncRows() {
    const q = this.quantity, c = this.c, p = q.presentation;
    c.classification.textContent = [p.role, ORIGIN[p.origin] || p.origin].filter(Boolean).join(" · ");
    c.definition.textContent = p.definition + (q.evaluation ? " Evaluation: " + q.evaluation + "." : "");
    c.dependencies.textContent = "Coordinates: " + COORDINATES.filter(n => !lacks(q.coordinates[n])).map(n =>
      n === "vertical" && q.vertical ? "vertical (" + q.vertical.levels.length + " native " +
        (q.vertical.domain === "substrate" ? "levels incl. the skin" : "levels") + ")" :
      n === "time" ? "time (" + this.axisValues().length + " " + (this.links.manifest.time_axes[q.time_axis]?.label || "saved") +
        " samples)" : n).join(", ") + ".";
    for (const name of COORDINATES) {
      this.fillRoles(name);
      const r = this.rows[name], caps = q.coordinates[name], reducing = this.requested[name] === "reduce";
      r.op.hidden = !reducing;
      if (reducing) {
        const ops = OPS.filter(op => op in (caps.reduce || {}) || op in (caps.reduce_unavailable || {}) || op === this.ops[name]);
        r.op.replaceChildren(...ops.map(op => {
          const ok = op in (caps.reduce || {});
          const o = option(op, OP_NAMES[op] + (ok ? "" : " · unavailable"), !ok);
          o.title = ok ? caps.reduce[op] : caps.reduce_unavailable?.[op] || "Not offered for this coordinate";
          return o;
        }));
        r.op.value = this.ops[name];
      }
    }
    for (const name of ["longitude", "latitude"]) this.rows[name].value.hidden = this.requested[name] !== "fixed";
    this.syncVertical();
    this.syncTime();
    const roles = this.roles(), preset = this.matchingPreset();
    c.preset.replaceChildren(option("", preset ? "Custom roles" : "Custom roles (no preset)"),
      ...Object.entries(this.presets).map(([key, pr]) => {
        const ok = this.valid(pr.roles);
        return option(key, pr.label + (ok ? "" : " (not available for this quantity)"), !ok);
      }));
    c.preset.value = preset || "";
    const fixedHorizontal = ["longitude", "latitude"].some(n => roles[n] === "fixed");
    c["snap-wrap"].hidden = !(fixedHorizontal && this.links.manifest.geometry?.grid === "latlon" &&
      this.explorer.horizontal?.kind === "spectral");
    const reason = this.windUnavailable();
    for (const id of ["wind-lines", "wind-limit", "wind-interval", "wind-density", "wind-arrows", "wind-spacing", "wind-length", "wind-colour"]) c[id].disabled = Boolean(reason);
    c["wind-fieldset"].title = reason || this.quantity.layers.wind.definition;
    c["wind-reason"].textContent = reason ? "The wind layer is unavailable: " + reason : "";
    c["wind-reason"].hidden = !reason;
  }

  /** The wind layer's display settings (no request depends on them). */
  windSettings() {
    const c = this.c, number = (el, fallback) => Number.isFinite(el.valueAsNumber) ? el.valueAsNumber : fallback;
    return {lines: c["wind-lines"].value, limit: Math.max(0, number(c["wind-limit"], 1)),
      interval: c["wind-interval"].valueAsNumber > 0 ? c["wind-interval"].valueAsNumber : null,
      density: number(c["wind-density"], 1), arrows: c["wind-arrows"].checked,
      spacing: Math.min(45, Math.max(2, number(c["wind-spacing"], 10))), length: c["wind-length"].value,
      colour: c["wind-colour"].checked};
  }

  windWanted() {
    const s = this.windSettings();
    return s.lines !== "none" || s.arrows;
  }

  windChanged() {
    const {selection} = this.selection();
    if (selection && JSON.stringify(selection) === this.loaded && this.data) this.draw();
    else this.update();
  }

  /** Why the current roles cannot carry the wind layer, or "" (the server applies the same rule). */
  windUnavailable() {
    const layer = this.quantity.layers?.wind, roles = this.roles();
    if (!layer?.available) return (layer?.reason || "this archive has no horizontal wind") + ".";
    if (!(["x", "y"].includes(roles.longitude) && ["x", "y"].includes(roles.latitude)))
      return "they are drawn on a longitude × latitude result.";
    if (!["fixed", "none"].includes(roles.vertical)) return "the wind needs the field's own surface: fix the vertical coordinate.";
    if (roles.time === "reduce" && this.ops.time !== "mean")
      return "they are drawn of one saved wind or of the time-mean wind, not of a time " + OP_NAMES[this.ops.time].toLowerCase() + ".";
    return "";
  }

  /** The preset whose roles are the current ones, or null. A label only. */
  matchingPreset() {
    const roles = this.roles();
    return Object.keys(this.presets).find(p => {
      const r = this.presets[p].roles;
      return this.valid(r) && COORDINATES.every(c => (c === "vertical" && lacks(this.quantity.coordinates[c]) &&
        r[c] === "fixed" ? "none" : r[c]) === roles[c] && (roles[c] !== "reduce" || this.ops[c] === "mean"));
    }) || null;
  }

  syncTime() {
    const t = this.rows.time, role = this.requested.time, values = this.axisValues();
    const axis = this.timeAxis();
    for (const k of ["index", "first", "last"]) {
      const select = t[k];
      if (select.dataset.axis !== axis) {
        select.replaceChildren(...values.map((s, i) => option(i, this.links.timeText(axis, i))));
        select.dataset.axis = axis;
      }
      // Atmospheric histories synthesize every saved state of the range: default to the last 60.
      const span = this.explorer.horizontal?.kind === "spectral" ? 60 : values.length;
      const fallback = k === "last" ? values.length - 1 : k === "first" ? Math.max(0, values.length - span) : 0;
      if (values.length) select.value = String(this.times[k] == null ? fallback : atOrBefore(values, this.times[k]));
    }
    t.index.hidden = role !== "fixed";
    t.first.hidden = t.last.hidden = !["x", "y", "reduce"].includes(role) || !values.length || this.quantity.recorded;
    const [a, b] = [Number(t.first.value), Number(t.last.value)].sort((m, n) => m - n);
    t.interval.hidden = t.first.hidden;
    if (!t.first.hidden) t.interval.textContent = (b - a + 1) + " saved samples in the interval " +
      (role === "reduce" ? "reduced by its " + OP_NAMES[this.ops.time].toLowerCase() : "plotted") + ".";
  }

  syncVertical() {
    const q = this.quantity, v = this.rows.vertical, role = this.effective("vertical");
    // A fixed surface always exists (the globe draws it): the lowest atmospheric level or the skin by default.
    if (q.vertical && !q.vertical.levels.some(l => l.index === this.vfixed.level))
      this.vfixed.level = q.vertical.levels[q.vertical.domain === "substrate" ? 0 : q.vertical.levels.length - 1].index;
    const shown = Boolean(q.vertical) && ["fixed", "x", "y"].includes(role);
    v.representation.hidden = !shown;
    v.display.hidden = !shown || !["x", "y"].includes(role);
    if (!shown) { v.level.hidden = v.value.hidden = true; return; }
    const caps = q.coordinates.vertical, fixed = role === "fixed";
    const allowed = fixed ? caps.fixed : [...caps.axis, ...(this.singleSample() ? caps.per_sample || [] : [])];
    const want = fixed ? this.vfixed.representation : this.vaxis;
    const reps = allowed.includes(want) || !(want in q.vertical.representations) ? allowed : [...allowed, want];
    v.representation.replaceChildren(...reps.map(r => option(r, q.vertical.representations[r].label +
      (allowed.includes(r) ? "" : " · unavailable here"), !allowed.includes(r))));
    v.representation.value = reps.includes(want) ? want : (allowed[0] ?? "");
    if (!fixed && !reps.includes(want) && allowed.length) this.vaxis = allowed[0];   // a name this quantity lacks
    const rep = v.representation.value, discrete = this.discrete(rep);
    v.level.hidden = !fixed || !discrete;
    v.value.hidden = !fixed || discrete;
    if (fixed && discrete) {
      v.level.replaceChildren(...q.vertical.levels.map(l => option(l.index,
        rep === "native" ? l.label : (q.vertical.representations[rep].label.split(" (")[0] + " = " +
          numberText(l[rep]) + " m" + (l.kind === "surface" ? " · diagnosed skin" : "")))));
      v.level.value = String(this.vfixed.level);
    }
    if (fixed && !discrete) v.value.value = this.vfixed.values[rep] ?? "";
  }

  /** What stops this quantity from answering the configured question: [{coordinate, text, fix, label}]. */
  issues() {
    const q = this.quantity, name = q.presentation.name, out = [];
    // Axes held by coordinates whose role this quantity accepts; an adaptation never takes one of them.
    const taken = new Set(COORDINATES.filter(c => q.coordinates[c].roles.includes(this.effective(c)))
      .map(c => this.effective(c)).filter(r => r === "x" || r === "y"));
    for (const c of COORDINATES) {
      const caps = q.coordinates[c], r = this.effective(c);
      if (!caps.roles.includes(r)) {
        const reason = caps.unavailable?.[r];
        const target = lacks(caps) ? "none" : caps.roles.includes("fixed") ? "fixed" :
          caps.roles.find(x => !taken.has(x)) || caps.roles[0];
        if (target === "x" || target === "y") taken.add(target);
        out.push({coordinate: c, text: lacks(caps) ? `${name} has no ${c} coordinate, so it cannot be ${ROLE_TEXT[r]}.` :
          `The ${c} of ${name} cannot be ${ROLE_TEXT[r]}` + (reason ? ": " + reason + "." : "."),
          label: `${NAMES[c]}: ${ROLE_NAMES[target].toLowerCase()}`, fix: () => { this.requested[c] = target; }});
        continue;
      }
      if (r === "reduce" && !(this.ops[c] in (caps.reduce || {}))) {
        const first = Object.keys(caps.reduce || {})[0];
        out.push({coordinate: c, text: `The ${OP_NAMES[this.ops[c]].toLowerCase()} over ${c} is not available for ${name}` +
          (caps.reduce_unavailable?.[this.ops[c]] ? ": " + caps.reduce_unavailable[this.ops[c]] + "." : "."),
          label: `${NAMES[c]}: ${OP_NAMES[first].toLowerCase()}`, fix: () => { this.ops[c] = first; }});
      }
    }
    const v = q.coordinates.vertical, role = this.effective("vertical");
    // Representation checks only where the role itself is accepted (a refused role is reported above).
    if (q.vertical && role === "fixed" && v.roles.includes("fixed") && v.fixed.length &&
        !v.fixed.includes(this.vfixed.representation)) {
      const rep = this.vfixed.representation, label = q.vertical.representations[rep]?.label || rep;
      out.push({coordinate: "vertical", text: `${name} cannot be evaluated on a fixed ${label} surface here.`,
        label: "Vertical: " + q.vertical.representations[v.fixed[0]].label, fix: () => { this.vfixed.representation = v.fixed[0]; }});
    }
    if (q.vertical && ["x", "y"].includes(role) && v.roles.includes(role)) {
      const allowed = [...v.axis, ...(this.singleSample() ? v.per_sample || [] : [])];
      if (!allowed.includes(this.vaxis) && allowed.length) {
        const label = q.vertical.representations[this.vaxis]?.label || this.vaxis;
        out.push({coordinate: "vertical", text: `${label} cannot be the vertical axis of this selection: its native-level ` +
          "positions differ from sample to sample, so it is an axis only for a single column at a single saved time.",
          label: "Vertical axis: " + q.vertical.representations[allowed[0]].label, fix: () => { this.vaxis = allowed[0]; }});
      }
    }
    return out;
  }

  /** The selection the rows describe, or a reason why there is none. */
  selection() {
    const q = this.quantity, roles = this.roles();
    for (const a of ["x", "y"]) {
      const along = COORDINATES.filter(c => roles[c] === a);
      if (along.length > 1) return {error: "Only one coordinate can be the " + ROLE_NAMES[a].toLowerCase() + ": " +
        along.join(" and ") + "."};
    }
    if (q.vertical && ["x", "y"].includes(roles.vertical) && !this.rows.vertical.representation.value)
      return {error: "No vertical representation can be an axis here."};
    // Snapping to a native node is offered only on Gaussian grids; elsewhere the point is exact.
    const sel = {quantity: this.c.quantity.value, options: {snap: !this.c["snap-wrap"].hidden && this.c.snap.checked}};
    if (this.windWanted() && !this.windUnavailable()) sel.options.wind = true;
    for (const name of ["longitude", "latitude"]) {
      const r = this.rows[name], role = roles[name];
      sel[name] = {role};
      if (role === "fixed") {
        if (!r.value.validity.valid || r.value.value === "") return {error: "Enter a valid " + name + "."};
        sel[name].value = r.value.valueAsNumber;
      }
      if (role === "reduce") sel[name].op = this.ops[name];
    }
    const v = this.rows.vertical;
    sel.vertical = {role: roles.vertical};
    if (q.vertical && roles.vertical !== "none") {
      if (roles.vertical === "reduce") sel.vertical.op = this.ops.vertical;
      else {
        const rep = v.representation.value;
        sel.vertical.representation = rep;
        if (roles.vertical === "fixed") {
          if (this.discrete(rep)) {
            const level = this.level();
            sel.vertical.value = rep === "native" ? level.index : level[rep];
          } else {
            if (!v.value.validity.valid || !Number.isFinite(v.value.valueAsNumber)) return {error: "Enter a valid vertical value."};
            sel.vertical.value = v.value.valueAsNumber;
          }
        }
      }
    }
    const t = this.rows.time, role = roles.time;
    sel.time = {role};
    if (role === "fixed") sel.time.index = Number(t.index.value);
    else if (["x", "y", "reduce"].includes(role) && !q.recorded) {
      let first = Number(t.first.value), last = Number(t.last.value);
      if (last < first) [first, last] = [last, first];
      sel.time.first = first; sel.time.last = last;
      if (role === "reduce") sel.time.op = this.ops.time;
    }
    return {selection: sel};
  }

  // ------------------------------------------------------------------
  // The globe draws this selection's quantity, saved time and fixed surface
  // ------------------------------------------------------------------

  /** ``{selection, notes}`` for the globe, or ``{error}``. */
  globeSelection() {
    const q = this.quantity, key = this.c.quantity.value, name = q.presentation.name, notes = [];
    if (!onGlobe(q)) return {error: `${name} has no longitude × latitude field at one saved time: the globe shows ` +
      "one saved state on one surface. The Explorer shows it."};
    const values = this.axisValues();
    const index = values.length ? Number(this.rows.time.index.value) : 0;
    const sel = {quantity: key, longitude: {role: "x"}, latitude: {role: "y"}, time: {role: "fixed", index},
      vertical: {role: "none"}};
    if (q.vertical) {
      const caps = q.coordinates.vertical, rep = this.vfixed.representation;
      if (!caps.fixed.includes(rep)) return {error: `The globe's surface (${q.vertical.representations[rep]?.label || rep}) ` +
        `is not available for ${name}.`};
      let value;
      if (this.discrete(rep)) {
        const level = this.level();
        value = rep === "native" ? level?.index : level?.[rep];
      } else value = Number(this.vfixed.values[rep]);
      if (!Number.isFinite(value)) return {error: "Enter a valid vertical value for the globe's surface."};
      sel.vertical = {role: "fixed", representation: rep, value};
      if (this.effective("vertical") !== "fixed") notes.push("Surface: the Explorer's fixed-surface setting (the " +
        "Explorer itself " + (this.effective("vertical") === "reduce" ? "reduces the vertical coordinate" :
          "plots the vertical coordinate along an axis; click a level in its plot to move the globe") + ").");
    }
    const role = this.effective("time");
    if (role !== "fixed") notes.push("Time: the globe shows one saved state; the Explorer " +
      (role === "reduce" ? "reduces time over its interval" : "plots time along an axis") + ".");
    return {selection: sel, notes};
  }

  /** The globe moved the saved time (playback, slider): the selection's fixed time. */
  setTime(index) {
    const values = this.axisValues();
    if (!values.length) return;
    index = Math.max(0, Math.min(values.length - 1, index));
    this.times.index = values[index];
    this.rows.time.index.value = String(index);
    return this.update();
  }

  /** A native level chosen in a profile or section: the fixed surface of the selection. */
  selectLevel(k) {
    const q = this.quantity;
    if (q.vertical?.domain !== "atmosphere" || !q.vertical.levels.some(l => l.index === k)) return;
    if (!q.coordinates.vertical.fixed.includes("native")) return;
    this.vfixed.representation = "native";
    this.vfixed.level = k;
    return this.update();
  }

  /** The selected native level (for highlighting), when the fixed surface is one. */
  selectedLevel() {
    return this.quantity.vertical?.domain === "atmosphere" && this.vfixed.representation === "native" ? this.vfixed.level : null;
  }

  setCursor(cursor) {
    this.cursor = cursor;
    this.rows.latitude.value.value = String(Number(cursor.lat.toFixed(4)));
    this.rows.longitude.value.value = String(Number((((cursor.lon % 360) + 360) % 360).toFixed(4)));
    return this.update();
  }

  get rangeKey() {
    const d = this.data;
    // One colour range per quantity, shared by every two-dimensional result of it.
    return d && d.axes.x && d.axes.y ? d.quantity : null;
  }

  storeRange() {
    const c = this.c, key = this.rangeKey;
    if (!key) return;
    this.ranges.set(key, {mode: c["range-mode"].value, min: c["range-min"].value, max: c["range-max"].value});
    this.showRange();
    this.draw();
  }

  loadRange() {
    const c = this.c, key = this.rangeKey;
    c["range-fieldset"].hidden = !key;
    if (!key || key === this.rangeShown) return;
    this.rangeShown = key;
    const stored = this.ranges.get(key) ?? {mode: "auto", min: "", max: ""};
    c["range-mode"].value = stored.mode; c["range-min"].value = stored.min; c["range-max"].value = stored.max;
    this.showRange();
  }

  showRange() {
    const c = this.c, fixed = c["range-mode"].value === "fixed";
    c["range-wrap"].hidden = !fixed;
    c["range-max"].setCustomValidity(fixed && !this.fixedRange() ? "The maximum must exceed the minimum" : "");
  }

  fixedRange() {
    const c = this.c;
    if (!this.rangeKey || c["range-mode"].value !== "fixed") return null;
    const lo = c["range-min"].valueAsNumber, hi = c["range-max"].valueAsNumber;
    return Number.isFinite(lo) && Number.isFinite(hi) && hi > lo ? [lo, hi] : null;
  }

  /** The configuration cannot be answered by this quantity: keep it, say why, offer the adaptation. */
  showIssues(issues) {
    const c = this.c, box = c.incompatible, name = this.quantity.presentation.name;
    for (const n of COORDINATES) this.rows[n].row.classList.toggle("incompatible", issues.some(i => i.coordinate === n));
    box.hidden = !issues.length;
    if (!issues.length) return;
    const head = Object.assign(document.createElement("strong"), {textContent: `${name} cannot answer this configuration`});
    const keep = Object.assign(document.createElement("p"), {textContent: "Your coordinate roles, fixed values, " +
      "statistics and interval are kept unchanged. Choose a quantity that has them, or adapt the configuration explicitly:"});
    const list = document.createElement("ul");
    list.append(...issues.map(i => Object.assign(document.createElement("li"), {textContent: i.text})));
    const adapt = Object.assign(document.createElement("button"), {type: "button", textContent: "Adapt: " +
      issues.map(i => i.label).join("; ")});
    adapt.dataset.e = "adapt";
    adapt.onclick = () => { for (const i of issues) i.fix(); this.update(); };
    box.replaceChildren(head, keep, list, adapt);
  }

  async update() {
    this.syncRows();
    const issues = this.issues();
    this.showIssues(issues);
    const c = this.c;
    c.product.textContent = this.describe();
    this.links.onSelection?.();
    const {selection, error} = issues.length ? {error: "Not available for this quantity: the configuration is kept; see why below."} :
      this.selection();
    if (error) {
      this.abort?.abort(); this.pending = null; this.data = null; this.loaded = null;
      c.plot.hidden = true; c.table.hidden = true; c.notes.replaceChildren(); c.png.disabled = true;
      c.readout.textContent = "";
      this.status(error, true);
      return;
    }
    const key = JSON.stringify(selection);
    if (key === this.loaded && this.data) {
      // Back to what is on hand: a request still in flight is for something else.
      this.abort?.abort(); this.pending = null;
      this.status(""); this.c.png.disabled = false; this.draw();
      return;
    }
    if (key === this.pending) return;
    this.pending = key;
    this.abort?.abort();
    const abort = new AbortController();
    this.abort = abort;
    c.png.disabled = true;
    this.status("Evaluating the selection…");
    try {
      const response = await fetch("select?" + new URLSearchParams({q: key}), {signal: abort.signal});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || response.statusText);
      if (this.abort !== abort) return;
      this.data = data; this.loaded = key; this.pending = null;
      this.status("");
      this.draw();
      c.png.disabled = false;
    } catch (err) {
      if (abort.signal.aborted || this.abort !== abort) return;
      this.data = null; this.loaded = this.pending = null;
      c.plot.hidden = true; c.table.hidden = true;
      this.status(err.message, true);
    }
  }

  status(text, error = false) {
    this.c.status.textContent = text;
    this.c.status.classList.toggle("error", error);
  }

  identity() {
    const s = this.links.manifest.source;
    return [s.directory, s.config_sha256 ? "config " + s.config_sha256.slice(0, 12) : "", s.completion || ""].filter(Boolean).join(" · ");
  }

  timeText() {
    const t = this.data.coordinates?.time;
    if (!t || t.role === "none") return "";
    if (t.role === "fixed") return this.links.timeText(t.axis, t.index) + " · one saved sample";
    const values = this.links.manifest.time_axes[t.axis]?.values;
    const span = values ? dayText(values[t.first]) + " to " + dayText(values[t.last]) : "";
    if (t.role === "reduce") {
      if (t.op !== "mean") return `${OP_NAMES[t.op].toLowerCase()} over ${t.count} saved samples, ${span}` +
        (t.masked_excluded ? ` (${t.masked_excluded} masked samples excluded)` : "");
      return t.count === 1 ? "mean of a single saved sample" :
        `time mean of ${t.count} saved samples, ${span} (${t.cadence} cadence; ${t.weighting})`;
    }
    return [`${t.count} saved samples`, span].filter(Boolean).join(", ");
  }

  where() {
    const k = this.data.coordinates || {}, parts = [];
    const lat = k.latitude, lon = k.longitude;
    if (lat?.role === "fixed" && lon?.role === "fixed") {
      parts.push(place({lat: lat.value, lon: lon.value}) + (lat.column !== undefined ? " (column " + lat.column + ")" : "") +
        " · " + lat.sampling);
    } else {
      if (lat?.role === "fixed") parts.push("at " + degrees(lat.value, "N", "S") + " · " + lat.sampling);
      if (lon?.role === "fixed") parts.push("at " + numberText(lon.value) + "°E · " + lon.sampling);
    }
    for (const [name, part] of [["longitude", lon], ["latitude", lat]]) {
      if (part?.role === "reduce") parts.push(name + " " + OP_NAMES[part.op].toLowerCase() + " (" + part.nodes + ")");
    }
    const v = k.vertical;
    if (v?.role === "fixed") parts.push((v.level || v.label + " = " + numberText(v.value)) +
      (v.interpolation ? " (" + v.interpolation + ")" : ""));
    if (v?.role === "reduce") parts.push("vertical " + OP_NAMES[v.op].toLowerCase() + " over " + v.nodes);
    return parts.join(" · ");
  }

  /** A vertical axis drawn as the display controls ask; other axes unchanged. */
  displayAxis(a) {
    if (!a || a.coordinate !== "vertical") return a;
    const positive = a.values.every(v => v > 0);
    this.rows.vertical.scale.options[2].disabled = !positive;
    const scale = this.display.scale === "auto" || (this.display.scale === "log" && !positive) ? a.scale : this.display.scale;
    const downward = this.display.direction === "auto" ? a.downward : this.display.direction === "down";
    return {...a, scale: scale === "log" && !positive ? "linear" : scale, downward};
  }

  symbol() {
    const p = this.data.presentation, s = p.symbol || p.name;
    const ops = COORDINATES.filter(c => this.data.roles?.[c] === "reduce").map(c => this.data.coordinates?.[c]?.op || "mean");
    if (!ops.length) return s;
    return ops.every(o => o === "mean") ? "[" + s + "]" : ops.map(o => o).join("·") + " " + s;
  }

  plotSpec() {
    const d = this.data, p = d.presentation;
    const label = (p.symbol ? p.name + " (" + p.symbol + ")" : p.name) + (p.units ? " · " + p.units : "");
    const title = [this.links.runName() + " · " + d.description, this.identity(),
      [this.where(), this.timeText()].filter(Boolean).join(" · "), d.sampling];
    const diverging = DIVERGING.test(d.quantity);
    const axes = ["x", "y"].filter(a => d.axes[a]);
    const atmosphere = this.quantity.vertical?.domain === "atmosphere" && d.quantity !== "mass_streamfunction";
    if (!axes.length) return {kind: "value", title, label, value: d.values};
    if (axes.length === 1) {
      const a = axes[0], coordinate = this.displayAxis(d[a]);
      return {kind: "line", orientation: a === "y" ? "profile" : "series", coordinate, values: d.values,
        markers: d.markers, title, valueLabel: label, diverging,
        surfaceLine: coordinate.coordinate === "vertical" && this.quantity.vertical?.domain === "substrate",
        selected: coordinate.coordinate === "vertical" && atmosphere ? this.selectedLevel() : null};
    }
    const lon = ((this.cursor?.lon % 360) + 360) % 360;
    const at = a => a.coordinate === "latitude" ? this.cursor?.lat : a.coordinate === "longitude" ? lon : NaN;
    const w = d.companions?.wind_u && d.companions?.wind_v && this.windWanted();
    const wind = w ? {u: d.companions.wind_u.values, v: d.companions.wind_v.values, psi: d.companions.psi?.values,
      info: d.wind, timeMean: d.roles.time === "reduce", settings: this.windSettings()} : null;
    return {kind: "field", x: this.displayAxis(d.x), y: this.displayAxis(d.y), values: d.values, surface: d.surface,
      title, label, diverging, cursor: at(d.x), cursorY: at(d.y), range: this.fixedRange(), wind,
      selected: d.y.coordinate === "vertical" && d.y.representation === "sigma" && atmosphere ? this.selectedLevel() : null};
  }

  draw() {
    const c = this.c;
    if (!this.data || this.pending) return;
    this.loadRange();
    const width = Math.max(320, c.plot.parentElement.clientWidth - 4 || 900);
    this.width = this.node.clientWidth;
    this.spec = this.plotSpec();
    if (this.spec.kind === "value") {
      c.plot.hidden = true; this.layout = null;
      this.status(this.spec.title[0] + ": " + (this.spec.value === null ? "masked" : valueText(this.spec.value)) +
        " " + (this.data.presentation.units || ""));
      this.notes();
      return;
    }
    c.plot.hidden = false;
    c.plot.style.width = width + "px";
    const ratio = Math.min(devicePixelRatio, 2);
    this.layout = this.spec.kind === "field" ? drawField(c.plot, this.spec, width, ratio) : drawLine(c.plot, this.spec, width, ratio);
    c.plot.style.height = this.layout.height + "px";
    this.notes();
  }

  notes() {
    const d = this.data, c = this.c, rows = [...(d.notes || [])];
    const range = this.spec.range;
    if (range) {
      let below = 0, above = 0, total = 0;
      for (const row of this.spec.values) for (const v of row) {
        if (!Number.isFinite(v)) continue;
        total++; if (v < range[0]) below++; else if (v > range[1]) above++;
      }
      rows.push("Fixed colour range " + valueText(range[0]) + " to " + valueText(range[1]) + ": " + below + " of " + total +
        " plotted values lie below it and " + above + " above it, drawn at the end colours.");
    }
    for (const [name, part] of Object.entries(d.coordinates || {})) {
      if (part.role === "reduce" && part.rule) rows.push(NAMES[name] + " " + OP_NAMES[part.op || "mean"].toLowerCase() + ": " +
        part.rule + "." + (part.masked_excluded ? " " + part.masked_excluded + " masked samples were excluded." : ""));
    }
    const masked = JSON.stringify(d.values).includes("null");
    if (masked) rows.push("Blank cells are masked: the surface is below the terrain or outside the sampled levels " +
      "at that place, or at one of the saved samples of a mean (minimum and maximum use the valid samples only).");
    c.notes.replaceChildren(...rows.map(text => Object.assign(document.createElement("p"), {textContent: text})));
    if (d.table) {
      const head = c.table.tHead.rows[0], body = c.table.tBodies[0];
      head.replaceChildren(...d.table.columns.map(t => Object.assign(document.createElement("th"), {textContent: t})));
      body.replaceChildren(...d.table.rows.map((values, k) => {
        const row = document.createElement("tr");
        for (const v of values) row.append(Object.assign(document.createElement("td"),
          {textContent: typeof v === "number" ? numberText(v) : v ?? ""}));
        if (this.quantity.vertical?.domain === "atmosphere") {
          if (k === this.selectedLevel()) row.className = "selected";
          row.onclick = () => this.selectLevel(k);
        }
        return row;
      }));
      c.table.hidden = false;
    } else c.table.hidden = true;
  }

  locate(e) {
    const r = this.c.plot.getBoundingClientRect(), L = this.layout;
    if (!L) return null;
    const x = (e.clientX - r.left) * L.width / r.width, y = (e.clientY - r.top) * L.height / r.height;
    const box = L.box;
    if (x < box.x || x > box.x + box.w || y > box.y + box.h || y < box.y - (L.strip || 0)) return null;
    if (L.kind === "line") {
      let best = -1, distance = Infinity;
      L.points.forEach((p, k) => { if (!p) return; const dd = L.vertical ? Math.abs(p[1] - y) : Math.abs(p[0] - x); if (dd < distance) { distance = dd; best = k; } });
      return best < 0 ? null : {index: best};
    }
    const find = (edges, v) => {
      for (let i = 0; i < edges.length - 1; i++) {
        const lo = Math.min(edges[i], edges[i + 1]), hi = Math.max(edges[i], edges[i + 1]);
        if (v >= lo && v <= hi) return i;
      }
      return -1;
    };
    const xv = L.x0 + (x - box.x) / box.w * (L.x1 - L.x0);
    const column = find(L.xe, xv);
    if (y < box.y) return column < 0 ? null : {column, surface: true};
    const yv = L.up ? L.y1 - (y - box.y) / box.h * (L.y1 - L.y0) : L.y0 + (y - box.y) / box.h * (L.y1 - L.y0);
    const row = find(L.ye, yv);
    return column < 0 || row < 0 ? null : {column, row};
  }

  /** A coordinate value as text: days, degrees or the axis's own units. */
  coordText(a, k) {
    const v = a.values[k];
    if (a.ticks === "time") return dayText(v);
    if (a.coordinate === "latitude") return degrees(v, "N", "S");
    if (a.coordinate === "longitude") return numberText(v) + "°E";
    return (a.rows?.[k]?.label ? a.rows[k].label + " · " : "") + a.label.split(" (")[0] + " " + numberText(v) +
      (a.units ? " " + a.units : "");
  }

  hover(e) {
    if (this.pending || !this.data) return;
    const hit = this.locate(e), d = this.data, c = this.c;
    if (!hit) { c.readout.textContent = ""; return; }
    const p = d.presentation, parts = [], symbol = this.symbol();
    if (this.layout.kind === "line") {
      const k = hit.index;
      parts.push(this.coordText(this.spec.coordinate, k), symbol + " " + numberText(d.values[k]) + " " + p.units);
      if (d.table) parts.push(d.table.columns.map((col, i) => col + " " + (typeof d.table.rows[k + (d.table.rows.length - d.values.length)]?.[i] === "number" ?
        numberText(d.table.rows[k + (d.table.rows.length - d.values.length)][i]) : "")).slice(1, 4).join(" · "));
    } else {
      parts.push(this.coordText(d.x, hit.column));
      if (hit.surface) {
        parts.push("diagnosed skin (depth 0)", symbol + " " + numberText(d.surface[hit.column]) + " " + p.units);
      } else {
        const k = hit.row;
        parts.push(this.coordText(d.y, k));
        for (const companion of Object.values(d.companions || {})) {
          const v = Array.isArray(companion.values[k]) ? companion.values[k][hit.column] : companion.values[hit.column];
          if (Number.isFinite(v)) parts.push(companion.label + " " + numberText(v) + " " + companion.units);
        }
        const v = d.values[k][hit.column];
        parts.push(symbol + " " + (v === null ? "masked" : valueText(v)) + " " + p.units);
      }
    }
    c.readout.textContent = parts.filter(Boolean).join(" · ");
  }

  click(e) {
    if (this.pending || !this.data) return;
    const hit = this.locate(e);
    if (!hit) return;
    const d = this.data, atmosphere = this.quantity.vertical?.domain === "atmosphere" && d.quantity !== "mass_streamfunction";
    if (this.layout.kind === "field") {
      if (Number.isInteger(hit.row) && d.y.coordinate === "vertical" && atmosphere && this.c["link-level"].checked)
        this.selectLevel(hit.row);
      const at = {};
      for (const [a, k] of [["x", hit.column], ["y", hit.row]]) {
        if (!Number.isInteger(k)) continue;
        if (d[a].coordinate === "latitude") at.lat = d[a].values[k];
        if (d[a].coordinate === "longitude") at.lon = d[a].values[k];
      }
      if ("lat" in at || "lon" in at) this.links.onCursor({lat: at.lat ?? this.cursor?.lat ?? 0, lon: at.lon ?? this.cursor?.lon ?? 0});
    } else if (this.spec.coordinate.coordinate === "vertical" && atmosphere && this.c["link-level"].checked) {
      this.selectLevel(hit.index);
    }
  }

  png() {
    if (this.spec.kind === "value") return;
    const canvas = document.createElement("canvas");
    if (this.spec.kind === "field") drawField(canvas, this.spec, 1100, 1.5); else drawLine(canvas, this.spec, 1100, 1.5);
    canvas.toBlob(blob => {
      if (!blob) return;
      const url = URL.createObjectURL(blob), a = document.createElement("a");
      const t = this.data.coordinates?.time || {};
      a.href = url;
      a.download = (this.links.runName() + "-" + this.data.description + "-" + this.data.quantity + "-" +
        (t.role === "fixed" ? "t" + t.index : "t" + t.first + "-" + t.last) + ".png").replace(/[^a-z0-9._-]/gi, "_");
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 5000);
    }, "image/png");
  }
}
