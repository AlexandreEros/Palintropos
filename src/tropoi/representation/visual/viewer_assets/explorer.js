import {fieldScale} from "./mesh.js";
import {numberText} from "./map.js";

// The explorer: one quantity, four coordinates, one role each. The server
// says which role combinations each quantity supports (manifest.explorer)
// and evaluates them; this view only builds selections and draws results.
// Native levels and cells are drawn as bands between the edges the server
// sends, so nothing is drawn between levels that the archive does not hold.

const INK = "#22343a", GRID = "rgba(34,52,58,.18)", CURSOR = "#ff8c1a", LINE = "#106d6c";
const COORDINATES = ["longitude", "latitude", "vertical", "time"];
const NAMES = {longitude: "Longitude", latitude: "Latitude", vertical: "Vertical", time: "Time"};
const ROLE_NAMES = {x: "Horizontal axis (X)", y: "Vertical axis (Y)", fixed: "Fixed", reduce: "Mean (reduce)",
  none: "Not a coordinate of this quantity"};
const DIVERGING = /^(u|v|zeta|delta|phi|mass_streamfunction|conducted_flux)$/;
// Presets set roles only; the quantity decides whether they are offered.
const PRESETS = {
  column: {longitude: "fixed", latitude: "fixed", vertical: "y", time: "fixed"},
  meridional_section: {longitude: "fixed", latitude: "x", vertical: "y", time: "fixed"},
  zonal_section: {longitude: "x", latitude: "fixed", vertical: "y", time: "fixed"},
  zonal_mean: {longitude: "reduce", latitude: "x", vertical: "y", time: "fixed"},
  time_section: {longitude: "fixed", latitude: "fixed", vertical: "y", time: "x"},
  time_series: {longitude: "fixed", latitude: "fixed", vertical: "fixed", time: "x"},
  map: {longitude: "x", latitude: "y", vertical: "fixed", time: "fixed"},
  recorded_series: {longitude: "none", latitude: "none", vertical: "none", time: "x"},
};

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
  const height = headerHeight(ctx, plot.title, width) + plotHeight + strip + 132;
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
  ctx.strokeStyle = GRID; ctx.lineWidth = 1; ctx.font = "11px system-ui"; ctx.fillStyle = INK; ctx.beginPath();
  const yt = tickSet(ay, y0, y1), xt = tickSet(ax, x0, x1);
  for (const t of yt.ticks) {
    ctx.moveTo(box.x, Y(t)); ctx.lineTo(box.x + box.w, Y(t));
    ctx.textAlign = "right"; ctx.fillText(yt.label(t), box.x - 6, Y(t) + 4);
    if (ay.secondary) { ctx.textAlign = "left"; ctx.fillText(pressureText(ay.secondary.factor * 10 ** t), box.x + box.w + 6, Y(t) + 4); }
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
  if (Number.isFinite(plot.cursor)) {
    ctx.strokeStyle = CURSOR; ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]); ctx.beginPath();
    ctx.moveTo(X(plot.cursor), box.y); ctx.lineTo(X(plot.cursor), box.y + box.h); ctx.stroke(); ctx.setLineDash([]);
  }
  if (Number.isInteger(plot.selected) && plot.selected < rows) {
    ctx.strokeStyle = INK; ctx.lineWidth = 1.2;
    const p = Y(ye[plot.selected]), q = Y(ye[plot.selected + 1]);
    ctx.strokeRect(box.x, Math.min(p, q), box.w, Math.abs(q - p));
  }
  ctx.strokeStyle = INK; ctx.lineWidth = 1; ctx.strokeRect(box.x, box.y, box.w, box.h);
  colorbar(ctx, scale, {...box, y: box.y}, plot.range ? plot.label + " · fixed colour range; values beyond it take the end colours" : plot.label);
  return {kind: "field", box, width, height, xe, ye, X, Y, x0, x1, y0, y1, rows, cols, up, strip, range: scale.range};
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

export class Explorer {
  /** links: manifest, runName, timeText(axis, index), getMap() → {quantity, level, time}, showMap(selection),
   * onCursor(c), onLevel(k), getLevel(). */
  constructor(node, links) {
    this.node = node;
    this.links = links;
    this.c = Object.fromEntries([...node.querySelectorAll("[data-e]")].map(e => [e.dataset.e, e]));
    this.explorer = links.manifest.explorer;
    this.ranges = new Map();
    this.rows = {};
    const c = this.c, quantities = this.explorer.quantities;
    const keys = Object.keys(quantities).filter(k => Object.keys(quantities[k].products).some(p => p !== "map"));
    c.quantity.replaceChildren(...keys.map(k => option(k, this.quantityLabel(k))));
    const linked = links.getMap()?.quantity;
    c.quantity.value = keys.includes(linked) ? linked : keys.includes("temperature") ? "temperature" : keys[0];
    this.buildRows();
    // A new quantity keeps the roles when it supports them.
    c.quantity.onchange = () => {
      const product = this.product();
      this.applyPreset(product && product in this.quantity.products ? product : this.firstPreset());
      this.update();
    };
    c.preset.onchange = () => { if (c.preset.value) { this.applyPreset(c.preset.value); this.update(); } };
    c.snap.onchange = () => this.update();
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
    const requested = new URLSearchParams(location.search).get("product");
    this.applyPreset(requested && requested in this.quantity.products ? requested : this.firstPreset());
    node.hidden = false;
  }

  get quantity() { return this.explorer.quantities[this.c.quantity.value]; }

  quantityLabel(key) {
    const p = this.explorer.quantities[key].presentation;
    return [p.symbol ? p.name + " (" + p.symbol + ")" : p.name, p.units].filter(Boolean).join(" · ");
  }

  firstPreset() {
    const products = this.quantity.products;
    return ["column", "time_series", "recorded_series", "meridional_section", "zonal_mean", "time_section"]
      .find(p => p in products) || Object.keys(products)[0];
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
      if (name === "longitude" || name === "latitude") {
        parts.value = Object.assign(document.createElement("input"), {type: "number", step: "any"});
        parts.value.setAttribute("aria-label", NAMES[name] + " (degrees " + (name === "latitude" ? "north" : "east") + ")");
        if (name === "latitude") { parts.value.min = "-90"; parts.value.max = "90"; }
        parts.value.onchange = () => {
          const lat = this.rows.latitude.value.valueAsNumber, lon = this.rows.longitude.value.valueAsNumber;
          if (Number.isFinite(lat) && Number.isFinite(lon)) this.links.onCursor({lat, lon});
        };
        row.append(parts.value);
      } else if (name === "vertical") {
        parts.representation = document.createElement("select");
        parts.representation.setAttribute("aria-label", "Vertical representation");
        parts.level = document.createElement("select");
        parts.level.setAttribute("aria-label", "Native level");
        parts.value = Object.assign(document.createElement("input"), {type: "number", step: "any"});
        parts.value.setAttribute("aria-label", "Vertical value");
        parts.representation.onchange = () => { this.syncVertical(); this.update(); };
        parts.level.onchange = parts.value.onchange = () => this.update();
        row.append(parts.representation, parts.level, parts.value);
      } else {
        parts.follow = Object.assign(document.createElement("label"), {className: "check"});
        parts.followBox = Object.assign(document.createElement("input"), {type: "checkbox", checked: true});
        parts.follow.append(parts.followBox, " Follow selection A");
        parts.index = document.createElement("select");
        parts.index.setAttribute("aria-label", "Saved time");
        parts.first = document.createElement("select");
        parts.first.setAttribute("aria-label", "First saved time");
        parts.last = document.createElement("select");
        parts.last.setAttribute("aria-label", "Last saved time");
        parts.followBox.onchange = () => this.update();
        parts.index.onchange = () => { parts.followBox.checked = false; this.update(); };
        parts.first.onchange = parts.last.onchange = () => this.update();
        row.append(parts.follow, parts.index, parts.first, parts.last);
      }
      role.onchange = () => { this.syncRows(); this.update(); };
      box.append(row);
      this.rows[name] = parts;
    }
  }

  /** Roles chosen in the rows. */
  roles() { return Object.fromEntries(COORDINATES.map(n => [n, this.rows[n].role.value])); }

  /** The product the roles name (the same table as the server's), or null. */
  product() {
    const roles = this.roles();
    for (const [name, spec] of Object.entries(this.explorer.products)) {
      if (COORDINATES.every(c => {
        const want = spec.roles[c] ?? "none";
        return (Array.isArray(want) ? want : [want]).includes(roles[c]);
      })) return name;
    }
    return null;
  }

  applyPreset(name) {
    const roles = PRESETS[name] || {};
    const q = this.quantity;
    for (const coordinate of COORDINATES) {
      let role = roles[coordinate] || "fixed";
      if (coordinate === "vertical" && !q.vertical) role = "none";
      if ((coordinate === "longitude" || coordinate === "latitude") && q.horizontal === "none") role = "none";
      if (coordinate === "time" && q.horizontal === "none") role = "x";
      this.fillRoles(coordinate, role);
    }
    this.syncRows();
  }

  fillRoles(coordinate, role) {
    const q = this.quantity, select = this.rows[coordinate].role;
    let roles = ["x", "y", "fixed"];
    if (coordinate === "longitude" || coordinate === "time") roles.push("reduce");
    if ((coordinate === "vertical" && !q.vertical) || ((coordinate === "longitude" || coordinate === "latitude") && q.horizontal === "none"))
      roles = ["none"];
    select.replaceChildren(...roles.map(r => option(r, coordinate === "time" && r === "fixed" ? "Fixed (one saved time)" : ROLE_NAMES[r])));
    select.value = roles.includes(role) ? role : roles[0];
  }

  /** Time axis of the selection: a fixed level's own axis, else the quantity's. */
  timeAxis() {
    const q = this.quantity, v = this.rows.vertical;
    if (q.vertical && v.role.value === "fixed") {
      const level = this.level();
      if (level?.time_axis) return level.time_axis;
    }
    return q.time_axis;
  }

  level() {
    const q = this.quantity, v = this.rows.vertical;
    if (!q.vertical) return null;
    return q.vertical.levels.find(l => String(l.index) === v.level.value) || null;
  }

  discrete(rep) {
    // Native levels, and every representation of a substrate column (native depths only).
    return rep === "native" || this.quantity.vertical?.domain === "substrate";
  }

  syncRows() {
    const q = this.quantity, c = this.c;
    c.classification.textContent = q.presentation.role + " · " + q.presentation.origin;
    c.definition.textContent = q.presentation.definition + (q.evaluation ? " Evaluation: " + q.evaluation + "." : "");
    for (const name of ["longitude", "latitude"]) {
      const r = this.rows[name];
      r.value.hidden = r.role.value !== "fixed";
    }
    this.syncVertical();
    const t = this.rows.time, role = t.role.value;
    const axis = this.links.manifest.time_axes[this.timeAxis()];
    const values = axis?.values || [];
    const fill = select => {
      const old = select.value;
      if (select.dataset.axis !== this.timeAxis()) {
        select.replaceChildren(...values.map((s, i) => option(i, this.links.timeText(this.timeAxis(), i))));
        select.dataset.axis = this.timeAxis();
        // Atmospheric histories synthesize a column per saved state: default to the last 60.
        const span = this.quantity.vertical?.domain === "atmosphere" ? 60 : values.length;
        select.value = select === t.last ? String(values.length - 1) :
          select === t.first ? String(Math.max(0, values.length - span)) : "0";
      } else select.value = old;
    };
    [t.index, t.first, t.last].forEach(fill);
    t.follow.hidden = role !== "fixed" || !this.links.getMap();
    t.index.hidden = role !== "fixed";
    t.first.hidden = t.last.hidden = !["x", "reduce"].includes(role) || !values.length;
    const product = this.product();
    c.preset.replaceChildren(option("", product ? "Custom roles" : "Choose a preset"),
      ...Object.entries(this.explorer.products).map(([key, p]) =>
        option(key, p.label + (key in q.products ? "" : " (not available)"), !(key in q.products))));
    c.preset.value = product && product in q.products ? product : "";
    c["snap-wrap"].hidden = !(product && ["column", "time_series", "time_section"].includes(product) &&
      this.links.manifest.geometry?.grid === "latlon" && q.vertical?.domain === "atmosphere");
  }

  syncVertical() {
    const q = this.quantity, v = this.rows.vertical, role = v.role.value;
    const product = this.product(), support = q.products[product];
    const shown = Boolean(q.vertical) && role !== "none";
    v.representation.hidden = !shown;
    if (!shown) { v.level.hidden = v.value.hidden = true; return; }
    const reps = support?.vertical || Object.keys(q.vertical.representations);
    const old = v.representation.value;
    v.representation.replaceChildren(...reps.map(r => option(r, q.vertical.representations[r].label)));
    if (reps.includes(old)) v.representation.value = old;
    const rep = v.representation.value, fixed = role === "fixed";
    const discrete = this.discrete(rep);
    v.level.hidden = !fixed || !discrete;
    v.value.hidden = !fixed || discrete;
    if (fixed && discrete) {
      const keep = v.level.value;
      v.level.replaceChildren(...q.vertical.levels.map(l => option(l.index,
        rep === "native" ? l.label : (q.vertical.representations[rep].label.split(" (")[0] + " = " +
          numberText(l[rep]) + " m" + (l.kind === "surface" ? " · diagnosed skin" : "")))));
      const linked = this.links.getLevel?.();
      v.level.value = q.vertical.levels.some(l => String(l.index) === keep) ? keep :
        String(linked ?? q.vertical.levels[q.vertical.domain === "substrate" ? 0 : q.vertical.levels.length - 1].index);
    }
    if (fixed && !discrete && v.value.value === "") v.value.value = rep === "sigma" ? "0.5" : rep === "pressure" ? "50000" : "1000";
  }

  /** The selection the rows describe, or a reason why there is none. */
  selection() {
    const q = this.quantity, product = this.product();
    if (!product) return {error: "This combination of roles has no plot. Two plotting axes give a map or section, one a line."};
    if (!(product in q.products)) {
      const reason = q.unavailable?.[product] || this.explorer.unavailable?.[product];
      return {error: this.explorer.products[product].label + " is not available for " + q.presentation.name.toLowerCase() +
        (reason ? ": " + reason : ".")};
    }
    // Snapping to a native node is offered only on Gaussian grids; elsewhere the point is exact.
    const sel = {quantity: this.c.quantity.value, options: {snap: !this.c["snap-wrap"].hidden && this.c.snap.checked}};
    for (const name of ["longitude", "latitude"]) {
      const r = this.rows[name], role = r.role.value;
      sel[name] = {role};
      if (role === "fixed") {
        if (!r.value.validity.valid || r.value.value === "") return {error: "Enter a valid " + name + "."};
        sel[name].value = r.value.valueAsNumber;
      }
      if (role === "reduce") sel[name].op = "mean";
    }
    const v = this.rows.vertical;
    sel.vertical = {role: v.role.value};
    if (q.vertical) {
      const rep = v.representation.value;
      sel.vertical.representation = rep;
      if (v.role.value === "fixed") {
        if (this.discrete(rep)) {
          const level = this.level();
          sel.vertical.value = rep === "native" ? level.index : level[rep];
        } else {
          if (!v.value.validity.valid || !Number.isFinite(v.value.valueAsNumber)) return {error: "Enter a valid vertical value."};
          sel.vertical.value = v.value.valueAsNumber;
        }
      }
    }
    const t = this.rows.time, role = t.role.value;
    sel.time = {role};
    const map = this.links.getMap();
    if (role === "fixed") {
      let index = Number(t.index.value);
      if (!t.follow.hidden && t.followBox.checked && map && map.axis === this.timeAxis()) {
        index = map.time; t.index.value = String(index);
      } else if (!t.follow.hidden && t.followBox.checked && map) {
        // Different cadences: the latest sample of this axis not after selection A's time.
        const values = this.links.manifest.time_axes[this.timeAxis()].values;
        const target = this.links.manifest.time_axes[map.axis].values[map.time];
        let k = values.findIndex(s => s > target);
        index = k < 0 ? values.length - 1 : Math.max(0, k - 1);
        t.index.value = String(index);
      }
      sel.time.index = index;
    } else if (role === "x" || role === "reduce") {
      let first = Number(t.first.value), last = Number(t.last.value);
      if (last < first) [first, last] = [last, first];
      sel.time.first = first; sel.time.last = last;
      if (role === "reduce") sel.time.op = "mean";
    }
    return {selection: sel, product};
  }

  setCursor(cursor) {
    this.cursor = cursor;
    this.rows.latitude.value.value = String(Number(cursor.lat.toFixed(4)));
    this.rows.longitude.value.value = String(Number((((cursor.lon % 360) + 360) % 360).toFixed(4)));
    return this.update(false);
  }

  /** Selection A changed: follow it, but never push a map back to it. */
  follow() { return this.update(false); }

  get rangeKey() {
    const p = this.data?.product;
    return p && ["meridional_section", "zonal_section", "zonal_mean", "time_section"].includes(p) ? this.data.quantity : null;
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

  /** ``push``: a map selection made here is sent to selection A (explorer controls only). */
  async update(push = true) {
    this.syncRows();
    const {selection, product, error} = this.selection();
    const c = this.c;
    c.product.textContent = product ? this.explorer.products[product].label : "No plot for these roles";
    if (error) {
      this.abort?.abort(); this.pending = null; this.data = null; this.loaded = null;
      c.plot.hidden = true; c.table.hidden = true; c.notes.replaceChildren(); c.png.disabled = true;
      this.status(error, true);
      return;
    }
    if (product === "map") {
      this.status("A longitude × latitude selection is drawn on the globe and map of selection A.");
      c.plot.hidden = true; c.table.hidden = true; c.notes.replaceChildren(); c.png.disabled = true;
      this.data = null;
      if (push) this.links.showMap(selection);
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
      c.plot.hidden = true;
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
    const d = this.data, t = d.time;
    if (!t) return "";
    if ("index" in t) return this.links.timeText(t.axis, t.index) + " · one saved sample";
    const values = this.links.manifest.time_axes[t.axis]?.values;
    const span = values ? dayText(values[t.first]) + " to " + dayText(values[t.last]) : "";
    if (d.sample) return d.sample.count === 1 ? "single saved state" :
      `equal-weight mean of ${d.sample.count} saved states, ${span} (${d.sample.cadence} cadence)`;
    return `${t.count} saved samples, ${span}`;
  }

  where() {
    const l = this.data.location;
    if (!l) return "";
    if ("latitude" in l && "longitude" in l) return place({lat: l.latitude, lon: l.longitude}) +
      (l.column !== undefined ? " (column " + l.column + ")" : "") + " · " + l.sampling;
    if ("latitude" in l) return "at " + degrees(l.latitude, "N", "S") + (l.sampling ? " · " + l.sampling : "");
    if ("longitude" in l) return "at " + numberText(l.longitude) + "°E" + (l.sampling ? " · " + l.sampling : "");
    return "";
  }

  plotSpec() {
    const d = this.data, p = d.presentation;
    const label = (p.symbol ? p.name + " (" + p.symbol + ")" : p.name) + (p.units ? " · " + p.units : "");
    const title = [this.links.runName() + " · " + this.explorer.products[d.product].label, this.identity(),
      [this.where(), this.timeText()].filter(Boolean).join(" · "), d.sampling];
    const diverging = DIVERGING.test(d.quantity);
    if (d.product === "column") {
      return {kind: "line", orientation: "profile", coordinate: d.y, values: d.values, markers: d.markers, title,
        valueLabel: label, diverging, surfaceLine: this.quantity.vertical?.domain === "substrate", selected: this.links.getLevel?.()};
    }
    if (d.product === "time_series" || d.product === "recorded_series") {
      return {kind: "line", orientation: "series", coordinate: d.x, values: d.values, title: [...title, d.level || ""],
        valueLabel: label, diverging};
    }
    const cursor = d.x.coordinate === "latitude" ? this.cursor?.lat :
      d.x.coordinate === "longitude" ? ((this.cursor?.lon % 360) + 360) % 360 : NaN;
    return {kind: "field", x: d.x, y: d.y, values: d.values, surface: d.surface, title, label, diverging, cursor,
      range: this.fixedRange(), selected: d.y.representation === "sigma" && this.quantity.vertical?.domain === "atmosphere" &&
        d.quantity !== "mass_streamfunction" ? this.links.getLevel?.() : null};
  }

  draw() {
    const c = this.c;
    if (!this.data || this.pending) return;
    this.loadRange();
    const width = Math.max(320, c.plot.parentElement.clientWidth - 4 || 900);
    this.width = this.node.clientWidth;
    this.spec = this.plotSpec();
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
    rows.push("Saved samples only: variations between saved samples are not represented.");
    c.notes.replaceChildren(...rows.map(text => Object.assign(document.createElement("p"), {textContent: text})));
    if (d.table) {
      const head = c.table.tHead.rows[0], body = c.table.tBodies[0];
      head.replaceChildren(...d.table.columns.map(t => Object.assign(document.createElement("th"), {textContent: t})));
      body.replaceChildren(...d.table.rows.map((values, k) => {
        const row = document.createElement("tr");
        for (const v of values) row.append(Object.assign(document.createElement("td"), {textContent: typeof v === "number" ? numberText(v) : v}));
        if (this.quantity.vertical?.domain === "atmosphere") {
          if (k === this.links.getLevel?.()) row.className = "selected";
          row.onclick = () => this.links.onLevel(k);
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

  hover(e) {
    if (this.pending || !this.data) return;
    const hit = this.locate(e), d = this.data, c = this.c;
    if (!hit) { c.readout.textContent = ""; return; }
    const p = d.presentation, parts = [];
    if (this.layout.kind === "line") {
      const k = hit.index, a = this.spec.coordinate;
      const coord = a.ticks === "time" ? dayText(a.values[k]) + " (" + numberText(a.values[k]) + " s)" :
        a.label.split(" (")[0] + " " + numberText(a.values[k]) + (a.units ? " " + a.units : "");
      parts.push(coord, (p.symbol || p.name) + " " + numberText(d.values[k]) + " " + p.units);
      if (d.table) parts.push(d.table.columns.map((col, i) => col + " " + (typeof d.table.rows[k + (d.table.rows.length - d.values.length)]?.[i] === "number" ?
        numberText(d.table.rows[k + (d.table.rows.length - d.values.length)][i]) : "")).slice(1, 4).join(" · "));
    } else {
      const xv = d.x.values[hit.column];
      parts.push(d.x.ticks === "time" ? dayText(xv) : d.x.coordinate === "latitude" ? degrees(xv, "N", "S") : numberText(xv) + "°E");
      if (hit.surface) {
        parts.push("diagnosed skin (depth 0)", (p.symbol || p.name) + " " + numberText(d.surface[hit.column]) + " " + p.units);
      } else {
        const k = hit.row, yv = d.y.values[k];
        parts.push((d.y.rows?.[k]?.label || "row " + (k + 1)) + " · " + d.y.label.split(" (")[0] + " " + numberText(yv));
        for (const companion of Object.values(d.companions || {})) {
          const v = companion.values[k]?.[hit.column] ?? companion.values[hit.column];
          if (Number.isFinite(v)) parts.push(companion.label + " " + numberText(v) + " " + companion.units);
        }
        parts.push((d.product === "zonal_mean" ? "[" + (p.symbol || p.name) + "]" : p.symbol || p.name) + " " +
          valueText(d.values[k][hit.column]) + " " + p.units);
      }
    }
    c.readout.textContent = parts.filter(Boolean).join(" · ");
  }

  click(e) {
    if (this.pending || !this.data) return;
    const hit = this.locate(e);
    if (!hit) return;
    const d = this.data;
    if (this.layout.kind === "field") {
      if (Number.isInteger(hit.row) && this.quantity.vertical?.domain === "atmosphere" && d.quantity !== "mass_streamfunction" &&
          this.c["link-level"].checked) this.links.onLevel(hit.row);
      const xv = d.x.values[hit.column];
      if (d.x.coordinate === "latitude") this.links.onCursor({lat: xv, lon: this.cursor?.lon ?? 0});
      if (d.x.coordinate === "longitude") this.links.onCursor({lat: this.cursor?.lat ?? 0, lon: xv});
    } else if (d.product === "column" && this.quantity.vertical?.domain === "atmosphere" && this.c["link-level"].checked) {
      this.links.onLevel(hit.index);
    }
  }

  png() {
    const canvas = document.createElement("canvas");
    if (this.spec.kind === "field") drawField(canvas, this.spec, 1100, 1.5); else drawLine(canvas, this.spec, 1100, 1.5);
    canvas.toBlob(blob => {
      if (!blob) return;
      const url = URL.createObjectURL(blob), a = document.createElement("a");
      const t = this.data.time || {};
      a.href = url;
      a.download = (this.links.runName() + "-" + this.data.product + "-" + this.data.quantity + "-" +
        ("index" in t ? "t" + t.index : "t" + t.first + "-" + t.last) + ".png").replace(/[^a-z0-9._-]/gi, "_");
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 5000);
    }, "image/png");
  }
}
