import {fieldScale} from "./mesh.js";
import {numberText} from "./map.js";

// Linked atmospheric structure: a column at the geographic cursor, sections
// through it and native-grid zonal reductions. The provider evaluates every
// value. Native full levels are drawn as bands between geometric-mean σ
// edges, so the plot never invents values between levels.

const PRODUCTS = {
  column: {name: "Column profile", op: "column"},
  meridional: {name: "Meridional section · latitude × σ at the cursor longitude", op: "meridional_section"},
  zonal: {name: "Zonal section · longitude × σ at the cursor latitude", op: "zonal_section"},
  zonal_mean: {name: "Zonal mean · latitude × σ", op: "zonal_mean"},
  streamfunction: {name: "Meridional mass streamfunction · latitude × σ", op: "mass_streamfunction"},
};
const FIELDS = {
  temperature: {name: "Air temperature", symbol: "T", units: "K"},
  u: {name: "Eastward wind", symbol: "u", units: "m s⁻¹", diverging: true},
  v: {name: "Northward wind", symbol: "v", units: "m s⁻¹", diverging: true},
  pressure: {name: "Air pressure", symbol: "p", units: "Pa"},
  geopotential_height: {name: "Geopotential height", symbol: "Z", units: "m"},
};
const PSI = {name: "Meridional mass streamfunction", symbol: "ψ", units: "kg s⁻¹", diverging: true};
const INK = "#22343a", GRID = "rgba(34,52,58,.18)", CURSOR = "#ff8c1a";

const degrees = (value, positive, negative) =>
  Math.abs(value).toFixed(2) + "°" + (value < 0 ? negative : positive);
// Archives use east longitudes 0–360; the signed form is added for reading maps.
export const place = c => {
  const east = ((c.lon % 360) + 360) % 360, signed = ((c.lon + 540) % 360) - 180;
  return degrees(c.lat, "N", "S") + ", " + numberText(Number(east.toFixed(4))) + "°E" +
    (signed < 0 ? " (" + degrees(signed, "E", "W") + ")" : "");
};
// Large magnitudes (ψ in kg s⁻¹) read better in exponent form.
const valueText = v => Number.isFinite(v) && v !== 0 && (Math.abs(v) >= 1e6 || Math.abs(v) < 1e-3) ?
  v.toExponential(3).replace("e+", "e") : numberText(v);
// Rounded axis ticks: 1, 2 or 5 times a power of ten.
function niceTicks(lo, hi, count = 5) {
  const raw = (hi - lo) / count, power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map(f => f * power).find(s => s >= raw);
  const ticks = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-9; t += step) ticks.push(Number(t.toPrecision(12)));
  return ticks;
}
const pressureText = p => p >= 100 ? numberText(p / 100) + " hPa" : numberText(p) + " Pa";

// Band edges in log10 σ for full levels, or for interfaces when the top one is σ = 0.
function bands(centres) {
  const top = centres[0] > 0 ? centres[0] : centres[1] * centres[1] / centres[2];
  const c = [top, ...centres.slice(1)];
  const log = c.map(Math.log10);
  const edges = [log[0] - (log[1] - log[0]) / 2];
  for (let k = 0; k < log.length - 1; k++) edges.push((log[k] + log[k + 1]) / 2);
  edges.push(Math.min(0, log[log.length - 1] + (log[log.length - 1] - log[log.length - 2]) / 2));
  return {log, edges};
}

function wrapped(ctx, text, width) {
  const lines = [];
  let current = "";
  for (let word of String(text).split(/\s+/)) {
    if (current && ctx.measureText(current + " " + word).width > width) { lines.push(current); current = ""; }
    // Break an archive name or hash longer than the line itself.
    while (!current && ctx.measureText(word).width > width) {
      let n = word.length - 1;
      while (n > 1 && ctx.measureText(word.slice(0, n)).width > width) n--;
      lines.push(word.slice(0, n)); word = word.slice(n);
    }
    current = current ? current + " " + word : word;
  }
  if (current) lines.push(current);
  return lines;
}

function headerHeight(ctx, lines, width) {
  ctx.font = "12px system-ui";
  return 26 + lines.slice(1).reduce((n, line) => n + 17 * wrapped(ctx, line, width - 28).length, 0) + 6;
}

function header(ctx, lines, width) {
  ctx.fillStyle = "#fffef9"; ctx.fillRect(0, 0, width, 9999);
  ctx.textAlign = "left"; ctx.fillStyle = INK;
  ctx.font = "600 14px system-ui"; ctx.fillText(lines[0], 14, 20, width - 28);
  let y = 40;
  ctx.font = "12px system-ui";
  for (const line of lines.slice(1)) for (const part of wrapped(ctx, line, width - 28)) { ctx.fillText(part, 14, y); y += 17; }
  return y - 11;
}

function colorbar(ctx, scale, box, label) {
  const y = box.y + box.h + 62;
  ctx.font = "600 12px system-ui"; ctx.fillStyle = INK; ctx.textAlign = "left";
  ctx.fillText(label, box.x, y - 7, box.w);
  if (!scale.range) return;
  for (let x = 0; x < box.w; x++) {
    const [r, g, b] = scale.lut[Math.round(255 * x / Math.max(1, box.w - 1))];
    ctx.fillStyle = `rgb(${r},${g},${b})`; ctx.fillRect(box.x + x, y, 1.1, 9);
  }
  ctx.font = "11px system-ui"; ctx.fillStyle = INK;
  ctx.fillText(valueText(scale.range[0]), box.x, y + 23);
  ctx.textAlign = "right"; ctx.fillText(valueText(scale.range[1]), box.x + box.w, y + 23);
}

/** Latitude or longitude × σ cells. Returns the layout used for hit testing. */
function drawSection(canvas, plot, width, ratio) {
  const ctx = canvas.getContext("2d");
  const rows = plot.values.length, cols = plot.x.length;
  const {log, edges} = bands(plot.levels);
  const flat = new Float32Array(rows * cols);
  plot.values.forEach((row, k) => flat.set(row, k * cols));
  const scale = fieldScale({values: flat}, {field: plot.diverging ? "u" : "", range: plot.range});
  const plotHeight = Math.round(Math.min(420, Math.max(220, width * 0.4)));
  const height = headerHeight(ctx, plot.title, width) + plotHeight + 128;
  canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  const top = header(ctx, plot.title, width);
  const box = {x: 66, y: top + 4, w: width - 66 - (plot.reference ? 92 : 24), h: plotHeight};
  const [x0, x1] = plot.xRange;
  const X = x => box.x + box.w * (x - x0) / (x1 - x0);
  const Y = l => box.y + box.h * (l - edges[0]) / (edges[edges.length - 1] - edges[0]);
  const xEdges = plot.x.map((x, i) => i ? (x + plot.x[i - 1]) / 2 : x0);
  xEdges.push(x1);
  for (let k = 0; k < rows; k++) for (let i = 0; i < cols; i++) {
    const [r, g, b] = scale.color(plot.values[k][i]);
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    const left = X(Math.max(x0, xEdges[i])), right = X(Math.min(x1, xEdges[i + 1]));
    ctx.fillRect(left, Y(edges[k]), right - left + 0.6, Y(edges[k + 1]) - Y(edges[k]) + 0.6);
  }
  ctx.strokeStyle = GRID; ctx.lineWidth = 1; ctx.font = "11px system-ui"; ctx.fillStyle = INK;
  ctx.beginPath();
  for (let d = Math.ceil(edges[0]); d <= 0; d++) {
    ctx.moveTo(box.x, Y(d)); ctx.lineTo(box.x + box.w, Y(d));
    ctx.textAlign = "right"; ctx.fillText("10" + (d ? superscript(d) : "⁰"), box.x - 6, Y(d) + 4);
    if (plot.reference) { ctx.textAlign = "left"; ctx.fillText(pressureText(plot.reference * 10 ** d), box.x + box.w + 6, Y(d) + 4); }
  }
  for (const t of plot.ticks) { ctx.moveTo(X(t), box.y); ctx.lineTo(X(t), box.y + box.h); }
  ctx.stroke();
  ctx.textAlign = "center";
  for (const t of plot.ticks) ctx.fillText(plot.tickLabel(t), X(t), box.y + box.h + 14);
  ctx.fillText(plot.xLabel, box.x + box.w / 2, box.y + box.h + 28);
  ctx.save(); ctx.translate(16, box.y + box.h / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText(plot.yLabel, 0, 0); ctx.restore();
  if (plot.reference) {
    ctx.save(); ctx.translate(box.x + box.w + 80, box.y + box.h / 2); ctx.rotate(Math.PI / 2);
    ctx.fillText("nominal p ≈ σ × p_ref", 0, 0); ctx.restore();
  }
  ctx.strokeStyle = CURSOR; ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]); ctx.beginPath();
  if (Number.isFinite(plot.cursor)) { ctx.moveTo(X(plot.cursor), box.y); ctx.lineTo(X(plot.cursor), box.y + box.h); }
  ctx.stroke(); ctx.setLineDash([]);
  if (Number.isInteger(plot.selected) && plot.selected < rows) {
    ctx.strokeStyle = INK; ctx.lineWidth = 1.2;
    ctx.strokeRect(box.x, Y(edges[plot.selected]), box.w, Y(edges[plot.selected + 1]) - Y(edges[plot.selected]));
  }
  ctx.strokeStyle = INK; ctx.strokeRect(box.x, box.y, box.w, box.h);
  colorbar(ctx, scale, box, plot.range ? plot.label + " · fixed colour range; values beyond it take the end colours" : plot.label);
  return {box, width, height, edges, xEdges, X, Y, rows, cols, kind: "section", log, x0, x1, range: scale.range};
}

function superscript(n) {
  const digits = "⁰¹²³⁴⁵⁶⁷⁸⁹";
  return (n < 0 ? "⁻" : "") + String(Math.abs(n)).split("").map(d => digits[d]).join("");
}

/** One quantity against height, pressure or σ, with the surface marked separately. */
function drawProfile(canvas, plot, width, ratio) {
  const ctx = canvas.getContext("2d");
  const plotHeight = Math.round(Math.min(480, Math.max(260, width * 0.5)));
  const legend = plot.markers.length * 18;
  const height = headerHeight(ctx, plot.title, width) + legend + plotHeight + 56;
  canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  let top = header(ctx, plot.title, width);
  ctx.font = "12px system-ui";
  for (const m of plot.markers) {
    ctx.fillStyle = m.color; ctx.strokeStyle = INK; ctx.lineWidth = 1;
    if (m.shape === "square") { ctx.fillRect(20, top + 2, 10, 10); ctx.strokeRect(20, top + 2, 10, 10); }
    else { ctx.beginPath(); ctx.arc(25, top + 7, 5.5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke(); }
    ctx.fillStyle = INK; ctx.textAlign = "left"; ctx.fillText(m.label, 38, top + 11, width - 52);
    top += 18;
  }
  const box = {x: 92, y: top + 8, w: width - 120, h: plotHeight};
  const xs = plot.points.map(p => p.x).concat(plot.markers.map(m => m.x));
  let [lo, hi] = [Math.min(...xs), Math.max(...xs)];
  if (plot.zero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
  if (lo === hi) { lo -= 1; hi += 1; }
  const pad = (hi - lo) * 0.06; lo -= pad; hi += pad;
  const ys = plot.points.map(p => p.y).concat(plot.markers.map(m => m.y));
  let [ylo, yhi] = [Math.min(...ys), Math.max(...ys)];
  if (ylo === yhi) { ylo -= 1; yhi += 1; }
  const X = x => box.x + box.w * (x - lo) / (hi - lo);
  const Y = y => box.y + box.h * (1 - (y - ylo) / (yhi - ylo));
  ctx.strokeStyle = GRID; ctx.fillStyle = INK; ctx.font = "11px system-ui"; ctx.beginPath();
  for (const x of niceTicks(lo, hi)) {
    ctx.moveTo(X(x), box.y); ctx.lineTo(X(x), box.y + box.h);
    ctx.textAlign = "center"; ctx.fillText(valueText(x), X(x), box.y + box.h + 14);
  }
  for (const y of niceTicks(ylo, yhi)) {
    ctx.moveTo(box.x, Y(y)); ctx.lineTo(box.x + box.w, Y(y));
    ctx.textAlign = "right"; ctx.fillText(plot.yTick(y), box.x - 6, Y(y) + 4);
  }
  ctx.stroke();
  if (plot.zero && lo < 0 && hi > 0) {
    ctx.strokeStyle = "rgba(34,52,58,.55)"; ctx.beginPath(); ctx.moveTo(X(0), box.y); ctx.lineTo(X(0), box.y + box.h); ctx.stroke();
  }
  ctx.textAlign = "center"; ctx.fillText(plot.label, box.x + box.w / 2, box.y + box.h + 32);
  ctx.save(); ctx.translate(14, box.y + box.h / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(plot.yLabel, 0, 0); ctx.restore();
  ctx.strokeStyle = "#106d6c"; ctx.lineWidth = 2; ctx.beginPath();
  plot.points.forEach((p, i) => i ? ctx.lineTo(X(p.x), Y(p.y)) : ctx.moveTo(X(p.x), Y(p.y)));
  ctx.stroke();
  plot.points.forEach((p, k) => {
    ctx.fillStyle = k === plot.selected ? CURSOR : "#106d6c";
    ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), k === plot.selected ? 4.5 : 2.6, 0, 2 * Math.PI); ctx.fill();
  });
  for (const m of plot.markers) {
    ctx.fillStyle = m.color; ctx.strokeStyle = INK; ctx.lineWidth = 1;
    if (m.shape === "square") { ctx.fillRect(X(m.x) - 5, Y(m.y) - 5, 10, 10); ctx.strokeRect(X(m.x) - 5, Y(m.y) - 5, 10, 10); }
    else { ctx.beginPath(); ctx.arc(X(m.x), Y(m.y), 5.5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke(); }
  }
  ctx.strokeStyle = INK; ctx.lineWidth = 1; ctx.strokeRect(box.x, box.y, box.w, box.h);
  return {box, width, height, X, Y, kind: "profile", points: plot.points.map(p => [X(p.x), Y(p.y)])};
}

export class StructureView {
  /** links: manifest, runName, timeLabel, getTime(), getField(), onCursor(c), onLevel(k), getLevel(). */
  constructor(node, links) {
    this.node = node;
    this.links = links;
    this.c = Object.fromEntries([...node.querySelectorAll("[data-s]")].map(e => [e.dataset.s, e]));
    const {manifest} = links, c = this.c, ops = manifest.operations.structure;
    const times = manifest.time_axes.snapshots.values;
    c.product.replaceChildren(...Object.entries(PRODUCTS).filter(([, p]) => ops.includes(p.op))
      .map(([key, p]) => new Option(p.name, key)));
    const stateTimes = times.map((t, i) => new Option(links.timeLabel(i), i));
    c.first.replaceChildren(...stateTimes);
    c.last.replaceChildren(...stateTimes.map(o => new Option(o.text, o.value)));
    c.first.value = c.last.value = String(links.getTime());
    const steps = new Set(times.slice(1).map((t, i) => t - times[i]));
    this.cadence = times.length < 2 ? "one saved state" : steps.size === 1 ?
      "saved every " + numberText([...steps][0] / 86400) + " d" : "irregularly saved states";
    for (const key of ["product", "field", "axis", "first", "last"]) c[key].onchange = () => {
      if (key === "first" || key === "last") c.follow.checked = false;
      this.update();
    };
    c.follow.onchange = () => this.update();
    c.snap.onchange = () => this.update();
    c.lat.onchange = c.lon.onchange = () => {
      if (c.lat.validity.valid && c.lon.validity.valid && c.lat.value !== "" && c.lon.value !== "")
        links.onCursor({lat: c.lat.valueAsNumber, lon: c.lon.valueAsNumber});
    };
    // Fixed section colour ranges, kept per plotted quantity (ψ or the field key)
    // so stepping through saved times keeps one scale. The map has its own.
    this.ranges = new Map();
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
    node.hidden = false;
  }

  get product() { return this.c.product.value; }

  /** The quantity a colour range belongs to. Columns have no colour scale. */
  get rangeKey() {
    return this.product === "column" ? null : this.product === "streamfunction" ? "psi" : this.c.field.value;
  }

  storeRange() {
    const c = this.c, key = this.rangeKey;
    if (!key) return;
    this.ranges.set(key, {mode: c["range-mode"].value, min: c["range-min"].value, max: c["range-max"].value});
    this.showRange();
    this.draw();
  }

  /** Show the stored range of the plotted quantity; quantities without one are automatic. */
  loadRange() {
    const c = this.c, key = this.rangeKey;
    c["range-fieldset"].hidden = !key;
    if (!key || key === this.rangeShown) return;
    this.rangeShown = key;
    const stored = this.ranges.get(key) ?? {mode: "auto", min: "", max: ""};
    c["range-mode"].value = stored.mode;
    c["range-min"].value = stored.min;
    c["range-max"].value = stored.max;
    this.showRange();
  }

  showRange() {
    const c = this.c, fixed = c["range-mode"].value === "fixed";
    c["range-wrap"].hidden = !fixed;
    const invalid = fixed && !this.fixedRange();
    c["range-max"].setCustomValidity(invalid ? "The maximum must exceed the minimum" : "");
  }

  /** [min, max] when the plotted quantity has a valid fixed range, otherwise null (automatic). */
  fixedRange() {
    const c = this.c;
    if (!this.rangeKey || c["range-mode"].value !== "fixed") return null;
    const lo = c["range-min"].valueAsNumber, hi = c["range-max"].valueAsNumber;
    return Number.isFinite(lo) && Number.isFinite(hi) && hi > lo ? [lo, hi] : null;
  }

  setCursor(cursor) {
    this.cursor = cursor;
    this.c.lat.value = String(Number(cursor.lat.toFixed(4)));
    this.c.lon.value = String(Number((((cursor.lon % 360) + 360) % 360).toFixed(4)));
    return this.update();
  }

  /** Called when selection A changes time, quantity or level. */
  follow() { return this.update(); }

  request() {
    const c = this.c, links = this.links, product = this.product;
    const time = c.follow.checked ? links.getTime() : Number(c.first.value);
    if (c.follow.checked) c.first.value = c.last.value = String(time);
    const base = {time};
    if (product === "column") return {kind: "column", ...base, lat: this.cursor.lat, lon: this.cursor.lon, snap: c.snap.checked ? 1 : 0};
    if (product === "meridional") return {kind: "meridional", ...base, at: this.cursor.lon};
    if (product === "zonal") return {kind: "zonal", ...base, at: this.cursor.lat};
    let last = c.follow.checked ? time : Number(c.last.value);
    if (last < time) { last = time; c.last.value = String(time); }
    return {kind: "zonal_mean", first: time, last};
  }

  syncControls() {
    const c = this.c, product = this.product;
    const linked = this.links.getField();
    const fields = product === "streamfunction" ? [] : Object.keys(FIELDS).filter(key =>
      product === "column" || product === "meridional" || product === "zonal" || ["temperature", "u", "v"].includes(key));
    const old = c.field.value;
    c.field.replaceChildren(...fields.map(key => new Option(FIELDS[key].name + " (" + FIELDS[key].symbol + ") · " + FIELDS[key].units, key)));
    if (this.fieldLinked !== linked && fields.includes(linked)) c.field.value = linked;
    else if (fields.includes(old)) c.field.value = old;
    this.fieldLinked = linked;
    c["field-wrap"].hidden = !fields.length;
    c["axis-wrap"].hidden = product !== "column";
    c["snap-wrap"].hidden = product !== "column";
    c["last-wrap"].hidden = !["zonal_mean", "streamfunction"].includes(product);
    this.loadRange();
  }

  async update() {
    if (!this.cursor) return;
    this.syncControls();
    const query = this.request();
    const key = JSON.stringify(query);
    if (key === this.key && this.data) { this.draw(); return; }
    this.key = key;
    this.abort?.abort();
    const abort = new AbortController();
    this.abort = abort;
    this.c.png.disabled = true;
    const count = query.kind === "zonal_mean" ? query.last - query.first + 1 : 1;
    this.status(count > 1 ? `Averaging ${count} saved states on the native grid…` : "Evaluating saved state…");
    try {
      const response = await fetch("structure?" + new URLSearchParams(query), {signal: abort.signal});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || response.statusText);
      if (this.abort !== abort) return;
      this.data = data; this.query = query;
      this.status("");
      this.draw();
      this.c.png.disabled = false;
    } catch (error) {
      if (abort.signal.aborted || this.abort !== abort) return;
      this.data = null; this.key = null;
      this.c.plot.hidden = true;
      this.status(error.message, true);
    }
  }

  status(text, error = false) {
    this.c.status.textContent = text;
    this.c.status.classList.toggle("error", error);
  }

  identity() {
    const s = this.links.manifest.source;
    return [s.directory, s.config_sha256 ? "config " + s.config_sha256.slice(0, 12) : ""].filter(Boolean).join(" · ");
  }

  timeText() {
    const q = this.query;
    if (q.kind !== "zonal_mean") return this.links.timeLabel(q.time) + " · instantaneous saved state (" + this.cadence + ")";
    const s = this.data.sample;
    if (s.count === 1) return this.links.timeLabel(q.first) + " · single saved state (" + this.cadence + ")";
    return `Equal-weight mean of ${s.count} saved states, ${this.links.timeLabel(q.first).split(" · ")[0]} to ` +
      `${this.links.timeLabel(q.last).split(" · ")[0]} (${s.cadence} cadence)`;
  }

  plotSpec() {
    const d = this.data, q = this.query, product = this.product, c = this.c;
    const physics = this.links.manifest.physics;
    const reference = physics.reference_pressure ?? physics.surface_pressure ?? null;
    const field = FIELDS[c.field.value];
    const level = this.links.getLevel();
    const lonTicks = [0, 60, 120, 180, 240, 300, 360], latTicks = [-90, -60, -30, 0, 30, 60, 90];
    const latLabel = t => t ? Math.abs(t) + "°" + (t < 0 ? "S" : "N") : "0°";
    const title = [this.links.runName(), this.identity()];
    if (product === "column") return this.profileSpec(field, title);
    if (product === "meridional" || product === "zonal") {
      const meridional = product === "meridional";
      return {
        kind: "section", values: d[c.field.value], x: d.points, levels: d.sigma, reference, selected: level,
      range: this.fixedRange(),
        xRange: meridional ? [-90, 90] : [0, 360], ticks: meridional ? latTicks : lonTicks,
        tickLabel: meridional ? latLabel : t => t + "°E", diverging: field.diverging,
        xLabel: meridional ? "Latitude" : "Longitude (east)", yLabel: "Nominal σ, native full levels (log)",
        cursor: meridional ? this.cursor.lat : ((this.cursor.lon % 360) + 360) % 360,
        label: field.name + " (" + field.symbol + ") · " + field.units,
        title: [...title, (meridional ? "Meridional section at " + numberText(d.at) + "°E" :
          "Zonal section at " + degrees(d.at, "N", "S")) + " · " + this.timeText(), d.sampling],
      };
    }
    const streamfunction = product === "streamfunction";
    const meta = streamfunction ? PSI : field;
    return {
      kind: "section", values: streamfunction ? d.streamfunction : d[c.field.value], x: d.lat,
      levels: streamfunction ? d.sigma_interfaces : d.sigma, reference, selected: streamfunction ? null : level,
      range: this.fixedRange(),
      xRange: [-90, 90], ticks: latTicks, tickLabel: latLabel, diverging: meta.diverging,
      xLabel: "Latitude (" + d.lat.length + " Gauss–Legendre nodes)",
      yLabel: streamfunction ? "Nominal σ at layer interfaces (log)" : "Nominal σ, native full levels (log)",
      cursor: this.cursor.lat, label: (streamfunction ? PSI.name + " (ψ) · kg s⁻¹, positive northward above the interface" :
        "Zonal mean [" + meta.symbol + "] · " + meta.units),
      title: [...title, this.timeText(), streamfunction ? d.definitions.streamfunction : d.definitions.zonal_mean],
    };
  }

  profileSpec(field, title) {
    const d = this.data, axis = this.c.axis.value, key = this.c.field.value, n = d.sigma.length;
    const y = k => axis === "height" ? d.geopotential_height[k] / 1000 :
      axis === "pressure" ? -Math.log10(d.pressure[k]) : -Math.log10(d.sigma[k]);
    const ySurface = axis === "height" ? d.surface_height / 1000 : axis === "pressure" ? -Math.log10(d.surface_pressure) : 0;
    const points = Array.from({length: n}, (_, k) => ({x: d[key][k], y: y(k)}));
    const markers = [];
    const low = d.lowest_air;
    if (key === "temperature") {
      markers.push({x: low.value, y: y(n - 1), color: "#6fb7b5", shape: "circle", offset: -8,
        label: "Lowest full level air T = " + numberText(low.value) + " K (σ " + numberText(low.sigma) + ")"});
      if (d.surface_temperature) markers.push({x: d.surface_temperature.value, y: ySurface, color: "#e0a03c",
        shape: "square", offset: 16, label: "Slab surface Tₛ = " + numberText(d.surface_temperature.value) + " K"});
    }
    const yTick = axis === "height" ? v => numberText(v) + " km" : axis === "pressure" ?
      v => pressureText(10 ** -v) : v => "σ " + numberText(10 ** -v);
    return {
      kind: "profile", points, markers, zero: !!field.diverging, yTick, selected: this.links.getLevel(),
      label: field.name + " (" + field.symbol + ") · " + field.units,
      yLabel: {height: "Geopotential height Z (km above datum)", pressure: "Pressure p (log)", sigma: "Nominal σ (log)"}[axis],
      title: [...title, "Column at " + place({lat: d.latitude, lon: d.longitude}) + " · " + d.sampling,
        this.timeText()],
    };
  }

  draw() {
    const c = this.c;
    if (!this.data) return;
    const width = Math.max(320, c.plot.parentElement.clientWidth - 4 || 900);
    this.width = this.node.clientWidth;
    this.spec = this.plotSpec();
    c.plot.hidden = false;
    c.plot.style.width = width + "px";
    this.layout = this.spec.kind === "profile" ? drawProfile(c.plot, this.spec, width, Math.min(devicePixelRatio, 2)) :
      drawSection(c.plot, this.spec, width, Math.min(devicePixelRatio, 2));
    c.plot.style.height = this.layout.height + "px";
    this.notes();
  }

  notes() {
    const d = this.data, c = this.c, rows = [];
    if (this.product === "column") {
      rows.push("Air temperature at the lowest full level (σ " + numberText(d.lowest_air.sigma) + ", p " +
        pressureText(d.lowest_air.pressure) + ", Z " + numberText(d.lowest_air.height) + " m) is not a surface temperature.");
      rows.push(d.surface_temperature ? "Tₛ: " + d.surface_temperature.definition + "; " + d.surface_temperature.sampling + "." :
        "This archive has no slab surface temperature at the selected time.");
      rows.push("Surface pressure pₛ " + pressureText(d.surface_pressure) + " · terrain height " + numberText(d.surface_height) + " m.");
      rows.push(...Object.values(d.conventions));
      this.table(d);
    } else {
      c.table.hidden = true;
      const range = this.spec.range;
      if (range) {
        let below = 0, above = 0, total = 0;
        for (const row of this.spec.values) for (const v of row) {
          if (!Number.isFinite(v)) continue;
          total++; if (v < range[0]) below++; else if (v > range[1]) above++;
        }
        rows.push("Fixed colour range " + valueText(range[0]) + " to " + valueText(range[1]) + ": " + below +
          " of " + total + " plotted values lie below it and " + above + " above it, drawn at the end colours.");
      }
      if (d.sample) {
        rows.push("Native grid: " + d.grid.latitudes + "; " + d.grid.longitudes + ".");
        if (this.product === "streamfunction") rows.push("Bottom (σ = 1) whole-column flux: max |ψ| " +
          valueText(d.column_flux_residual.max_abs) + " kg s⁻¹, against max |ψ| " +
          valueText(d.column_flux_residual.max_abs_interior) + " kg s⁻¹ in the section. " + d.column_flux_residual.note + ".");
      }
    }
    rows.push("Daily or periodic snapshots only: transient extrema between saved states are not represented.");
    c.notes.replaceChildren(...rows.map(text => Object.assign(document.createElement("p"), {textContent: text})));
  }

  table(d) {
    const c = this.c, body = c.table.tBodies[0];
    body.replaceChildren(...d.sigma.map((sigma, k) => {
      const row = document.createElement("tr");
      if (k === this.links.getLevel()) row.className = "selected";
      for (const value of [k + 1, sigma, d.pressure[k], d.geopotential_height[k], d.temperature[k], d.u[k], d.v[k]]) {
        const cell = document.createElement("td");
        cell.textContent = numberText(value);
        row.append(cell);
      }
      row.onclick = () => this.links.onLevel(k);
      return row;
    }));
    c.table.hidden = false;
  }

  locate(e) {
    const r = this.c.plot.getBoundingClientRect(), L = this.layout;
    if (!L) return null;
    const x = (e.clientX - r.left) * L.width / r.width, y = (e.clientY - r.top) * L.height / r.height;
    const box = L.box;
    if (x < box.x || x > box.x + box.w || y < box.y || y > box.y + box.h) return null;
    if (L.kind === "profile") {
      let best = 0, distance = Infinity;
      L.points.forEach(([px, py], k) => { const dd = Math.hypot(px - x, py - y); if (dd < distance) { distance = dd; best = k; } });
      return {level: best, x: null};
    }
    const l = L.edges[0] + (y - box.y) / box.h * (L.edges[L.edges.length - 1] - L.edges[0]);
    let level = L.edges.findIndex((edge, k) => k > 0 && l <= edge) - 1;
    if (level < 0) level = L.rows - 1;
    const xv = L.x0 + (x - box.x) / box.w * (L.x1 - L.x0);
    let column = L.xEdges.findIndex((edge, i) => i > 0 && xv <= edge) - 1;
    if (column < 0) column = L.cols - 1;
    return {level, column, x: xv};
  }

  hover(e) {
    const hit = this.locate(e), d = this.data, c = this.c;
    if (!hit || !d) { c.readout.textContent = ""; return; }
    const k = hit.level, product = this.product, parts = [];
    if (product === "column") {
      parts.push("Level " + (k + 1) + " · σ " + numberText(d.sigma[k]), "p " + pressureText(d.pressure[k]),
        "Z " + numberText(d.geopotential_height[k]) + " m", "T " + numberText(d.temperature[k]) + " K",
        "u " + numberText(d.u[k]) + " m s⁻¹", "v " + numberText(d.v[k]) + " m s⁻¹");
    } else {
      const i = hit.column, x = this.spec.x[i];
      parts.push(product === "zonal" ? numberText(x) + "°E" : degrees(x, "N", "S"));
      if (product === "streamfunction") {
        parts.push("interface " + k + " · σ " + numberText(d.sigma_interfaces[k]),
          "ψ " + valueText(d.streamfunction[k][i]) + " kg s⁻¹");
      } else {
        const field = FIELDS[c.field.value];
        parts.push("level " + (k + 1) + " · σ " + numberText(d.sigma[k]));
        if (d.pressure) parts.push("p " + pressureText(d.pressure[k][i]), "Z " + numberText(d.geopotential_height[k][i]) + " m");
        else parts.push("p ≈ σ[pₛ] " + pressureText(d.sigma[k] * d.surface_pressure[i]));
        parts.push((d.sample ? "[" + field.symbol + "] " : field.symbol + " ") + numberText(this.spec.values[k][i]) + " " + field.units);
      }
    }
    c.readout.textContent = parts.join(" · ");
  }

  click(e) {
    const hit = this.locate(e);
    if (!hit) return;
    const product = this.product;
    if (product !== "streamfunction" && this.c["link-level"].checked) this.links.onLevel(hit.level);
    if (product === "meridional" || product === "zonal_mean" || product === "streamfunction")
      this.links.onCursor({lat: this.spec.x[hit.column], lon: this.cursor.lon});
    else if (product === "zonal") this.links.onCursor({lat: this.cursor.lat, lon: this.spec.x[hit.column]});
  }

  png() {
    const canvas = document.createElement("canvas");
    const spec = this.spec;
    if (spec.kind === "profile") drawProfile(canvas, spec, 1100, 1.5);
    else drawSection(canvas, spec, 1100, 1.5);
    canvas.toBlob(blob => {
      if (!blob) return;
      const url = URL.createObjectURL(blob), a = document.createElement("a");
      const q = this.query;
      a.href = url;
      a.download = (this.links.runName() + "-" + this.product + "-" + (this.c.field.value || "psi") + "-" +
        (q.kind === "zonal_mean" ? "t" + q.first + "-" + q.last : "t" + q.time) + ".png").replace(/[^a-z0-9._-]/gi, "_");
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 5000);
    }, "image/png");
  }
}
