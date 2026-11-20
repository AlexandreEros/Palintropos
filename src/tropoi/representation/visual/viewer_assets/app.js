import {loadFrame} from "./data.js";
import {SurfaceView} from "./render.js";
import {StructureView} from "./structure.js";

const byId = id => document.getElementById(id);
const panels = [];
const coordinates = {
  native: "Native full level",
  pressure: "Pressure (Pa)",
  sigma: "Nominal sigma (p/pₛ)",
  geopotential_height: "Geopotential height (m)",
};
let manifest;
let times;
let levels;
let cursor;
let structure;

// One geographic cursor links the globes, maps, column and sections.
function setCursor(point) {
  if (!point || !Number.isFinite(point.lat) || !Number.isFinite(point.lon)) return;
  cursor = {lat: Math.max(-90, Math.min(90, point.lat)), lon: ((point.lon % 360) + 360) % 360};
  for (const panel of panels) panel.view.setCursor(cursor);
  structure?.setCursor(cursor);
}

// A level chosen in the column or a section becomes selection A's native level.
function selectLevel(k) {
  const panel = panels[0];
  if (!panel?.field.levels || !Number.isInteger(k) || k < 0 || k >= levels.length) return;
  panel.controls.coordinate.value = "native";
  panel.updateCoordinate();
  panel.controls.level.value = String(k);
  panel.changeLevel();
}

async function json(path, signal) {
  const response = await fetch(path, {signal});
  if (!response.ok) throw new Error(await response.text() || response.statusText);
  return response.json();
}

function options(element, choices) {
  element.replaceChildren(...choices.map(([value, text]) => new Option(text, value)));
}

function download(blob, name) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name.replace(/[^a-z0-9._-]/gi, "_") + ".png";
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

function quantityLabel(field) {
  const {name, symbol, units} = field.presentation;
  return [symbol ? name + " (" + symbol + ")" : name, units].filter(Boolean).join(" · ");
}

function runName() {
  return manifest.source.run_id || manifest.source.directory;
}

function timeLabel(index) {
  const seconds = times[index];
  return "Day " + Number((seconds / 86400).toPrecision(8)) + " · " + seconds + " s";
}

function sync(source, dimension) {
  if (panels.length !== 2 || !byId("link-" + dimension).checked) return;
  const target = panels.find(panel => panel !== source);
  if (dimension === "time") {
    target.controls.time.value = source.controls.time.value;
  } else {
    if (!target.field.levels || !source.field.levels) return;
    const selection = source.controls.coordinate.value;
    if (![...target.controls.coordinate.options].some(o => o.value === selection)) return;
    target.controls.coordinate.value = selection;
    target.controls.level.value = source.controls.level.value;
    target.updateCoordinate();
    target.controls.value.value = source.controls.value.value;
  }
  return target.load();
}

function scaleTogether() {
  const [a, b] = panels;
  let range;
  if (a && b && a.frame && b.frame && a.key === b.key && byId("link-color").checked) {
    const ranges = [a.extent, b.extent];
    if (ranges.every(Boolean)) {
      range = [Math.min(...ranges.map(r => r[0])), Math.max(...ranges.map(r => r[1]))];
    }
  }
  for (const panel of panels) {
    if (panel.frame) panel.draw(range);
  }
}

class Panel {
  constructor(letter, initial) {
    this.node = byId("panel").content.firstElementChild.cloneNode(true);
    byId("views").append(this.node);
    this.node.querySelector("h2").textContent = "Selection " + letter;
    this.controls = Object.fromEntries([...this.node.querySelectorAll("[data-id]")]
      .map(element => [element.dataset.id, element]));
    const c = this.controls;
    this.view = new SurfaceView(c.surface);
    this.view.onSelect = setCursor;
    this.view.cursor = cursor;
    this.playing = false;
    const keys = (manifest.operations.map || []).filter(key => key !== "wind");
    options(c.field, keys.map(key => [key, quantityLabel(manifest.fields[key])]));
    c.field.value = initial?.key ||
      ["temperature", "free_surface_height", "zeta"].find(key => keys.includes(key)) || keys[0];
    c.time.max = String(times.length - 1);
    const requested = new URLSearchParams(location.search).get("time");
    const initialTime = /^\d+$/.test(requested) && Number(requested) < times.length ? requested : "0";
    c.time.value = initial?.controls.time.value || initialTime;
    c.exaggeration.value = initial?.controls.exaggeration.value ||
      String(manifest.fields.free_surface_height ? 75 : levels.length ? 8 : 1);
    c.opacity.value = initial?.controls.opacity.value || (manifest.fields.free_surface_height ? "0.45" : "0.9");
    options(c.level, levels.map((sigma, index) => [index, (index + 1) + " · σ = " + Number(sigma.toPrecision(6))]));
    c.level.value = initial?.controls.level.value || String(levels.length - 1);
    this.updateField();
    if (initial && this.field.levels) {
      c.coordinate.value = initial.controls.coordinate.value;
      this.updateCoordinate();
      c.value.value = initial.controls.value.value;
    }
    c.field.onchange = () => { this.updateField(); this.load(); };
    c.time.oninput = () => this.changeTime(Number(c.time.value));
    c.previous.onclick = () => this.changeTime(Number(c.time.value) - 1);
    c.next.onclick = () => this.changeTime(Number(c.time.value) + 1);
    c.coordinate.onchange = () => { this.updateCoordinate(); this.changeLevel(); };
    c.value.onchange = c.level.onchange = () => this.changeLevel();
    c.play.onclick = () => this.playing ? this.stop() : this.play();
    c["range-mode"].onchange = () => {
      const fixed = c["range-mode"].value === "fixed";
      c["range-wrap"].hidden = !fixed;
      if (fixed && this.view.range) {
        if (c["range-min"].value === "") c["range-min"].value = String(Number(this.view.range[0].toPrecision(6)));
        if (c["range-max"].value === "") c["range-max"].value = String(Number(this.view.range[1].toPrecision(6)));
      }
      scaleTogether();
    };
    c["range-min"].onchange = c["range-max"].onchange = () => scaleTogether();
    for (const id of ["opacity", "exaggeration", "light", "winds"]) {
      c[id].oninput = () => {
        if (c.exaggeration.validity.valid) scaleTogether();
      };
    }
    c.png.onclick = async () => {
      const selection = this.key + "-t" + times[Number(c.time.value)] + "-" + this.surfaceLabel();
      try { download(await this.view.png(), runName() + "-" + selection); }
      catch (error) { this.status(error.message, true); }
    };
  }

  get key() { return this.controls.field.value; }
  get field() { return manifest.fields[this.key]; }
  get coordinate() { return this.field.levels ? this.controls.coordinate.value : "native"; }

  updateField() {
    const c = this.controls;
    const p = this.field.presentation;
    c.classification.textContent = p.role + " · " + p.origin;
    c.definition.textContent = p.definition;
    c.vertical.hidden = !this.field.levels;
    const old = c.coordinate.value;
    options(c.coordinate, Object.entries(coordinates).filter(([key]) => {
      if (key === "pressure" && this.key === "pressure") return false;
      if (key === "geopotential_height" && ["geopotential_height", "geopotential"].includes(this.key)) return false;
      return key === "native" || manifest.operations[key]?.includes(this.key);
    }));
    if ([...c.coordinate.options].some(o => o.value === old)) c.coordinate.value = old;
    const shell = this.shell();
    c["wind-wrap"].hidden = !manifest.operations.wind || (levels.length > 0 && !this.field.levels);
    c["opacity-wrap"].hidden = !shell;
    c["exaggeration-wrap"].hidden = !shell && !manifest.fields.terrain;
    c["light-wrap"].hidden = !manifest.star?.positions;
    this.updateCoordinate();
  }

  shell() {
    // A level is lifted to its geopotential height; without the run's gravity
    // there is none, and the level is drawn as a flat map on the sphere.
    return Boolean((this.field.levels && manifest.fields.geopotential_height) || manifest.fields.free_surface_height);
  }

  updateCoordinate() {
    const c = this.controls;
    const coord = this.coordinate;
    c["level-wrap"].hidden = coord !== "native";
    c["value-wrap"].hidden = coord === "native";
    if (coord === "native") return;
    const sigma = levels[Number(c.level.value)] ?? 0.5;
    const reference = manifest.physics.reference_pressure ?? manifest.physics.surface_pressure;
    const defaults = {sigma, pressure: reference ? Math.round(reference * sigma) : 50000,
      geopotential_height: 1000};
    c["value-label"].textContent = coordinates[coord];
    c.value.removeAttribute("max");
    c.value.removeAttribute("min");
    if (coord === "sigma") { c.value.min = "0"; c.value.max = "1"; }
    if (coord === "pressure") c.value.min = "0";
    c.value.value = String(defaults[coord]);
  }

  changeLevel() {
    sync(this, "level");
    return this.load();
  }

  async changeTime(index) {
    this.controls.time.value = String(Math.max(0, Math.min(times.length - 1, index)));
    await Promise.all([sync(this, "time"), this.load()]);
  }

  async play() {
    if (this.playing || times.length < 2) return;
    panels.forEach(panel => panel.stop());
    this.playing = true;
    const generation = this.playGeneration;
    this.controls.play.textContent = "Pause";
    if (Number(this.controls.time.value) === times.length - 1) await this.changeTime(0);
    while (this.playing && generation === this.playGeneration) {
      await new Promise(done => setTimeout(done, 1000 / Number(this.controls.speed.value)));
      if (!this.playing || generation !== this.playGeneration) return;
      let index = Number(this.controls.time.value) + 1;
      if (index === times.length) {
        if (!this.controls.loop.checked) break;
        index = 0;
      }
      await this.changeTime(index);
    }
    if (generation === this.playGeneration) this.stop();
  }

  stop() {
    this.playGeneration = (this.playGeneration || 0) + 1;
    this.playing = false;
    this.controls.play.textContent = "Play";
  }

  surfaceLabel() {
    const c = this.controls;
    if (!this.field.levels) {
      if (manifest.fields.free_surface_height) return "Shallow-water free surface";
      return levels.length ? "Lower boundary" : "Horizontal field";
    }
    if (this.coordinate === "native") return "Native level " + c.level.selectedOptions[0].text;
    return coordinates[this.coordinate] + " = " + c.value.value;
  }

  status(message, error = false) {
    this.controls.status.textContent = message;
    this.controls.status.classList.toggle("error", error);
  }

  async load() {
    const c = this.controls;
    this.request?.abort();
    const request = new AbortController();
    this.request = request;
    this.frame = this.extent = null;
    c.png.disabled = c.surface.hidden = true;
    const time = Number(c.time.value);
    c["time-label"].textContent = timeLabel(time) + " · " + (time + 1) + "/" + times.length;
    c.previous.disabled = time === 0;
    c.next.disabled = time === times.length - 1;
    c.play.disabled = c.time.disabled = times.length <= 1;
    const axis = manifest.time_axes[this.field.time_axis];
    if (this.field.time_axis === "slab" && !axis.values.includes(times[time])) {
      this.status("No slab snapshot was saved at this time.");
      return;
    }
    const coord = this.coordinate;
    const value = coord === "native" ? (this.field.levels ? Number(c.level.value) : 0) : c.value.valueAsNumber;
    if (coord !== "native" && (!c.value.validity.valid || !Number.isFinite(value) ||
        (["sigma", "pressure"].includes(coord) && value <= 0))) {
      this.status("Enter a valid " + coordinates[coord] + " selection.", true);
      return;
    }
    this.status("Loading " + this.field.presentation.name.toLowerCase() + "…");
    try {
      const frame = await loadFrame({time, field: this.key, coordinate: coord, value}, request.signal);
      if (this.request !== request || request.signal.aborted) return;
      this.frame = frame;
      // Avoid a negative displayed radius when exaggerating terrain below datum.
      const geo = manifest.geometry;
      const radius = geo.dynamical_radius_m || geo.reference_radius_m;
      let minimum = 0;
      for (const heights of [frame.terrain, frame.height]) {
        if (heights) for (const value of heights) if (value < minimum) minimum = value;
      }
      c.exaggeration.max = String(minimum < 0 ? Math.min(1000, -.95 * radius / minimum) : 1000);
      if (Number(c.exaggeration.value) > Number(c.exaggeration.max)) c.exaggeration.value = c.exaggeration.max;
      c.surface.hidden = false;
      this.draw();
      this.extent = this.view.range?.slice();
      if (panels.length > 1) scaleTogether();
      this.status("");
      c.png.disabled = false;
      if (this === panels[0]) structure?.follow();
    } catch (error) {
      if (!request.signal.aborted && this.request === request) {
        this.frame = null;
        c.surface.hidden = true;
        this.status(error.message, true);
        this.stop();
      }
    }
  }

  fixedRange() {
    const c = this.controls;
    if (c["range-mode"].value !== "fixed") return null;
    const lo = c["range-min"].valueAsNumber, hi = c["range-max"].valueAsNumber;
    return Number.isFinite(lo) && Number.isFinite(hi) && hi > lo ? [lo, hi] : null;
  }

  draw(range) {
    const c = this.controls;
    range = this.fixedRange() || range;
    const time = Number(c.time.value);
    const geo = manifest.geometry;
    const p = this.field.presentation;
    this.view.draw(this.frame, {
      radius: geo.dynamical_radius_m || geo.reference_radius_m,
      title: runName(), field: this.key, label: p.name + (p.symbol ? " (" + p.symbol + ")" : ""),
      units: p.units, selection: timeLabel(time) + " · " + this.surfaceLabel() + " · " + p.role + " · " + p.origin +
        (this.fixedRange() ? " · fixed colour range" : ""),
      winds: !c["wind-wrap"].hidden && c.winds.checked, shell: this.shell(),
      opacity: Number(c.opacity.value), exaggeration: Number(c.exaggeration.value),
      star: c.light.value === "star" ? manifest.star?.positions?.[time] : null, center: geo.initial_center,
      maskLabels: {1: manifest.fields.free_surface_height ? "Nonpositive fluid thickness" : "Below terrain", 2: "Above highest sampled full level", 3: "Below lowest sampled full level"},
      range
    });
  }

  destroy() {
    this.stop();
    this.request?.abort();
    this.view.destroy();
    this.node.remove();
  }
}

function compare() {
  const enabled = panels.length === 1;
  if (enabled) panels.push(new Panel("B", panels[0]));
  else panels.pop().destroy();
  byId("views").classList.toggle("comparing", enabled);
  byId("compare").setAttribute("aria-pressed", String(enabled));
  byId("compare").textContent = enabled ? "Close comparison" : "Compare selections";
  byId("links").hidden = !enabled;
  byId("link-level-label").hidden = !levels.length;
  if (enabled) panels[1].load();
  else scaleTogether();
}

// Keep the recorded samples in their own cadence, including real gaps.
function plotSeries({x, y, x_label, units}, name, key, source) {
  const canvas = byId("chart");
  const ctx = canvas.getContext("2d");
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  if (x.length !== y.length) throw new Error("Diagnostic time and value counts disagree.");
  for (let i = 0; i < x.length; i++) {
    if (!Number.isFinite(x[i]) || !Number.isFinite(y[i])) continue;
    xmin = Math.min(xmin, x[i]); xmax = Math.max(xmax, x[i]);
    ymin = Math.min(ymin, y[i]); ymax = Math.max(ymax, y[i]);
  }
  if (!Number.isFinite(xmin)) throw new Error("No finite diagnostic samples.");
  const left = 115, top = 92, right = canvas.width - 45, bottom = canvas.height - 65;
  const scale = (v, lo, hi) => lo === hi ? 0.5 : (v - lo) / (hi - lo);
  const px = value => left + (right - left) * scale(value, xmin, xmax);
  const py = value => bottom - (bottom - top) * scale(value, ymin, ymax);
  const tick = v => Number(v.toPrecision(5)).toString();
  ctx.fillStyle = "#fffef9";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#19383b";
  ctx.font = "24px Georgia";
  ctx.fillText(name + (units ? " (" + units + ")" : ""), 22, 32, canvas.width - 44);
  ctx.font = "14px sans-serif";
  ctx.fillText(runName() + " · " + source + " · " + key, 23, 58, canvas.width - 46);
  for (let i = 0; i <= 4; i++) {
    const f = i / 4;
    if (ymin !== ymax || i === 2) {
      const value = ymin + f * (ymax - ymin);
      const row = py(value);
      ctx.strokeStyle = "#d5deda"; ctx.beginPath(); ctx.moveTo(left, row); ctx.lineTo(right, row); ctx.stroke();
      ctx.textAlign = "right"; ctx.fillText(tick(value), left - 12, row + 5);
    }
    if (xmin !== xmax || i === 2) {
      const value = xmin + f * (xmax - xmin);
      ctx.textAlign = "center"; ctx.fillText(tick(value), px(value), bottom + 23);
    }
  }
  ctx.textAlign = "center";
  ctx.fillText(x_label || "Time", (right + left) / 2, bottom + 51);
  ctx.textAlign = "left";
  ctx.strokeStyle = "#106d6c"; ctx.lineWidth = 2;
  ctx.beginPath();
  let joined = false;
  for (let i = 0; i < y.length; i++) {
    if (!Number.isFinite(x[i]) || !Number.isFinite(y[i])) { joined = false; continue; }
    const xx = px(x[i]), yy = py(y[i]);
    if (joined) ctx.lineTo(xx, yy);
    else ctx.moveTo(xx, yy);
    joined = true;
    ctx.fillRect(xx - 1, yy - 1, 2, 2);
  }
  ctx.stroke();
  return y.length;
}

function showDiagnostics(keys) {
  byId("series").hidden = false;
  const select = byId("diagnostic");
  const nameFor = key => manifest.diagnostics[key].presentation?.name || key;
  options(select, keys.map(key => [key, nameFor(key)]));
  select.value = keys.includes("mean_T") ? "mean_T" : keys[0];
  let request;
  async function load() {
    request?.abort();
    request = new AbortController();
    const {signal} = request;
    const key = select.value;
    byId("series-png").disabled = byId("chart").hidden = true;
    byId("series-note").textContent = "Loading series…";
    try {
      const data = await json("diagnostic?" + new URLSearchParams({field: key}), signal);
      if (signal.aborted) return;
      const product = manifest.diagnostics[key];
      const samples = plotSeries(data, nameFor(key), key, product.path);
      byId("series-note").textContent = [samples + " samples", product.time_axis, product.path].join(" · ");
      byId("series-png").disabled = byId("chart").hidden = false;
    } catch (error) {
      if (!signal.aborted) byId("series-note").textContent = error.message;
    }
  }
  select.onchange = load;
  byId("series-png").onclick = () => {
    const key = select.value;
    byId("chart").toBlob(blob => { if (blob) download(blob, runName() + "-" + key); });
  };
  return load();
}

async function openArchive() {
  const [inventory, metadata] = await Promise.all([json("runs"), json("manifest")]);
  manifest = metadata;
  byId("run-title").textContent = runName();
  document.title = runName() + " · Palintropos";
  if (inventory.runs.length > 1) {
    byId("run-picker").hidden = false;
    const select = byId("runs");
    options(select, inventory.runs.map(run => [run.id, run.label]));
    select.value = inventory.current;
    select.onchange = () => {
      const run = inventory.runs.find(run => String(run.id) === select.value);
      if (!run) return;
      // Carry the geographic cursor and structure product; saved times differ per archive.
      const keep = cursor ? {lat: cursor.lat, lon: cursor.lon} : {};
      if (structure) keep.product = structure.product;
      location.assign(run.url + "?" + new URLSearchParams(keep));
    };
  }
  times = manifest.time_axes.snapshots?.values || [];
  levels = manifest.axes.vertical?.full_levels || [];
  const facts = [manifest.source.solver?.toUpperCase(), times.length + " saved states"];
  if (levels.length) facts.push(levels.length + " full levels");
  if (times.length) facts.push(timeLabel(0).split(" · ")[0] + " to " + timeLabel(times.length - 1).split(" · ")[0]);
  byId("run-facts").textContent = facts.filter(Boolean).join(" · ");
  if (times.length && manifest.operations.map?.length) {
    const query = new URLSearchParams(location.search);
    const star = manifest.star?.positions?.[0];
    const center = manifest.geometry.initial_center;
    cursor = {lat: Number(query.get("lat") ?? star?.latitude ?? center?.latitude ?? 0),
      lon: Number(query.get("lon") ?? star?.longitude ?? center?.longitude ?? 0)};
    if (!Number.isFinite(cursor.lat) || !Number.isFinite(cursor.lon)) cursor = {lat: 0, lon: 0};
    byId("toolbar").hidden = false;
    const first = new Panel("A");
    panels.push(first);
    byId("compare").onclick = compare;
    byId("link-color").onchange = scaleTogether;
    byId("link-time").onchange = () => sync(first, "time");
    byId("link-level").onchange = () => sync(first, "level");
    if (manifest.operations.structure?.length) {
      structure = new StructureView(byId("structure"), {
        manifest, runName, timeLabel,
        getTime: () => Number(first.controls.time.value),
        getField: () => first.key,
        getLevel: () => first.field.levels && first.coordinate === "native" ? Number(first.controls.level.value) : null,
        onCursor: setCursor, onLevel: selectLevel,
      });
      const product = query.get("product");
      if (product && [...structure.c.product.options].some(o => o.value === product)) structure.c.product.value = product;
    }
    setCursor(cursor);
    // Inspection handle for acceptance tests and the browser console.
    window.viewer = {panels, get structure() { return structure; }, get cursor() { return cursor; }};
    await first.load();
  } else {
    const keys = (manifest.operations.series || []).filter(k => !manifest.diagnostics[k].columns);
    if (keys.length) await showDiagnostics(keys);
    const reasons = Object.values(manifest.unavailable);
    if (reasons.length || !keys.length) {
      byId("notice").hidden = false;
      byId("notice").textContent = reasons.join(" · ") || "No supported fields or recorded series in this archive.";
    }
  }
}

openArchive().catch(error => {
  byId("run-title").textContent = "Could not open archive";
  byId("notice").hidden = false;
  byId("notice").textContent = error.message;
});
