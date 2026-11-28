import {loadFrame} from "./data.js";
import {SurfaceView} from "./render.js";
import {Explorer, dayText} from "./explorer.js";

// The viewer opens any archive through its explorer description: quantities,
// what each of their coordinates accepts, native levels and time axes. The
// globe panels draw one saved state on one surface; the explorer draws any
// role assignment, longitude × latitude included.

const byId = id => document.getElementById(id);
const panels = [];
let manifest;
let quantities;
let cursor;
let explorer;

function setCursor(point) {
  if (!point || !Number.isFinite(point.lat) || !Number.isFinite(point.lon)) return;
  cursor = {lat: Math.max(-90, Math.min(90, point.lat)), lon: ((point.lon % 360) + 360) % 360};
  for (const panel of panels) panel.view.setCursor(cursor);
  explorer?.setCursor(cursor);
}

// A level chosen in a profile or section becomes selection A's native level.
function selectLevel(k) {
  const panel = panels[0];
  const levels = panel?.quantity.vertical?.levels;
  if (!levels || !Number.isInteger(k) || !levels.some(l => l.index === k)) return;
  if (![...panel.controls.coordinate.options].some(o => o.value === "native")) return;
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

function quantityLabel(q) {
  const {name, symbol, units} = q.presentation;
  return [symbol ? name + " (" + symbol + ")" : name, units].filter(Boolean).join(" · ");
}

function runName() {
  return manifest.source.run_id || manifest.source.directory;
}

const axisValues = axis => manifest.time_axes[axis]?.values || [];

function timeText(axis, index) {
  const seconds = axisValues(axis)[index];
  const name = manifest.time_axes[axis]?.label;
  return dayText(seconds).replace("day", "Day") + " · " + seconds + " s" + (name ? " · " + name : "");
}

/** The sample of ``axis`` nearest ``seconds`` without passing it (the first otherwise). */
function sampleAtOrBefore(axis, seconds) {
  const values = axisValues(axis);
  const k = values.findIndex(s => s > seconds);
  return k < 0 ? values.length - 1 : Math.max(0, k - 1);
}

function sync(source, dimension) {
  if (panels.length !== 2 || !byId("link-" + dimension).checked) return;
  const target = panels.find(panel => panel !== source);
  if (dimension === "time") {
    target.controls.time.value = String(sampleAtOrBefore(target.axis, source.seconds));
  } else {
    if (!target.quantity.vertical || !source.quantity.vertical) return;
    const selection = source.controls.coordinate.value;
    if (![...target.controls.coordinate.options].some(o => o.value === selection)) return;
    target.controls.coordinate.value = selection;
    target.updateCoordinate();
    if ([...target.controls.level.options].some(o => o.value === source.controls.level.value))
      target.controls.level.value = source.controls.level.value;
    target.controls.value.value = source.controls.value.value;
    target.updateAxis();
  }
  return target.load();
}

function scaleTogether() {
  const [a, b] = panels;
  let range;
  if (a && b && a.frame && b.frame && a.key === b.key && byId("link-color").checked) {
    const ranges = [a.extent, b.extent];
    if (ranges.every(Boolean)) range = [Math.min(...ranges.map(r => r[0])), Math.max(...ranges.map(r => r[1]))];
  }
  for (const panel of panels) if (panel.frame) panel.draw(range);
}

// The globe draws one saved state on one surface: quantities whose longitude
// and latitude can be axes, at a fixed time and a fixed (or no) vertical coordinate.
const onGlobe = q => {
  const c = q.coordinates;
  return c.longitude.roles.includes("x") && c.latitude.roles.includes("y") && c.time.roles.includes("fixed") &&
    (c.vertical.roles.includes("fixed") || c.vertical.roles.includes("none"));
};
const mapKeys = () => Object.keys(quantities).filter(k => onGlobe(quantities[k]));

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
    const keys = mapKeys();
    options(c.field, keys.map(key => [key, quantityLabel(quantities[key])]));
    c.field.value = initial?.key ||
      ["temperature", "free_surface_height", "zeta"].find(key => keys.includes(key)) || keys[0];
    const atmosphere = Object.values(quantities).some(q => q.vertical?.domain === "atmosphere");
    c.exaggeration.value = initial?.controls.exaggeration.value ||
      String(manifest.fields.free_surface_height ? 75 : atmosphere ? 8 : 1);
    c.opacity.value = initial?.controls.opacity.value || (manifest.fields.free_surface_height ? "0.45" : "0.9");
    this.updateField(initial);
    const requested = new URLSearchParams(location.search).get("time");
    const count = axisValues(this.axis).length;
    c.time.value = initial ? String(sampleAtOrBefore(this.axis, initial.seconds)) :
      /^\d+$/.test(requested) && Number(requested) < count ? requested : "0";
    // A new quantity or level may have its own time axis: keep the time last shown.
    c.field.onchange = () => { const seconds = this.shownSeconds; this.updateField(); this.keepTime(seconds); this.load(); };
    c.time.oninput = () => this.changeTime(Number(c.time.value));
    c.previous.onclick = () => this.changeTime(Number(c.time.value) - 1);
    c.next.onclick = () => this.changeTime(Number(c.time.value) + 1);
    c.coordinate.onchange = () => { const seconds = this.shownSeconds; this.updateCoordinate(); this.changeLevel(seconds); };
    c.value.onchange = c.level.onchange = () => this.changeLevel(this.shownSeconds);
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
      c[id].oninput = () => { if (c.exaggeration.validity.valid) scaleTogether(); };
    }
    c.png.onclick = async () => {
      const selection = this.key + "-t" + this.seconds + "-" + this.surfaceLabel();
      try { download(await this.view.png(), runName() + "-" + selection); }
      catch (error) { this.status(error.message, true); }
    };
  }

  get key() { return this.controls.field.value; }
  get quantity() { return quantities[this.key]; }
  get field() { return manifest.fields[this.key] || {}; }
  get coordinate() { return this.quantity.vertical ? this.controls.coordinate.value : "native"; }
  get discrete() { return this.coordinate === "native" || this.quantity.vertical?.domain === "substrate"; }
  get level() {
    const levels = this.quantity.vertical?.levels || [];
    return levels.find(l => String(l.index) === this.controls.level.value) || null;
  }
  /** The time axis of the selection: a native level's own, else the quantity's. */
  get axis() {
    return (this.discrete && this.level?.time_axis) || this.quantity.time_axis;
  }
  get seconds() { return axisValues(this.axis)[Number(this.controls.time.value)]; }

  keepTime(seconds) {
    if (seconds !== undefined) this.controls.time.value = String(sampleAtOrBefore(this.axis, seconds));
    this.updateAxis();
  }

  updateAxis() {
    const c = this.controls, count = axisValues(this.axis).length;
    c.time.max = String(Math.max(0, count - 1));
    if (Number(c.time.value) > count - 1) c.time.value = String(count - 1);
  }

  updateField(initial) {
    const c = this.controls, q = this.quantity, p = q.presentation;
    c.classification.textContent = p.role + " · " + p.origin;
    c.definition.textContent = p.definition + (q.evaluation ? " Evaluation: " + q.evaluation + "." : "");
    c.vertical.hidden = !q.vertical;
    if (q.vertical) {
      const old = initial?.controls.coordinate.value || c.coordinate.value;
      const reps = q.coordinates.vertical.fixed;
      options(c.coordinate, reps.map(r => [r, q.vertical.representations[r].label]));
      if (reps.includes(old)) c.coordinate.value = old;
      const keep = initial?.controls.level.value || c.level.value;
      this.fillLevels();
      const levels = q.vertical.levels;
      c.level.value = levels.some(l => String(l.index) === keep) ? keep :
        String(q.vertical.domain === "substrate" ? levels[0].index : levels[levels.length - 1].index);
    }
    const shell = this.shell();
    c["wind-wrap"].hidden = !manifest.operations.wind || (q.vertical === null && Object.values(quantities).some(x => x.vertical));
    c["opacity-wrap"].hidden = !shell;
    c["exaggeration-wrap"].hidden = !shell && !manifest.fields.terrain;
    c["light-wrap"].hidden = !manifest.star?.axes;
    this.updateCoordinate(initial);
  }

  fillLevels() {
    const c = this.controls, q = this.quantity, rep = this.coordinate;
    options(c.level, q.vertical.levels.map(l => [l.index, rep === "native" || !(rep in l) ? l.label :
      q.vertical.representations[rep].label.split(" (")[0] + " = " + Number(l[rep].toPrecision(6)) + " m" +
      (l.kind === "surface" ? " · diagnosed skin" : "")]));
  }

  shell() {
    // An atmospheric level is lifted to its geopotential height; without the
    // run's gravity there is none, and the level is drawn as a flat map.
    return Boolean((this.quantity.vertical?.domain === "atmosphere" && manifest.fields.geopotential_height) ||
      manifest.fields.free_surface_height);
  }

  updateCoordinate(initial) {
    const c = this.controls, coord = this.coordinate;
    if (!this.quantity.vertical) { this.updateAxis(); return; }
    const keep = c.level.value;
    this.fillLevels();
    if ([...c.level.options].some(o => o.value === keep)) c.level.value = keep;
    c["level-wrap"].hidden = !this.discrete;
    c["value-wrap"].hidden = this.discrete;
    c["level-label"].textContent = coord === "native" ? "Native level" : "Native depth";
    this.updateAxis();
    if (this.discrete) return;
    const level = this.quantity.vertical.levels.find(l => String(l.index) === c.level.value);
    const sigma = level?.sigma ?? 0.5;
    const reference = manifest.physics.reference_pressure ?? manifest.physics.surface_pressure;
    const defaults = {sigma, pressure: reference ? Math.round(reference * sigma) : 50000,
      geopotential_height: 1000, height_above_surface: 1000};
    const rep = this.quantity.vertical.representations[coord];
    c["value-label"].textContent = rep.label + (rep.units ? " (" + rep.units + ")" : "");
    c.value.removeAttribute("max");
    c.value.removeAttribute("min");
    if (coord === "sigma") { c.value.min = "0"; c.value.max = "1"; }
    if (coord === "pressure") c.value.min = "0";
    c.value.value = initial?.controls.value.value || String(defaults[coord]);
  }

  changeLevel(seconds = this.seconds) {
    this.keepTime(seconds);
    sync(this, "level");
    return this.load();
  }

  async changeTime(index) {
    this.controls.time.value = String(Math.max(0, Math.min(axisValues(this.axis).length - 1, index)));
    await Promise.all([sync(this, "time"), this.load()]);
  }

  async play() {
    const count = axisValues(this.axis).length;
    if (this.playing || count < 2) return;
    panels.forEach(panel => panel.stop());
    this.playing = true;
    const generation = this.playGeneration;
    this.controls.play.textContent = "Pause";
    if (Number(this.controls.time.value) === count - 1) await this.changeTime(0);
    while (this.playing && generation === this.playGeneration) {
      await new Promise(done => setTimeout(done, 1000 / Number(this.controls.speed.value)));
      if (!this.playing || generation !== this.playGeneration) return;
      let index = Number(this.controls.time.value) + 1;
      if (index === count) {
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
    const c = this.controls, q = this.quantity;
    if (!q.vertical) {
      if (manifest.fields.free_surface_height) return "Shallow-water free surface";
      if (Object.values(quantities).some(x => x.vertical?.domain === "substrate")) return "Surface";
      return Object.values(quantities).some(x => x.vertical) ? "Lower boundary" : "Horizontal field";
    }
    if (this.discrete) return (this.coordinate === "native" ? "Native level " : "") + c.level.selectedOptions[0].text;
    return q.vertical.representations[this.coordinate].label + " = " + c.value.value;
  }

  selection() {
    const c = this.controls, q = this.quantity;
    const sel = {quantity: this.key, longitude: {role: "x"}, latitude: {role: "y"},
      time: {role: "fixed", index: Number(c.time.value)}, vertical: {role: "none"}};
    if (q.vertical) {
      const rep = this.coordinate;
      const value = rep === "native" ? this.level.index : this.discrete ? this.level[rep] : c.value.valueAsNumber;
      sel.vertical = {role: "fixed", representation: rep, value};
    }
    return sel;
  }

  /** Show a longitude × latitude selection made in the explorer. */
  show(selection) {
    const c = this.controls;
    if (!(selection.quantity in quantities) || !onGlobe(quantities[selection.quantity])) return;
    if (c.field.value !== selection.quantity) { c.field.value = selection.quantity; this.updateField(); }
    const v = selection.vertical;
    if (v.role === "fixed" && [...c.coordinate.options].some(o => o.value === v.representation)) {
      c.coordinate.value = v.representation;
      this.updateCoordinate();
      const levels = this.quantity.vertical.levels;
      const level = v.representation === "native" ? levels.find(l => l.index === v.value) :
        levels.find(l => l[v.representation] === v.value);
      if (level) c.level.value = String(level.index);
      else c.value.value = String(v.value);
    }
    this.updateAxis();
    if (Number.isInteger(selection.time.index)) c.time.value = String(selection.time.index);
    return this.load();
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
    const time = Number(c.time.value), count = axisValues(this.axis).length;
    this.shownSeconds = this.seconds;
    c["time-label"].textContent = timeText(this.axis, time) + " · " + (time + 1) + "/" + count;
    c.previous.disabled = time === 0;
    c.next.disabled = time === count - 1;
    c.play.disabled = c.time.disabled = count <= 1;
    const slab = manifest.time_axes[this.field.time_axis];
    if (this.field.time_axis === "slab" && !slab.values.includes(this.seconds)) {
      this.status("No slab snapshot was saved at this time.");
      return;
    }
    const coord = this.coordinate;
    if (!this.discrete) {
      const value = c.value.valueAsNumber;
      if (!c.value.validity.valid || !Number.isFinite(value) || (["sigma", "pressure"].includes(coord) && value <= 0)) {
        this.status("Enter a valid " + this.quantity.vertical.representations[coord].label + " selection.", true);
        return;
      }
    }
    this.status("Loading " + this.quantity.presentation.name.toLowerCase() + "…");
    try {
      const frame = await loadFrame(this.selection(), request.signal);
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
      this.status(frame.cells ? "Native cells: the flat map and its readout show each archived cell's value; " +
        "the globe interpolates between cell centres." : "");
      c.png.disabled = false;
      if (this === panels[0]) explorer?.follow();
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
    const p = this.quantity.presentation;
    this.view.draw(this.frame, {
      radius: geo.dynamical_radius_m || geo.reference_radius_m,
      title: runName(), field: this.key, label: p.name + (p.symbol ? " (" + p.symbol + ")" : ""),
      units: p.units, selection: timeText(this.axis, time) + " · " + this.surfaceLabel() + " · " + p.role + " · " + p.origin +
        (this.fixedRange() ? " · fixed colour range" : ""),
      winds: !c["wind-wrap"].hidden && c.winds.checked, shell: this.shell(),
      opacity: Number(c.opacity.value), exaggeration: Number(c.exaggeration.value),
      star: c.light.value === "star" ? manifest.star?.axes?.[this.axis]?.[time] : null, center: geo.initial_center,
      maskLabels: {1: manifest.fields.free_surface_height ? "Nonpositive fluid thickness" : "Below terrain",
        2: "Above highest sampled full level", 3: "Below lowest sampled full level"},
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
  byId("link-level-label").hidden = !Object.values(quantities).some(q => q.vertical);
  if (enabled) panels[1].load();
  else scaleTogether();
}

function facts() {
  const h = manifest.explorer.horizontal || {}, out = [];
  if (manifest.source.solver) out.push(manifest.source.solver.toUpperCase());
  if (manifest.source.model) out.push(manifest.source.model);
  if (h.kind === "points") out.push(h.columns + " diagnostic point" + (h.columns === 1 ? "" : "s") + " (no global field)");
  else if (h.columns) out.push(h.columns + " columns, " + h.kind + " " + h.nlat + " × " + h.nlon);
  const levels = (manifest.axes.vertical?.full_levels || []).length;
  if (levels) out.push(levels + " full levels");
  const cells = (manifest.axes.substrate?.centres || []).length;
  if (cells) out.push(cells + " substrate cells and the diagnosed skin");
  for (const [name, axis] of Object.entries(manifest.time_axes)) {
    if (!axis.values?.length || name === "slab" || name === "daily") continue;
    out.push(axis.values.length + " " + (axis.label || "saved state") + "s, " + dayText(axis.values[0]) + " to " +
      dayText(axis.values[axis.values.length - 1]));
  }
  if (manifest.source.completion) out.push(manifest.source.completion);
  return out.filter(Boolean).join(" · ");
}

async function openArchive() {
  const [inventory, metadata] = await Promise.all([json("runs"), json("manifest")]);
  manifest = metadata;
  quantities = manifest.explorer.quantities;
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
      // Carry the geographic cursor; saved times differ per archive.
      const keep = cursor ? {lat: cursor.lat, lon: cursor.lon} : {};
      location.assign(run.url + "?" + new URLSearchParams(keep));
    };
  }
  byId("run-facts").textContent = facts();
  const query = new URLSearchParams(location.search);
  const points = manifest.explorer.horizontal?.points;
  const star = manifest.star?.axes && Object.values(manifest.star.axes)[0]?.[0];
  const center = manifest.geometry.initial_center;
  cursor = {lat: Number(query.get("lat") ?? points?.[0]?.latitude ?? star?.latitude ?? center?.latitude ?? 0),
    lon: Number(query.get("lon") ?? points?.[0]?.longitude ?? star?.longitude ?? center?.longitude ?? 0)};
  if (!Number.isFinite(cursor.lat) || !Number.isFinite(cursor.lon)) cursor = {lat: 0, lon: 0};
  let first = null;
  if (mapKeys().length) {
    byId("toolbar").hidden = false;
    first = new Panel("A");
    panels.push(first);
    byId("compare").onclick = compare;
    byId("link-color").onchange = scaleTogether;
    byId("link-time").onchange = () => sync(first, "time");
    byId("link-level").onchange = () => sync(first, "level");
  }
  const explorable = Object.keys(quantities).length > 0;
  if (explorable) {
    explorer = new Explorer(byId("explorer"), {
      manifest, runName, timeText,
      getMap: () => first && {quantity: first.key, axis: first.axis, time: Number(first.controls.time.value)},
      showMap: selection => first?.show(selection),
      getLevel: () => first?.quantity.vertical?.domain === "atmosphere" && first.coordinate === "native" ?
        Number(first.controls.level.value) : null,
      onCursor: setCursor, onLevel: selectLevel,
    });
  }
  setCursor(cursor);
  // Inspection handle for acceptance tests and the browser console.
  window.viewer = {panels, get explorer() { return explorer; }, get cursor() { return cursor; }};
  if (first) await first.load();
  const reasons = Object.values(manifest.unavailable).filter(v => typeof v === "string");
  if (!first && !explorable) {
    byId("notice").hidden = false;
    byId("notice").textContent = reasons.join(" · ") || "No supported quantities in this archive.";
  }
}

openArchive().catch(error => {
  byId("run-title").textContent = "Could not open archive";
  byId("notice").hidden = false;
  byId("notice").textContent = error.message;
});
