// Horizontal-wind representations of a longitude × latitude Explorer result.
//
// v_h = r̂ × ∇ψ + ∇χ. Three distinguishable representations:
// * ψ contours: level sets of the streamfunction at a uniform interval Δψ.
//   They are the streamlines of the ROTATIONAL wind (all of a non-divergent
//   wind), and their spacing is Δn ≈ Δψ / |v_rot|: closer lines, faster flow.
//   The rotational wind keeps larger ψ on its right, so each contour is
//   oriented with larger ψ on its right and its arrows point downstream.
// * Streamlines of the full wind (u, v), integrated by overlays.js.
// * Quiver: arrows sampled at native mesh nodes.
// The server supplies ψ, u, v and the kinetic-energy split (viewer_atmosphere).

const DEG = Math.PI / 180;

/** The smallest of 1, 2, 2.5, 5 × 10^k at least `raw` (compose._nice_step). */
export function niceStep(raw) {
  if (!(raw > 0) || !Number.isFinite(raw)) return null;
  const base = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map(f => f * base).find(s => s >= raw * (1 - 1e-12));
  return Number(step.toPrecision(3));
}

/** The largest of 1, 2, 4, 5 × 10^k not above `vmax` (the top of compose._nice_speeds). */
export function niceSpeed(vmax) {
  if (!(vmax > 0)) return 1;
  const step = 10 ** Math.floor(Math.log10(vmax));
  return Number(Math.max(...[1, 2, 4, 5].map(f => f * step).filter(s => s <= vmax)).toPrecision(3));
}

/** Every multiple of `interval` inside [lo, hi] (the StreamfunctionLayer rule; 0 is a level). */
export function contourLevels(lo, hi, interval) {
  if (!(interval > 0) || !(hi >= lo)) return [];
  const out = [];
  for (let k = Math.ceil(lo / interval); k <= Math.floor(hi / interval); k++) out.push(k * interval);
  return out;
}

// Oriented marching squares. Cell corners a=(i,j), b=(i+1,j), c=(i+1,j+1), d=(i,j+1)
// with i eastward and j northward; edges 0 (a–b), 1 (b–c), 2 (c–d), 3 (d–a). Each
// segment [from, to] keeps the corners at or above the level on its right.
const SEGMENTS = {1: [[3, 0]], 2: [[0, 1]], 3: [[3, 1]], 4: [[1, 2]], 6: [[0, 2]], 7: [[3, 2]], 8: [[2, 3]],
  9: [[2, 0]], 11: [[2, 1]], 12: [[1, 3]], 13: [[1, 0]], 14: [[0, 3]]};
// Saddles: the cell-centre mean decides whether the high corners connect.
const SADDLE = {5: {high: [[1, 0], [3, 2]], low: [[3, 0], [1, 2]]}, 10: {high: [[0, 3], [2, 1]], low: [[0, 1], [2, 3]]}};

/** Isolines of `values` (rows south → north, columns west → east; NaN = masked) at `levels`.
 * Positions are fractional indices [x, y]. `periodic`: the last column repeats the
 * first (the closing meridian), so a line leaving the east edge continues at the west
 * edge and a loop around the planet closes. Returns, per level, polylines
 * {points, closed}, oriented with larger values on the right of the direction of travel. */
export function isolines(values, nx, ny, levels, periodic = false) {
  const at = (i, j) => values[j * nx + i];
  const wrapCol = i => (periodic && i === nx - 1 ? 0 : i);
  const edgeKey = (i, j, e) => e === 0 ? "h" + i + ":" + j : e === 2 ? "h" + i + ":" + (j + 1) :
    e === 3 ? "v" + wrapCol(i) + ":" + j : "v" + wrapCol(i + 1) + ":" + j;
  return levels.map(level => {
    const segs = [], starts = new Map();
    const point = (i, j, e) => {
      const [p, q] = e === 0 ? [[i, j], [i + 1, j]] : e === 1 ? [[i + 1, j], [i + 1, j + 1]] :
        e === 2 ? [[i, j + 1], [i + 1, j + 1]] : [[i, j], [i, j + 1]];
      const v0 = at(...p), v1 = at(...q), t = v1 === v0 ? 0.5 : (level - v0) / (v1 - v0);
      return [p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])];
    };
    for (let j = 0; j < ny - 1; j++) for (let i = 0; i < nx - 1; i++) {
      const a = at(i, j), b = at(i + 1, j), c = at(i + 1, j + 1), d = at(i, j + 1);
      if (!(Number.isFinite(a) && Number.isFinite(b) && Number.isFinite(c) && Number.isFinite(d))) continue;
      const code = (a >= level) | (b >= level) << 1 | (c >= level) << 2 | (d >= level) << 3;
      let pairs = SEGMENTS[code];
      if (!pairs && SADDLE[code]) pairs = (a + b + c + d) / 4 >= level ? SADDLE[code].high : SADDLE[code].low;
      for (const [e0, e1] of pairs || []) {
        const seg = {from: edgeKey(i, j, e0), to: edgeKey(i, j, e1), p: point(i, j, e0), q: point(i, j, e1)};
        starts.set(seg.from, segs.length);
        segs.push(seg);
      }
    }
    const used = new Uint8Array(segs.length), ends = new Set(segs.map(s => s.to)), lines = [];
    const follow = first => {
      const points = [segs[first].p];
      let k = first;
      while (k !== undefined && !used[k]) {
        used[k] = 1;
        points.push(segs[k].q);
        k = starts.get(segs[k].to);
      }
      return {points, closed: k === first};
    };
    // Open lines start where no segment ends (a mask or the mesh boundary); the rest are loops.
    segs.forEach((s, k) => { if (!ends.has(s.from)) lines.push(follow(k)); });
    segs.forEach((s, k) => { if (!used[k]) lines.push(follow(k)); });
    // A level equal to an extremum at a node degenerates to that point (a stagnation point of
    // the flow): zero-length segments around it carry no line and are not drawn.
    const extent = ({points}) => points.some(([x, y]) => Math.abs(x - points[0][0]) + Math.abs(y - points[0][1]) > 1e-12);
    return {level, lines: lines.filter(extent)};
  });
}

/** ψ (rows south → north) for contouring: each pole row is replaced by its mean. ψ at a pole
 * is a single value; synthesis roundoff along a pole row would otherwise make spurious
 * crossings on its zero-length edges. Display only. */
export function poleSnapped(values, lat, nx) {
  const out = Float64Array.from(values);
  lat.forEach((y, j) => {
    if (Math.abs(Math.abs(y) - 90) > 1e-9) return;
    let s = 0, n = 0;
    for (let i = 0; i < nx; i++) { const v = out[j * nx + i]; if (Number.isFinite(v)) { s += v; n++; } }
    if (n) for (let i = 0; i < nx; i++) if (Number.isFinite(out[j * nx + i])) out[j * nx + i] = s / n;
  });
  return out;
}

/** Arrowhead stations every `spacing` pixels of arc along a pixel polyline [[x, y], …]
 * (first at half a spacing; none on lines shorter than 0.4 spacing). Each station is
 * [x, y, dx, dy] with (dx, dy) the unit direction of travel. Segments longer than
 * `jump` pixels (a periodic seam) are not part of the arc. */
export function arrowStations(pixels, spacing, jump = Infinity) {
  const out = [];
  let total = 0;
  for (let k = 1; k < pixels.length; k++) {
    const l = Math.hypot(pixels[k][0] - pixels[k - 1][0], pixels[k][1] - pixels[k - 1][1]);
    if (l < jump) total += l;
  }
  if (total < 0.4 * spacing) return out;
  let next = Math.min(spacing / 2, total / 2), run = 0;
  for (let k = 1; k < pixels.length; k++) {
    const [x0, y0] = pixels[k - 1], [x1, y1] = pixels[k], l = Math.hypot(x1 - x0, y1 - y0);
    if (!(l > 0) || l >= jump) continue;
    while (run + l >= next) {
      const f = (next - run) / l;
      out.push([x0 + f * (x1 - x0), y0 + f * (y1 - y0), (x1 - x0) / l, (y1 - y0) / l]);
      next += spacing;
    }
    run += l;
  }
  return out;
}

/** Quiver stations at native mesh nodes: rows about `spacing`° apart, and along each row about
 * `spacing`° of arc apart (the longitude stride widens as 1/cos φ, so arrows do not crowd toward
 * the poles). Pole rows are skipped: east and north are undefined there. Masked nodes are skipped.
 * Returns [{lat, lon, u, v}] for rows south → north, columns west → east (closing meridian excluded). */
export function quiverStations(lat, lon, u, v, spacing) {
  const nlat = lat.length, nlon = lon.length, out = [];
  const dlat = Math.abs(lat[1] - lat[0]) || spacing, dlon = Math.abs(lon[1] - lon[0]) || spacing;
  const rowStride = Math.max(1, Math.round(spacing / dlat));
  const first = Math.floor(rowStride / 2);
  for (let j = first; j < nlat; j += rowStride) {
    const y = lat[j];
    if (Math.abs(y) > 90 - spacing / 2) continue;
    const stride = Math.max(1, Math.round(spacing / (dlon * Math.cos(y * DEG))));
    const count = Math.max(1, Math.floor(nlon / stride));
    for (let n = 0; n < count; n++) {
      const i = Math.round(n * nlon / count + (Math.floor((j - first) / rowStride) % 2) * nlon / count / 2) % nlon;
      const k = j * nlon + i;
      if (Number.isFinite(u[k]) && Number.isFinite(v[k])) out.push({lat: y, lon: lon[i], u: u[k], v: v[k]});
    }
  }
  return out;
}
