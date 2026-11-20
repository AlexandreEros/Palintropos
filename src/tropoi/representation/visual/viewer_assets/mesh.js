// Geographic coordinates and displaced surfaces. The coordinate basis, outward
// winding and tangent normals follow terrain_globe.html. The provider supplies
// increasing latitude and periodic longitude without a duplicate seam column.
export const DEG = Math.PI / 180;
export const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
export const wrap = lon => ((lon % 360) + 360) % 360;
export const dot = (a, b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
export function direction(lat, lon) {
  const a = lat * DEG, b = lon * DEG, c = Math.cos(a);
  return [c * Math.sin(b), Math.sin(a), c * Math.cos(b)];
}

// Signed fields share a symmetric blue/ivory/red scale unless bounds are
// supplied explicitly. Sequential fields use a dark-purple to yellow scale.
const signed = [[46,72,147],[84,147,196],[182,218,224],[244,240,223],[231,176,134],[192,92,74],[136,42,65]];
const sequential = [[50,35,81],[65,72,129],[59,112,147],[46,150,145],[106,181,122],[187,205,98],[247,224,97]];
const relief = [[66,88,95],[102,121,119],[145,148,130],[190,174,144],[229,220,191]];

export function palette(stops, t) {
  const p = clamp(t, 0, 1) * (stops.length-1);
  const a = Math.floor(p), b = Math.min(a+1, stops.length-1), f = p-a;
  return stops[a].map((c, i) => Math.round(c + f*(stops[b][i]-c)));
}

export function usable(frame, shell, k) {
  if (frame.mask?.[k] || !Number.isFinite(frame.values[k])) return false;
  return !shell || (Number.isFinite(frame.height[k]) && Number.isFinite(frame.terrain[k]) &&
                    frame.height[k] >= frame.terrain[k]);
}

export function fieldScale(frame, options) {
  let lo = Infinity, hi = -Infinity;
  for (let k=0; k<frame.values.length; k++) {
    if (!usable(frame, options.shell, k)) continue;
    lo = Math.min(lo, frame.values[k]);
    hi = Math.max(hi, frame.values[k]);
  }
  const diverging = options.scale === "diverging" || /vorticity|divergence|^(u|v|zeta|delta|phi)$/.test(options.field ?? "");
  if (options.range) [lo, hi] = options.range;
  else if (diverging && hi >= lo && hi !== lo) {
    hi = Math.max(Math.abs(lo), Math.abs(hi));
    lo = -hi;
  }
  const range = Number.isFinite(lo) && Number.isFinite(hi) && hi >= lo ? [lo, hi] : null;
  const stops = diverging ? signed : sequential;
  const lut = Array.from({length:256}, (_, i) => palette(stops, i/255));
  const color = v => lut[hi > lo ? Math.round(255 * clamp((v-lo)/(hi-lo), 0, 1)) : 128];
  return {range, lut, color};
}

// Split cells on the same southwest/northeast diagonal as the globe. The
// sampled triangle drives the map, cursor, streamline and mask decisions.
function lower(axis, x) {
  let lo=0, hi=axis.length;
  while (lo<hi) {
    const mid = (lo+hi) >>> 1;
    if (axis[mid] <= x) lo=mid+1;
    else hi=mid;
  }
  return Math.max(0,lo-1);
}

export function latitudeCell(lat, y) {
  const j = Math.min(lower(lat, y), lat.length-2);
  return [j, clamp((y-lat[j])/(lat[j+1]-lat[j]), 0, 1)];
}
export function longitudeCell(lon, x) {
  const angle = wrap(x-lon[0])+lon[0], i=lower(lon, angle);
  const east = i+1 === lon.length ? lon[0]+360 : lon[i+1];
  return [i,(angle-lon[i])/(east-lon[i])];
}
export function triangle(width, i, j, fx, fy) {
  const a=j*width+i, b=j*width+(i+1)%width, d=b+width;
  return fx >= fy ? [a,b,d,1-fx,fx-fy,fy] : [a,d,a+width,1-fy,fx,fy-fx];
}
export function sampleAt(frame, lat, lon) {
  if (lat<frame.lat[0] || lat>frame.lat[frame.lat.length-1]) return null;
  const [i,fx]=longitudeCell(frame.lon,lon), [j,fy]=latitudeCell(frame.lat,lat);
  return triangle(frame.lon.length,i,j,fx,fy);
}
export const interpolate = (values,t) => t[3]*values[t[0]] + t[4]*values[t[1]] + t[5]*values[t[2]];

export function buildSurface(frame, heights, options, valid, color) {
  const {lat,lon}=frame, cols=lon.length, count=lat.length*cols;
  const positions=new Float32Array(count*3), colors=new Uint8Array(count*3);
  const normals=new Float32Array(count*3), indices=[];
  const scale=options.exaggeration/options.radius;
  for (let j=0;j<lat.length;j++) for (let i=0;i<cols;i++) {
    const k=j*cols+i, unit=direction(lat[j],lon[i]);
    const r=1+scale*(heights?.[k] || 0);
    for (let q=0;q<3;q++) positions[3*k+q]=r*unit[q];
    colors.set(color(k),3*k);
    if (j===lat.length-1) continue;
    const east=j*cols+(i+1)%cols, north=k+cols, ne=east+cols;
    if (valid(k) && valid(east) && valid(ne)) indices.push(k,east,ne);
    if (valid(k) && valid(ne) && valid(north)) indices.push(k,ne,north);
  }
  for (let j=0;j<lat.length;j++) for (let i=0;i<cols;i++) {
    const k=j*cols+i, east=3*(j*cols+(i+1)%cols), west=3*(j*cols+(i+cols-1)%cols);
    const north=3*(Math.min(j+1,lat.length-1)*cols+i), south=3*(Math.max(j-1,0)*cols+i);
    const ex=positions[east]-positions[west], ey=positions[east+1]-positions[west+1], ez=positions[east+2]-positions[west+2];
    const nx=positions[north]-positions[south], ny=positions[north+1]-positions[south+1], nz=positions[north+2]-positions[south+2];
    let n=[ey*nz-ez*ny,ez*nx-ex*nz,ex*ny-ey*nx], len=Math.hypot(...n);
    if (!(len>1e-9)) { n=direction(lat[j],lon[i]); len=1; }
    for (let q=0;q<3;q++) normals[3*k+q]=n[q]/len;
  }
  return {positions,normals,colors,indices:count>65535 ? new Uint32Array(indices) : new Uint16Array(indices)};
}

export function surfaceLayers(frame, options, scale) {
  const layers=[];
  if (options.shell) {
    let min=Infinity,max=-Infinity;
    for (const h of frame.terrain) if (Number.isFinite(h)) { min=Math.min(min,h); max=Math.max(max,h); }
    const groundColor=k=>palette(relief,max>min && Number.isFinite(frame.terrain[k]) ? (frame.terrain[k]-min)/(max-min) : 0.45);
    layers.push(buildSurface(frame,frame.terrain,options,k=>Number.isFinite(frame.terrain[k]),groundColor));
  }
  const field=buildSurface(frame,frame.height,options,
    k=>usable(frame,options.shell,k), k=>Number.isFinite(frame.values[k]) && scale.range ? scale.color(frame.values[k]) : [0,0,0]);
  layers.push(field);
  return layers;
}
