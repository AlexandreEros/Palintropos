import {DEG, clamp, direction, interpolate, sampleAt, usable} from "./mesh.js";

// Same construction as the standalone sphere streamlines: equal-area
// Fibonacci seeds, tangent east/north wind in Cartesian coordinates, and
// normalized RK4 stepping in both directions. Sample the actual coordinates,
// including Gaussian latitudes and the periodic seam.
function geographic(p) {
  return [Math.asin(clamp(p[1],-1,1))/DEG,Math.atan2(p[0],p[2])/DEG];
}
function normalize(v) {
  const n=Math.hypot(...v);
  return v.map(x=>x/n);
}
function advance(p,distance,d) {
  return normalize(p.map((x,i)=>x+distance*d[i]));
}

// Occupancy bins of about `separation` degrees on the sphere (longitude bins
// widen toward the poles so each bin spans a similar distance).
function binner(separation) {
  const rows=Math.max(1,Math.round(180/separation));
  const counts=Array.from({length:rows},(_,j)=>{
    const c=Math.cos(((j+0.5)*180/rows-90)*DEG);
    return Math.max(1,Math.round(360*c/separation));
  });
  return (lat,lon)=>{
    const j=Math.min(rows-1,Math.max(0,Math.floor((lat+90)/180*rows)));
    const n=counts[j], i=Math.min(n-1,Math.floor((((lon%360)+360)%360)/360*n));
    return j*4096+i;
  };
}

/** Streamlines of frame.u, frame.v: [[lat, lon], …] per line, upstream end first.
 * options (the globe uses the defaults): seeds (Fibonacci seed count, 80),
 * maxArc (degrees traced each way from a seed, 18), maxSteps (each way, 96),
 * separation (degrees; when set, a line stops where it would enter a bin
 * another line occupies, and a seed in an occupied bin starts no line, so the
 * lines are evenly spaced instead of crowding: null). */
export function streamlines(frame, shell, options={}) {
  if (!frame.u || !frame.v) return [];
  const seeds=options.seeds ?? 80, maxArc=options.maxArc ?? 18, maxSteps=options.maxSteps ?? 96;
  const bin=options.separation ? binner(options.separation) : null, owner=new Map();
  const cartesian=Array.from({length:3},()=>new Float32Array(frame.values.length));
  let maxSpeed=0, minCell=360/frame.lon.length;
  for (let j=0;j<frame.lat.length;j++) {
    if (j) minCell=Math.min(minCell,frame.lat[j]-frame.lat[j-1]);
    const a=frame.lat[j]*DEG;
    for (let i=0;i<frame.lon.length;i++) {
      const k=j*frame.lon.length+i;
      if (!usable(frame,shell,k)) continue;
      const u=frame.u[k], v=frame.v[k], b=frame.lon[i]*DEG;
      cartesian[0][k]=u*Math.cos(b)-v*Math.sin(a)*Math.sin(b);
      cartesian[1][k]=v*Math.cos(a);
      cartesian[2][k]=-u*Math.sin(b)-v*Math.sin(a)*Math.cos(b);
      if (Number.isFinite(u+v)) maxSpeed=Math.max(maxSpeed,Math.hypot(u,v));
    }
  }
  const threshold=Math.max(1e-9,maxSpeed*0.003);
  if (maxSpeed<=threshold) return [];
  const step=Math.min(1.8,minCell*0.45)*DEG;
  function tangent(p) {
    const t=sampleAt(frame,...geographic(p));
    if (!t || !usable(frame,shell,t[0]) || !usable(frame,shell,t[1]) || !usable(frame,shell,t[2])) return null;
    const w=cartesian.map(component=>interpolate(component,t));
    const radial=w[0]*p[0]+w[1]*p[1]+w[2]*p[2];
    const d=w.map((v,i)=>v-radial*p[i]), s=Math.hypot(...d);
    return s>threshold ? d.map(v=>v/s) : null;
  }
  // With a separation, a line also stops where it returns to a bin it crossed at least `gap`
  // steps earlier along itself: a closed orbit ends once it has closed, a spiral is cut.
  const gap=bin ? Math.ceil(1.5*options.separation*DEG/step)+1 : 0;
  let visits=null;
  function trace(seed, h, id) {
    const points=[], sign=h>0 ? 1 : -1;
    let p=seed;
    const count=Math.min(maxSteps,Math.ceil(maxArc*DEG/step));
    if (bin && sign<0) visits=new Map([[bin(...geographic(seed)),0]]);
    for (let s=0;s<count;s++) {
      const a=tangent(p);
      if (!a) break;
      const b=tangent(advance(p,h/2,a));
      if (!b) break;
      const c=tangent(advance(p,h/2,b));
      if (!c) break;
      const d=tangent(advance(p,h,c));
      if (!d) break;
      const next=advance(p,h/6,a.map((v,i)=>v+2*b[i]+2*c[i]+d[i]));
      if (!tangent(next)) break;
      const g=geographic(next);
      if (bin) {
        const key=bin(...g), b=owner.get(key), at=sign*(s+1), seen=visits.get(key);
        if (b!==undefined && b!==id) break;
        if (seen!==undefined && Math.abs(at-seen)>=gap) break;
        if (seen===undefined) visits.set(key,at);
      }
      points.push(g);
      p=next;
    }
    return points;
  }
  const lines=[], golden=180*(3-Math.sqrt(5));
  for (let i=0;i<seeds;i++) {
    const lat=Math.asin(1-2*(i+0.5)/seeds)/DEG, lon=(i*golden)%360-180, seed=direction(lat,lon);
    if (bin && owner.has(bin(lat,lon))) continue;
    const line=[...trace(seed,-step,lines.length).reverse(),[lat,lon],...trace(seed,step,lines.length)];
    if (line.length>=3) {
      if (bin) for (const [y,x] of line) if (!owner.has(bin(y,x))) owner.set(bin(y,x),lines.length);
      lines.push(line);
    }
  }
  return lines;
}

export function globeLines(frame, options, winds) {
  const grids=[], flow=[];
  const radiusScale=options.exaggeration/options.radius;
  function at(lat,lon,wind) {
    const sample=sampleAt(frame,lat,lon);
    if (!sample) return null;
    const fieldValid=[sample[0],sample[1],sample[2]].every(k=>usable(frame,options.shell,k));
    if (!fieldValid && (wind || !options.shell)) return null;
    let r=1;
    if (options.shell) r+=radiusScale*interpolate(fieldValid ? frame.height : frame.terrain,sample);
    return direction(lat,lon).map(x=>x*r*1.001);
  }
  function path(points,wind) {
    const dest=wind ? flow : grids;
    let previous=null;
    for (const point of points) {
      const p=at(...point,wind);
      if (previous && p) dest.push(...previous,...p);
      previous=p;
    }
  }
  for (let x=-180;x<180;x+=30) path(Array.from(frame.lat,lat=>[lat,x]),false);
  for (let y=-60;y<=60;y+=30) {
    const row=Array.from(frame.lon,lon=>[y,lon]);
    row.push([y,frame.lon[0]+360]);
    path(row,false);
  }
  // The linked cursor: its meridian and parallel are the planes of the
  // meridional and zonal sections.
  const cursor=options.cursor, marks=[];
  if (cursor && Number.isFinite(cursor.lat) && Number.isFinite(cursor.lon)) {
    const lon=(cursor.lon+540)%360-180;
    const meridian=[], parallel=[];
    for (let lat=-90;lat<=90;lat+=1) meridian.push([lat,lon]);
    for (let x=0;x<=360;x+=1) parallel.push([cursor.lat,x]);
    for (const line of [meridian,parallel]) {
      let previous=null;
      for (const point of line) {
        const sample=sampleAt(frame,...point);
        let r=1.004;
        if (options.shell && sample) r+=1.004*radiusScale*interpolate(frame.height,sample);
        const p=direction(...point).map(x=>x*r);
        if (previous) marks.push(...previous,...p);
        previous=p;
      }
    }
  }
  for (const line of winds) {
    path(line,true);
    const m=Math.floor(line.length/2), p=line[m], prev=line[m-1];
    const cos=Math.max(0.1,Math.cos(p[0]*DEG));
    const dx=(((p[1]-prev[1]+540)%360)-180)*cos, dy=p[0]-prev[0], speed=Math.hypot(dx,dy);
    if (!speed) continue;
    const east=dx/speed, north=dy/speed;
    for (const side of [-1,1]) path([p,[p[0]-2*north+side*east,p[1]+(-2*east-side*north)/cos]],true);
  }
  return [new Float32Array(grids),new Float32Array(flow),new Float32Array(marks)];
}
