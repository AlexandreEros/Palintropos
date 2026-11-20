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

export function streamlines(frame, shell) {
  if (!frame.u || !frame.v) return [];
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
  function trace(seed, h) {
    const points=[];
    let p=seed;
    const count=Math.min(96,Math.ceil(18*DEG/step));
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
      points.push(geographic(next));
      p=next;
    }
    return points;
  }
  const lines=[], golden=180*(3-Math.sqrt(5)), seeds=80;
  for (let i=0;i<seeds;i++) {
    const lat=Math.asin(1-2*(i+0.5)/seeds)/DEG, lon=(i*golden)%360-180, seed=direction(lat,lon);
    const line=[...trace(seed,-step).reverse(),[lat,lon],...trace(seed,step)];
    if (line.length>=3) lines.push(line);
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
  for (const line of winds) {
    path(line,true);
    const m=Math.floor(line.length/2), p=line[m], prev=line[m-1];
    const cos=Math.max(0.1,Math.cos(p[0]*DEG));
    const dx=(((p[1]-prev[1]+540)%360)-180)*cos, dy=p[0]-prev[0], speed=Math.hypot(dx,dy);
    if (!speed) continue;
    const east=dx/speed, north=dy/speed;
    for (const side of [-1,1]) path([p,[p[0]-2*north+side*east,p[1]+(-2*east-side*north)/cos]],true);
  }
  return [new Float32Array(grids),new Float32Array(flow)];
}
