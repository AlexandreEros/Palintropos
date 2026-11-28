import {Globe,homeCamera} from "./globe.js";
import {DEG,clamp,fieldScale,interpolate,surfaceLayers,usable} from "./mesh.js";
import {globeLines,streamlines} from "./overlays.js";
import {numberText} from "./map.js";

function make(tag,className,text) {
  const element=document.createElement(tag);
  element.className=className;
  if (text) element.textContent=text;
  return element;
}
// The node (cell centre) nearest a point: rows by latitude, columns periodic in longitude.
function nearestNode(frame,lat,lon) {
  let j=0, i=0, best=Infinity;
  frame.lat.forEach((y,k)=>{ const d=Math.abs(y-lat); if (d<best) { best=d; j=k; } });
  best=Infinity;
  frame.lon.forEach((x,k)=>{ const d=Math.abs(((x-lon)%360+540)%360-180); if (d<best) { best=d; i=k; } });
  return j*frame.lon.length+i;
}
function checkFrame(frame,options) {
  const n=frame.lat.length*frame.lon.length;
  if (frame.lat.length<2 || frame.lon.length<3 || frame.values.length!==n)
    throw new Error("Surface data must be a latitude × longitude grid.");
  for (const name of ["terrain","height","mask","u","v"]) {
    if (frame[name] && frame[name].length!==n) throw new Error("Inconsistent grid shape for "+name);
  }
  if (options.shell && (!frame.terrain || !frame.height)) throw new Error("Layer geometry requires terrain and upper-surface heights.");
  if (!(options.radius>0) || !Number.isFinite(options.radius)) throw new Error("Planet radius must be positive.");
}

/** The globe: one saved state on one surface. All lengths are in metres;
 * latitude and longitude are degrees. Mask zero denotes valid data. draw()
 * accepts options.range [min,max] for comparable panels. png() returns a Blob
 * of the globe as drawn, with its title, selection and colour scale, which is
 * also shown live beneath the globe (drawLegend). The flat
 * longitude × latitude projection is the Explorer's: there is no second map.
 */
export class SurfaceView {
  constructor(container) {
    this.root=make("div","surface-view");
    this.canvas=make("canvas","globe-canvas");
    this.canvas.setAttribute("aria-label","Globe. Drag or use the arrow keys to rotate. Scroll or use plus and minus to zoom. Click to select a location.");
    this.canvas.tabIndex=0;
    this.canvas.style.cssText="width:100%;height:clamp(220px,32vw,415px);display:block;touch-action:none;background:#141d24";
    const toolbar=make("div","surface-camera");
    const reset=make("button","reset-camera","Reset camera");
    reset.type="button";
    this.description=make("span","surface-light");
    this.description.style.marginInlineStart="0.7em";
    toolbar.append(reset,this.description);
    this.readout=make("div","surface-readout");
    this.readout.style.minHeight="1.5em";
    // The live colour scale of the field as drawn: its quantity, units and numerical endpoints.
    this.legend=make("div","surface-legend");
    this.legend.style.cssText="display:grid;grid-template-columns:auto 1fr auto;gap:0.2em 0.5em;align-items:center;font-size:0.85em;margin:0.3em 0";
    this.legendLabel=make("span","surface-legend-label");
    this.legendLabel.style.gridColumn="1 / -1";
    this.legendMin=make("span","surface-legend-min");
    this.legendBar=make("div","surface-legend-bar");
    this.legendBar.style.cssText="height:10px";
    this.legendBar.setAttribute("aria-hidden","true");
    this.legendMax=make("span","surface-legend-max");
    this.legend.append(this.legendLabel,this.legendMin,this.legendBar,this.legendMax);
    this.legend.hidden=true;
    this.root.append(this.canvas,toolbar,this.legend,this.readout);
    container.appendChild(this.root);
    this.globe=new Globe(this.canvas);
    this.listeners=new AbortController();
    this.home=homeCamera();
    this.on(reset,"click",()=>{ this.globe.camera={...this.home}; this.schedule(); });
    this.installInteraction();
    this.observer=new ResizeObserver(()=>{ if (this.scene) this.schedule(); });
    this.observer.observe(this.root); this.observer.observe(this.canvas);
  }
  on(element,type,handler,more={}) {
    element.addEventListener(type,handler,{signal:this.listeners.signal,...more});
  }
  schedule() {
    if (this.animation) return;
    this.animation=requestAnimationFrame(()=>{ this.animation=0; this.globe.render(); });
  }
  draw(frame,opts={}) {
    const options={radius:1,exaggeration:1,opacity:0.75,shell:false,cursor:this.cursor,...opts};
    options.opacity=clamp(options.opacity,0,1);
    if (options.star && (!Number.isFinite(options.star.latitude) || !Number.isFinite(options.star.longitude))) options.star=null;
    checkFrame(frame,options);
    if (!this.scene && Number.isFinite(options.center?.latitude) && Number.isFinite(options.center?.longitude)) {
      this.home.pitch=clamp(options.center.latitude,-90,90)*DEG;
      this.home.yaw=-options.center.longitude*DEG;
      this.globe.camera={...this.home};
    }
    const scale=fieldScale(frame,options);
    const winds=options.winds ? streamlines(frame,options.shell) : [];
    const layers=surfaceLayers(frame,options,scale), lines=globeLines(frame,options,winds);
    this.scene={frame,options,scale,winds,layers,lines};
    this.range=scale.range;
    this.globe.setScene(this.scene);
    this.root.dataset.backend=this.globe.backend;
    this.description.textContent=options.star ?
      "Geometric day/night shading darkens the globe only; colours, map and readouts keep model values" :
      "Neutral display lighting";
    this.readout.textContent="";
    this.drawLegend();
  }
  drawLegend() {
    const {options,scale}=this.scene;
    this.legendLabel.textContent=[options.label||options.field,options.units].filter(Boolean).join(" · ");
    const shown=Boolean(scale.range);
    this.legendMin.hidden=this.legendMax.hidden=this.legendBar.hidden=!shown;
    if (shown) {
      // The scale's own colour table, sampled at 33 evenly spaced stops.
      const stops=Array.from({length:33},(_,i)=>{ const [r,g,b]=scale.lut[Math.round(255*i/32)];
        return "rgb("+r+","+g+","+b+") "+(100*i/32).toFixed(2)+"%"; });
      this.legendBar.style.background="linear-gradient(to right,"+stops.join(",")+")";
      this.legendMin.textContent=numberText(scale.range[0]);
      this.legendMax.textContent=numberText(scale.range[1]);
      this.legend.setAttribute("aria-label","Colour scale "+this.legendLabel.textContent+" from "+
        this.legendMin.textContent+" to "+this.legendMax.textContent);
    }
    this.legend.hidden=false;
  }
  /** Turn the globe to face a point (degrees). */
  face(lat,lon) {
    this.globe.camera.pitch=clamp(lat,-90,90)*DEG; this.globe.camera.yaw=-lon*DEG; this.schedule();
  }
  /** Mark the linked geographic cursor (degrees) on the globe and map. */
  setCursor(cursor) {
    this.cursor=cursor;
    if (!this.scene) return;
    this.scene.options.cursor=cursor;
    this.scene.lines=globeLines(this.scene.frame,this.scene.options,this.scene.winds);
    this.globe.setScene(this.scene);
  }
  installInteraction() {
    const c=this.canvas, globe=this.globe;
    let pressed=null;
    this.on(c,"pointerdown",e=>{ pressed=[e.clientX,e.clientY]; });
    this.on(c,"pointerup",e=>{
      if (!pressed || Math.hypot(e.clientX-pressed[0],e.clientY-pressed[1])>4 || !this.scene) return;
      const hit=globe.pick(e.clientX,e.clientY);
      if (hit) this.onSelect?.({lat:hit.lat,lon:hit.lon});
    });
    const zoom=f=>{ globe.camera.distance=clamp(globe.camera.distance*f,1.4,9); this.schedule(); };
    let drag=null;
    this.on(c,"pointerdown",e=>{
      if (e.button!==0) return;
      c.setPointerCapture(e.pointerId); drag=[e.clientX,e.clientY];
    });
    this.on(c,"pointermove",e=>{
      if (!drag) {
        if (this.scene) this.show(globe.pick(e.clientX,e.clientY));
        return;
      }
      const step=3/c.clientHeight;
      globe.camera.yaw+=(e.clientX-drag[0])*step;
      globe.camera.pitch=clamp(globe.camera.pitch+(e.clientY-drag[1])*step,-Math.PI/2,Math.PI/2);
      drag=[e.clientX,e.clientY]; this.schedule();
    });
    this.on(c,"lostpointercapture",()=>{drag=null;});
    this.on(c,"pointerleave",()=>this.show(null));
    this.on(c,"wheel",e=>{e.preventDefault(); zoom(e.deltaY>0 ? 1.1 : 1/1.1);},{passive:false});
    this.on(c,"keydown",e=>{
      if (!["ArrowLeft","ArrowRight","ArrowUp","ArrowDown","+","=","-","Home"].includes(e.key)) return;
      e.preventDefault();
      const camera=globe.camera;
      if (e.key==="Home") globe.camera={...this.home};
      if (e.key==="-" || e.key==="+" || e.key==="=") zoom(e.key==="-" ? 1.1 : 1/1.1);
      if (e.key==="ArrowLeft") camera.yaw+=5*DEG;
      if (e.key==="ArrowRight") camera.yaw-=5*DEG;
      if (e.key==="ArrowUp") camera.pitch=clamp(camera.pitch+5*DEG,-Math.PI/2,Math.PI/2);
      if (e.key==="ArrowDown") camera.pitch=clamp(camera.pitch-5*DEG,-Math.PI/2,Math.PI/2);
      this.schedule();
    });
  }
  show(hit) {
    if (!hit || !this.scene) { this.readout.textContent=""; return; }
    const {frame,options}=this.scene,{lat,lon,t}=hit;
    const parts=[Math.abs(lat).toFixed(2)+"°"+(lat<0?"S":"N")+", "+Math.abs(lon).toFixed(2)+"°"+(lon<0?"W":"E")];
    if (!t && !frame.cells) parts.push("Outside sampled latitudes");
    else {
      const invalid=frame.cells ? [] : t.slice(0,3).filter(k=>!usable(frame,options.shell,k));
      if (invalid.length) {
        const code=frame.mask?.[invalid[0]] || 0;
        parts.push(options.maskLabels?.[code] || (code ? "Masked (code "+code+")" : "No valid sample"));
      } else if (!hit.terrain && frame.cells) {
        const k=nearestNode(frame,lat,lon);
        parts.push((options.label || options.field || "Field")+" "+numberText(frame.values[k])+" "+(options.units||"")+
          " · archived cell at "+numberText(frame.lat[Math.floor(k/frame.lon.length)])+"°, "+numberText(frame.lon[k%frame.lon.length])+"°E");
      } else if (!hit.terrain) parts.push((options.label || options.field || "Field")+" "+numberText(interpolate(frame.values,t))+" "+(options.units||""));
      if (options.shell) {
        parts.push("terrain "+numberText(interpolate(frame.terrain,t))+" m");
        if (!invalid.length && !hit.terrain) parts.push("layer height "+numberText(interpolate(frame.height,t))+" m");
      }
    }
    this.readout.textContent=parts.join(" · ");
  }
  png() {
    if (!this.scene) return Promise.reject(new Error("No field selected."));
    // Render now and copy in the same task: the WebGL drawing buffer is not preserved.
    this.globe.render();
    const {options,scale}=this.scene, src=this.canvas, w=Math.max(1000,src.width), gh=Math.round(src.height*w/src.width);
    const lines=[String(options.title||""), typeof options.selection==="string" ? options.selection : ""].filter(Boolean);
    const canvas=make("canvas",""), ctx=canvas.getContext("2d");
    canvas.width=w; canvas.height=gh+lines.length*22+86;
    ctx.fillStyle="#f6f5ef"; ctx.fillRect(0,0,canvas.width,canvas.height);
    ctx.fillStyle="#242f32"; ctx.textAlign="left";
    lines.forEach((line,i)=>{ ctx.font=i ? "14px system-ui" : "600 16px system-ui"; ctx.fillText(line,14,22+22*i,w-28); });
    const top=lines.length*22+12;
    ctx.drawImage(src,0,top,w,gh);
    const y=top+gh+34, x0=20, bw=w-40;
    ctx.font="600 13px system-ui"; ctx.fillText([options.label||options.field,options.units].filter(Boolean).join(" · "),x0,y-8,bw);
    if (scale.range) {
      for (let x=0;x<bw;x++) {
        const [r,g,b]=scale.lut[Math.round(255*clamp(x/(bw-1),0,1))];
        ctx.fillStyle="rgb("+r+","+g+","+b+")"; ctx.fillRect(x0+x,y,1.1,10);
      }
      ctx.fillStyle="#263437"; ctx.font="12px system-ui";
      ctx.fillText(numberText(scale.range[0]),x0,y+26); ctx.textAlign="right"; ctx.fillText(numberText(scale.range[1]),x0+bw,y+26);
    }
    return new Promise((resolve,reject)=>canvas.toBlob(blob=>blob ? resolve(blob) : reject(new Error("PNG encoding failed.")),"image/png"));
  }
  destroy() {
    this.listeners.abort(); this.observer.disconnect(); cancelAnimationFrame(this.animation);
    this.globe.destroy(); this.scene=null; this.root.remove();
  }
}
