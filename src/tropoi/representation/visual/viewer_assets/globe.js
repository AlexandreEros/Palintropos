// Perspective and orbit convention from terrain_globe.html. The WebGL and
// software paths consume identical indexed meshes, camera and line geometry.
import {DEG, clamp, direction} from "./mesh.js";
const F=1/Math.tan(17.5*DEG), BG=[20,29,36];
export const homeCamera = () => ({yaw:-20*DEG,pitch:14*DEG,distance:3.8});
function matrices(camera, aspect) {
  const cy=Math.cos(camera.yaw),sy=Math.sin(camera.yaw),cp=Math.cos(camera.pitch),sp=Math.sin(camera.pitch);
  const mv=[cy,sp*sy,-cp*sy,0, 0,cp,sp,0, sy,-sp*cy,cp*cy,0, 0,0,-camera.distance,1];
  const zNear=0.04, zFar=2048, nf=1/(zNear-zFar);
  return {mv,projection:[F/aspect,0,0,0,0,F,0,0,0,0,(zFar+zNear)*nf,-1,0,0,2*zNear*zFar*nf,0]};
}
function light(mv, star) {
  if (star) return direction(star.latitude,star.longitude);
  // Transform the camera-fixed neutral lamp into the model coordinate basis.
  const lamp=[-.45,.55,.70], norm=Math.hypot(...lamp);
  return [0,4,8].map(i=>(mv[i]*lamp[0]+mv[i+1]*lamp[1]+mv[i+2]*lamp[2])/norm);
}
function project(x,y,z,mv,width,height) {
  const depth=-(mv[2]*x+mv[6]*y+mv[10]*z+mv[14]);
  const s=height*F/(2*depth);
  return [width/2+s*(mv[0]*x+mv[4]*y+mv[8]*z),height/2-s*(mv[1]*x+mv[5]*y+mv[9]*z),1/depth];
}
function projected(positions,mv,w,h) {
  const out=new Float32Array(positions.length);
  for (let k=0;k<out.length;k+=3) out.set(project(positions[k],positions[k+1],positions[k+2],mv,w,h),k);
  return out;
}

const vertex=`
attribute vec3 pos, normal, color;
uniform mat4 view, projection;
uniform vec3 light;
uniform float stellar;
varying vec3 rgb;
void main(){
  gl_Position=projection*view*vec4(pos,1.0);
  gl_PointSize=6.0;
  vec3 n=normalize(mix(normal,normalize(pos),stellar));
  float day=max(0.0,dot(n,light));
  float shade=mix(0.55+0.45*day,0.18+0.82*day,stellar);
  rgb=color*shade;
}`;
const fragment=`
precision mediump float;
uniform vec4 ink;
uniform float alpha;
varying vec3 rgb;
void main(){ gl_FragColor=ink.a>0.0 ? ink : vec4(rgb,alpha); }`;

function createProgram(gl) {
  const program=gl.createProgram();
  for (const [type,source] of [[gl.VERTEX_SHADER,vertex],[gl.FRAGMENT_SHADER,fragment]]) {
    const shader=gl.createShader(type);
    gl.shaderSource(shader,source); gl.compileShader(shader);
    if (!gl.getShaderParameter(shader,gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
    gl.attachShader(program,shader); gl.deleteShader(shader);
  }
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program,gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  return program;
}
class WebGLGlobe {
  constructor(gl) {
    this.gl=gl;
    this.uint=gl.getExtension("OES_element_index_uint");
    this.program=createProgram(gl);
    this.attributes={}; this.uniforms={}; this.buffers=[];
    for (const name of ["pos","normal","color"]) this.attributes[name]=gl.getAttribLocation(this.program,name);
    for (const name of ["view","projection","light","stellar","alpha","ink"]) this.uniforms[name]=gl.getUniformLocation(this.program,name);
  }
  buffer(data,target) {
    const gl=this.gl, buffer=gl.createBuffer();
    gl.bindBuffer(target,buffer); gl.bufferData(target,data,gl.STATIC_DRAW);
    this.buffers.push(buffer);
    return buffer;
  }
  clearBuffers() {
    for (const buffer of this.buffers) this.gl.deleteBuffer(buffer);
    this.buffers=[];
  }
  setScene(scene) {
    this.clearBuffers();
    const gl=this.gl;
    this.surfaces=scene.layers.map(s=>{
      const wide=s.indices instanceof Uint32Array;
      if (wide && !this.uint) throw new Error("32-bit element indices unavailable");
      return {pos:this.buffer(s.positions,gl.ARRAY_BUFFER),normal:this.buffer(s.normals,gl.ARRAY_BUFFER),
        color:this.buffer(s.colors,gl.ARRAY_BUFFER),index:this.buffer(s.indices,gl.ELEMENT_ARRAY_BUFFER),
        type:wide ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT,count:s.indices.length};
    });
    this.lines=scene.lines.map(a=>({pos:this.buffer(a,gl.ARRAY_BUFFER),count:a.length/3}));
    const star=scene.options.star;
    this.star=star ? this.buffer(new Float32Array(direction(star.latitude,star.longitude).map(v=>v*1000)),gl.ARRAY_BUFFER) : null;
  }
  attr(name,buffer,type) {
    const gl=this.gl, loc=this.attributes[name];
    gl.bindBuffer(gl.ARRAY_BUFFER,buffer);
    gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc,3,type,type===gl.UNSIGNED_BYTE,0,0);
  }
  ink(buffer,mode,count,color) {
    const gl=this.gl;
    this.attr("pos",buffer,gl.FLOAT);
    for (const name of ["normal","color"]) { gl.disableVertexAttribArray(this.attributes[name]); gl.vertexAttrib3f(this.attributes[name],0,0,1); }
    gl.uniform4fv(this.uniforms.ink,color);
    gl.drawArrays(mode,0,count);
  }
  render(scene,camera) {
    const gl=this.gl, u=this.uniforms, {mv,projection}=matrices(camera,gl.canvas.width/gl.canvas.height);
    gl.viewport(0,0,gl.canvas.width,gl.canvas.height);
    gl.clearColor(BG[0]/255,BG[1]/255,BG[2]/255,1); gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
    gl.useProgram(this.program);
    gl.uniformMatrix4fv(u.view,false,mv); gl.uniformMatrix4fv(u.projection,false,projection);
    gl.uniform3fv(u.light,light(mv,scene.options.star)); gl.uniform1f(u.stellar,scene.options.star ? 1 : 0);
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL);
    if (this.star) this.ink(this.star,gl.POINTS,1,[1,.86,.55,1]);
    gl.enable(gl.CULL_FACE); gl.cullFace(gl.BACK);
    gl.enable(gl.POLYGON_OFFSET_FILL); gl.polygonOffset(1,1);
    this.surfaces.forEach((s,i)=>{
      const alpha=i===1 ? scene.options.opacity : 1;
      if (!alpha) return;
      if (i===1) { gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA); }
      gl.uniform1f(u.alpha,alpha); gl.uniform4f(u.ink,0,0,0,0);
      this.attr("pos",s.pos,gl.FLOAT); this.attr("normal",s.normal,gl.FLOAT); this.attr("color",s.color,gl.UNSIGNED_BYTE);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,s.index); gl.drawElements(gl.TRIANGLES,s.count,s.type,0);
      gl.disable(gl.BLEND);
    });
    gl.disable(gl.POLYGON_OFFSET_FILL); gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
    this.lines.forEach((line,i)=>this.ink(line.pos,gl.LINES,line.count,i ? [1,1,1,.90] : [0.1,.13,.17,.38]));
    gl.disable(gl.BLEND);
  }
  destroy() { this.clearBuffers(); this.gl.deleteProgram(this.program); }
}

// Software renderer for machines with WebGL disabled. Reciprocal depth makes
// terrain occlusion and coincident surface tests agree with the camera model.
function softwareRender(canvas,scene,camera) {
  const w=canvas.width,h=canvas.height,ctx=canvas.getContext("2d");
  const pixels=ctx.createImageData(w,h), data=pixels.data, depth=new Float32Array(w*h);
  const {mv}=matrices(camera,w/h), lamp=light(mv,scene.options.star);
  const word=new Uint32Array(data.buffer); word.fill((255<<24)|(BG[2]<<16)|(BG[1]<<8)|BG[0]);
  function paint(index,r,g,b,alpha) {
    const k=index*4;
    data[k]+=(r-data[k])*alpha; data[k+1]+=(g-data[k+1])*alpha; data[k+2]+=(b-data[k+2])*alpha;
  }
  if (scene.options.star) {
    const [x,y,z]=project(...lamp.map(v=>v*1000),mv,w,h);
    if (z>0) for (let iy=Math.max(0,Math.floor(y)-2);iy<=Math.min(h-1,Math.floor(y)+2);iy++)
      for (let ix=Math.max(0,Math.floor(x)-2);ix<=Math.min(w-1,Math.floor(x)+2);ix++) paint(iy*w+ix,255,220,140,1);
  }
  scene.layers.forEach((s,layer)=>{
    const alpha=layer===1 ? scene.options.opacity : 1;
    if (!alpha) return;
    const p=projected(s.positions,mv,w,h), rgb=new Float32Array(p.length);
    for (let k=0;k<p.length;k+=3) {
      const n=scene.options.star ? s.positions : s.normals;
      const incidence=Math.max(0,(n[k]*lamp[0]+n[k+1]*lamp[1]+n[k+2]*lamp[2])/(Math.hypot(n[k],n[k+1],n[k+2])||1));
      const shade=scene.options.star ? .18+.82*incidence : .55+.45*incidence;
      for (let q=0;q<3;q++) rgb[k+q]=s.colors[k+q]*shade;
    }
    for (let t=0;t<s.indices.length;t+=3) {
      const a=3*s.indices[t],b=3*s.indices[t+1],c=3*s.indices[t+2];
      const ax=p[a],ay=p[a+1], bx=p[b],by=p[b+1], cx=p[c],cy=p[c+1];
      const det=(by-cy)*(ax-cx)+(cx-bx)*(ay-cy);
      if (det>=0 || Math.min(p[a+2],p[b+2],p[c+2])<=0) continue;
      const x0=Math.max(0,Math.floor(Math.min(ax,bx,cx))),x1=Math.min(w-1,Math.ceil(Math.max(ax,bx,cx)));
      const y0=Math.max(0,Math.floor(Math.min(ay,by,cy))),y1=Math.min(h-1,Math.ceil(Math.max(ay,by,cy)));
      for (let y=y0;y<=y1;y++) for (let x=x0;x<=x1;x++) {
        let u=((by-cy)*(x+.5-cx)+(cx-bx)*(y+.5-cy))/det;
        let v=((cy-ay)*(x+.5-cx)+(ax-cx)*(y+.5-cy))/det;
        let r=1-u-v;
        if (u<0 || v<0 || r<0) continue;
        u*=p[a+2]; v*=p[b+2]; r*=p[c+2];
        const z=u+v+r, index=y*w+x;
        if (z<depth[index]-1e-8) continue;
        depth[index]=z;
        paint(index,(u*rgb[a]+v*rgb[b]+r*rgb[c])/z,(u*rgb[a+1]+v*rgb[b+1]+r*rgb[c+1])/z,
          (u*rgb[a+2]+v*rgb[b+2]+r*rgb[c+2])/z,alpha);
      }
    }
  });
  scene.lines.forEach((line,idx)=>{
    const p=projected(line,mv,w,h), ink=idx ? [255,255,255,.9] : [25,33,43,.38];
    for (let k=0;k<p.length;k+=6) {
      const dx=p[k+3]-p[k],dy=p[k+4]-p[k+1],n=Math.max(1,Math.ceil(Math.max(Math.abs(dx),Math.abs(dy))));
      if (p[k+2]<=0 || p[k+5]<=0 || n>w+h) continue;
      for (let j=0;j<=n;j++) {
        const f=j/n,x=Math.round(p[k]+f*dx),y=Math.round(p[k+1]+f*dy),i=y*w+x;
        if (x>=0 && x<w && y>=0 && y<h && p[k+2]+f*(p[k+5]-p[k+2])>=depth[i]-0.00015) paint(i,...ink);
      }
    }
  });
  ctx.putImageData(pixels,0,0);
}

export class Globe {
  constructor(canvas) {
    this.canvas=canvas;
    const gl=canvas.getContext("webgl",{alpha:false,antialias:true});
    this.gpu=gl ? new WebGLGlobe(gl) : null;
    this.backend=this.gpu ? "WebGL" : "Canvas 2D";
    this.camera=homeCamera();
  }
  setScene(scene) {
    this.scene=scene;
    this.gpu?.setScene(scene);
    this.render();
  }
  render() {
    if (!this.scene) return;
    const canvas=this.canvas, rect=canvas.getBoundingClientRect();
    const ratio=this.gpu ? Math.min(devicePixelRatio,2) : 1;
    const width=Math.max(1,Math.round(rect.width*ratio)),height=Math.max(1,Math.round(rect.height*ratio));
    if (canvas.width!==width || canvas.height!==height) { canvas.width=width; canvas.height=height; }
    if (this.gpu) this.gpu.render(this.scene,this.camera);
    else softwareRender(canvas,this.scene,this.camera);
  }
  // Pick the nearest actually rendered triangle on the displaced geometry.
  // Mask holes expose terrain. Perspective-correct barycentric weights give
  // the scalar and geographic position even at strong exaggeration.
  pick(clientX,clientY) {
    if (!this.scene) return null;
    const rect=this.canvas.getBoundingClientRect(), x=clientX-rect.left,y=clientY-rect.top;
    const {mv}=matrices(this.camera,rect.width/rect.height);
    let best=0, result=null;
    this.scene.layers.forEach((layer,i)=>{
      if (i===1 && this.scene.options.opacity===0) return;
      const p=projected(layer.positions,mv,rect.width,rect.height);
      for (let k=0;k<layer.indices.length;k+=3) {
        const ia=layer.indices[k],ib=layer.indices[k+1],ic=layer.indices[k+2],a=3*ia,b=3*ib,c=3*ic;
        const det=(p[b+1]-p[c+1])*(p[a]-p[c])+(p[c]-p[b])*(p[a+1]-p[c+1]);
        if (det>=0) continue;
        let u=((p[b+1]-p[c+1])*(x-p[c])+(p[c]-p[b])*(y-p[c+1]))/det;
        let v=((p[c+1]-p[a+1])*(x-p[c])+(p[a]-p[c])*(y-p[c+1]))/det;
        let r=1-u-v;
        if (u<0 || v<0 || r<0) continue;
        u*=p[a+2]; v*=p[b+2]; r*=p[c+2];
        const depth=u+v+r;
        if (depth<best) continue;
        best=depth;
        const pos=layer.positions;
        const world=[0,1,2].map(q=>(u*pos[a+q]+v*pos[b+q]+r*pos[c+q])/depth);
        result={lat:Math.asin(clamp(world[1]/Math.hypot(...world),-1,1))/DEG,lon:Math.atan2(world[0],world[2])/DEG,
          terrain:this.scene.options.shell && i===0,t:[ia,ib,ic,u/depth,v/depth,r/depth]};
      }
    });
    return result;
  }
  destroy() { this.gpu?.destroy(); this.scene=null; }
}
