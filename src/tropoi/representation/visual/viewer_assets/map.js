import {clamp, latitudeCell, longitudeCell, usable, wrap} from "./mesh.js";

export function numberText(value) {
  return Number.isFinite(value) ? String(Number(value.toPrecision(6))) : "unavailable";
}

// Geographic cells vary in latitude on Gaussian grids. Rasterize the same
// two triangles as the globe and leave out-of-support pixels explicitly gray.
function raster(frame,options,scale,w,h) {
  const image=document.createElement("canvas");
  image.width=w; image.height=h;
  const ctx=image.getContext("2d"), pixels=ctx.createImageData(w,h), data=pixels.data;
  const width=frame.lon.length, valid=Uint8Array.from(frame.values,(_,k)=>usable(frame,options.shell,k));
  const cols=Array.from({length:w},(_,x)=>longitudeCell(frame.lon,-180+360*(x+.5)/w));
  const values=frame.values;
  for (let y=0;y<h;y++) {
    const lat=90-180*(y+.5)/h, [j,fy]=latitudeCell(frame.lat,lat);
    // Native cells cover the sphere: polar half-rows belong to the polar cells.
    const outside=!frame.cells && (lat<frame.lat[0] || lat>frame.lat[frame.lat.length-1]);
    for (let x=0;x<w;x++) {
      const [i,fx]=cols[x],a=j*width+i,e=j*width+(i+1)%width,d=e+width;
      const b=fx>=fy ? e : d,c=fx>=fy ? d : a+width;
      let rgb=[159,166,167];
      if (frame.cells) {
        // Native cells: the value of the cell whose centre is nearest.
        const k=(fy<.5 ? j : j+1)*width+(fx<.5 ? i : (i+1)%width);
        if (!outside && valid[k]) rgb=scale.color(values[k]);
      } else if (!outside && valid[a] && valid[b] && valid[c]) {
        const u=1-Math.max(fx,fy), v=fx>=fy ? fx-fy : fx, t=1-u-v;
        rgb=scale.color(u*values[a]+v*values[b]+t*values[c]);
      }
      const k=4*(y*w+x);
      data[k]=rgb[0]; data[k+1]=rgb[1]; data[k+2]=rgb[2]; data[k+3]=255;
    }
  }
  ctx.putImageData(pixels,0,0);
  return image;
}

function flow(ctx,lines,box) {
  const X=lon=>box.x+box.w*(wrap(lon+180)/360), Y=lat=>box.y+box.h*(90-lat)/180;
  ctx.save(); ctx.beginPath(); ctx.rect(box.x,box.y,box.w,box.h); ctx.clip();
  ctx.strokeStyle="rgba(255,255,255,.85)"; ctx.lineWidth=1;
  for (const line of lines) {
    ctx.beginPath();
    let prev=null;
    for (const [lat,lon] of line) {
      const x=X(lon),y=Y(lat);
      if (prev===null || Math.abs(x-prev)>box.w/2) ctx.moveTo(x,y);
      else ctx.lineTo(x,y);
      prev=x;
    }
    const m=Math.floor(line.length/2), [lat,lon]=line[m];
    const x=X(lon),y=Y(lat),dx=x-X(line[m-1][1]),dy=y-Y(line[m-1][0]),len=Math.hypot(dx,dy);
    if (len>0 && Math.abs(dx)<box.w/2) {
      const a=dx/len,b=dy/len;
      ctx.moveTo(x-4*a+2*b,y-4*b-2*a); ctx.lineTo(x,y); ctx.lineTo(x-4*a-2*b,y-4*b+2*a);
    }
    ctx.stroke();
  }
  ctx.restore();
}

function wrapText(ctx,text,maxWidth) {
  const lines=[];
  let current="";
  for (const word of text.split(/\s+/)) {
    if (current && ctx.measureText(current+" "+word).width>maxWidth) { lines.push(current); current=word; }
    else current=current ? current+" "+word : word;
  }
  if (current) lines.push(current);
  return lines;
}

export function drawMap(canvas,scene,cssWidth,ratio=1) {
  const {frame,options,scale,winds}=scene, w=Math.max(120,Math.round(cssWidth));
  const ctx=canvas.getContext("2d");
  const selection=typeof options.selection==="string" ? options.selection : JSON.stringify(options.selection ?? {});
  ctx.font="12px system-ui";
  const title=String(options.title || ""), titleLines=wrapText(ctx,title,w-40);
  const labels=wrapText(ctx,selection,w-40), hMap=Math.round((w-68)/2);
  const box={x:38,y:14+titleLines.length*19+labels.length*17,w:w-57,h:hMap};
  const h=box.y+hMap+88;
  canvas.width=Math.round(w*ratio); canvas.height=Math.round(h*ratio);
  ctx.scale(ratio,ratio); ctx.fillStyle="#f6f5ef"; ctx.fillRect(0,0,w,h);
  ctx.fillStyle="#242f32"; ctx.textAlign="left"; ctx.font="600 14px system-ui";
  let y=18;
  for (const line of titleLines) { ctx.fillText(line,14,y,w-28); y+=19; }
  ctx.font="12px system-ui";
  for (const line of labels) { ctx.fillText(line,14,y,w-28); y+=17; }
  const pixels=raster(frame,options,scale,Math.max(1,Math.round(box.w*ratio)),Math.max(1,Math.round(box.h*ratio)));
  ctx.drawImage(pixels,box.x,box.y,box.w,box.h);
  const X=lon=>box.x+box.w*(lon+180)/360,Y=lat=>box.y+box.h*(90-lat)/180;
  ctx.lineWidth=.65; ctx.strokeStyle="rgba(34,46,48,.32)"; ctx.beginPath();
  for (let lon=-180;lon<=180;lon+=30) { ctx.moveTo(X(lon),box.y); ctx.lineTo(X(lon),box.y+box.h); }
  for (let lat=-90;lat<=90;lat+=30) { ctx.moveTo(box.x,Y(lat)); ctx.lineTo(box.x+box.w,Y(lat)); }
  ctx.stroke(); flow(ctx,winds,box);
  const cursor=options.cursor;
  if (cursor && Number.isFinite(cursor.lat) && Number.isFinite(cursor.lon)) {
    // The linked selection: meridional section along x, zonal section along y.
    const cx=X((cursor.lon+540)%360-180), cy=Y(cursor.lat);
    ctx.save(); ctx.strokeStyle="#ff8c1a"; ctx.lineWidth=1.2; ctx.setLineDash([5,4]); ctx.beginPath();
    ctx.moveTo(cx,box.y); ctx.lineTo(cx,box.y+box.h); ctx.moveTo(box.x,cy); ctx.lineTo(box.x+box.w,cy); ctx.stroke();
    ctx.setLineDash([]); ctx.lineWidth=2; ctx.strokeStyle="#1b1b1b"; ctx.beginPath(); ctx.arc(cx,cy,5,0,2*Math.PI); ctx.stroke();
    ctx.strokeStyle="#ff8c1a"; ctx.lineWidth=1.4; ctx.beginPath(); ctx.arc(cx,cy,5,0,2*Math.PI); ctx.stroke(); ctx.restore();
  }
  ctx.fillStyle="#263437"; ctx.font="11px system-ui"; ctx.textAlign="right";
  for (const lat of [-60,0,60]) ctx.fillText(lat===0 ? "0°" : Math.abs(lat)+"°"+(lat<0?"S":"N"),box.x-3,Y(lat)+4);
  ctx.textAlign="center";
  for (const lon of [-180,-90,0,90,180]) {
    if (w<280 && Math.abs(lon)===90) continue;
    ctx.fillText(lon===0 || Math.abs(lon)===180 ? Math.abs(lon)+"°" : Math.abs(lon)+"°"+(lon<0?"W":"E"),X(lon),box.y+box.h+14);
  }
  const left=box.x,right=box.x+box.w, legendY=box.y+box.h+43;
  ctx.textAlign="left"; ctx.font="600 12px system-ui";
  ctx.fillText([options.label || options.field,options.units].filter(Boolean).join(" · "),left,legendY-7,box.w);
  if (scale.range) {
    for (let x=0;x<box.w;x++) {
      const [r,g,b]=scale.lut[scale.range[1]>scale.range[0] ? Math.round(255*clamp(x/(box.w-1),0,1)) : 128];
      ctx.fillStyle="rgb("+r+","+g+","+b+")"; ctx.fillRect(left+x,legendY,1.1,9);
    }
    ctx.font="11px system-ui"; ctx.fillStyle="#263437";
    ctx.fillText(numberText(scale.range[0]),left,legendY+23);
    ctx.textAlign="right"; ctx.fillText(numberText(scale.range[1]),right,legendY+23);
  } else { ctx.fillStyle="#263437"; ctx.fillText("No valid samples",left,legendY+10); }
  ctx.textAlign="left"; ctx.fillStyle="#9fa6a7"; ctx.fillRect(left,legendY+30,10,8); ctx.fillStyle="#485556";
  ctx.font="10px system-ui"; ctx.fillText("Masked or outside sampled latitudes",left+14,legendY+38);
  return {box,width:w,height:h};
}
