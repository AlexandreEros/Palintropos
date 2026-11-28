// One saved state on one surface, in the globe's binary format. Arrays borrow
// the received buffer. The same selection without the format option is an
// ordinary JSON result of the selection model.
export async function loadFrame(selection, signal) {
  const q = JSON.stringify({...selection, options: {...(selection.options || {}), format: "frame"}});
  const response = await fetch("select?" + new URLSearchParams({q}), {signal});
  if (!response.ok) throw new Error((await response.json()).error || response.statusText);
  const bytes = await response.arrayBuffer();
  const length = new DataView(bytes).getUint32(0, true);
  const directory = JSON.parse(new TextDecoder().decode(new Uint8Array(bytes, 4, length)));
  const frame = {};
  for (const [name, [offset, size]] of Object.entries(directory)) {
    const Type = name === "mask" || name === "cells" ? Uint8Array : Float32Array;
    frame[name] = new Type(bytes, 4 + length + offset, size);
  }
  // Native cells (columnar archives): draw and read each cell's archived value.
  frame.cells = Boolean(frame.cells?.[0]);
  return frame;
}
