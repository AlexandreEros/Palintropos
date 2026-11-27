// One scientific selection per response. Arrays borrow the received buffer.
export async function loadFrame(selection, signal) {
  const response = await fetch("select?" + new URLSearchParams({q: JSON.stringify(selection)}), {signal});
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
