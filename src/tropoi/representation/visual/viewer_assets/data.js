// One scientific selection per response. Arrays borrow the received buffer.
export async function loadFrame(params, signal) {
  const response = await fetch("frame?" + new URLSearchParams(params), {signal});
  if (!response.ok) throw new Error((await response.json()).error || response.statusText);
  const bytes = await response.arrayBuffer();
  const length = new DataView(bytes).getUint32(0, true);
  const directory = JSON.parse(new TextDecoder().decode(new Uint8Array(bytes, 4, length)));
  const frame = {};
  for (const [name, [offset, size]] of Object.entries(directory)) {
    const Type = name === "mask" ? Uint8Array : Float32Array;
    frame[name] = new Type(bytes, 4 + length + offset, size);
  }
  return frame;
}
