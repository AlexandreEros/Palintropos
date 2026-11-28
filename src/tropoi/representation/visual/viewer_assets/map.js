// Number formatting shared by the Explorer and the globe readout. The flat
// longitude × latitude map that once lived here is the Explorer's field plot.

export function numberText(value) {
  return Number.isFinite(value) ? String(Number(value.toPrecision(6))) : "unavailable";
}
