/**
 * Orton cone → ordinal index, mirroring `pyrochrome.pipeline.cones` so the
 * browser builds the exact same `cone_num` feature the models were trained on.
 */

const CONE_ORDER = [
  "022",
  "021",
  "020",
  "019",
  "018",
  "017",
  "016",
  "015",
  "014",
  "013",
  "012",
  "011",
  "010",
  "09",
  "08",
  "07",
  "06",
  "05",
  "05.5",
  "04",
  "03",
  "02",
  "01",
  "1",
  "2",
  "3",
  "4",
  "5",
  "5.5",
  "6",
  "7",
  "8",
  "9",
  "10",
  "11",
  "12",
  "13",
  "14",
];

const CONE_INDEX = new Map(CONE_ORDER.map((cone, i) => [cone, i]));

/** Ordinal index for a cone label, or NaN if unknown. */
export function coneToOrdinal(value: string): number {
  const normalised = value.replace("&#189;", ".5").replace(/\s/g, "");
  return CONE_INDEX.has(normalised) ? (CONE_INDEX.get(normalised) as number) : NaN;
}
