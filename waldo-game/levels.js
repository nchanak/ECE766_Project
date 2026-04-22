/**
 * Level config (client-side only; no server)
 *
 * - type: "image" — scene from assets/
 * - waldo: { x, y } — center of Waldo in normalized image coords: top-left (0,0), bottom-right (1,1)
 *   To measure: read pixel (px, py) in an editor or the browser, then
 *     x = px / image width,   y = py / image height
 * - hitRadius — max click distance to count as a hit, as a fraction of image width (e.g. 0.06 ≈ 6% of width);
 *   tune for figure size.
 *
 */
window.WALDO_LEVELS = [
  {
    id: "level-city",
    type: "image",
    title: "City",
    src: "assets/city.png",
    waldo: { x: 0.1, y: 0.74 },
    hitRadius: 0.03,
  },
  {
    id: "level-neighborhood",
    type: "image",
    title: "Neighborhood",
    src: "assets/neighborhood.png",
    waldo: { x: 0.25, y: 0.5 },
    hitRadius: 0.06,
  },
  {
    id: "level-street",
    type: "image",
    title: "Street",
    src: "assets/street.png",
    waldo: { x: 0.345, y: 0.27 },
    hitRadius: 0.03,
  },
  {
    id: "demo-canvas",
    type: "canvas",
    title: "Demo (random crow)",
    hitRadius: 0.065,
  },
];
