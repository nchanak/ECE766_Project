/**
 * Level config
 * - type: "canvas" — built-in demo scene (no image file)
 * - type: "image" — local image; waldo is normalized 0–1 (center); hitRadius is a fraction of image width (hit if distance ≤ width × hitRadius)
 *
 * Add your own scene: put the image under waldo-game/assets/ and add an object with type: "image".
 */
window.WALDO_LEVELS = [
  {
    id: "demo-canvas",
    type: "canvas",
    title: "Demo",
    hitRadius: 0.065,
  },
];
