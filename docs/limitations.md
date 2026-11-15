# Limitations

**Floor Plan Generator produces an approximate, schematic hypothesis — not a
verified architectural drawing.** Read this before relying on any output.

## What the system cannot do

- **No metric dimensions.** A single RGB photo without camera intrinsics or known
  reference objects cannot yield reliable meters. Room dimensions are reported as
  `null`/normalized, never fabricated.
- **No hidden geometry.** Walls behind the camera, occluded objects, and closed
  doors are not reconstructed. Hidden objects are never labeled as detected.
- **No guaranteed completeness.** Detectors are heuristic; missed furniture is
  normal.
- **Depth is relative.** The default depth estimator uses classical image cues
  (horizon/gradient geometry). It is a unitless relative estimate, not metric
  depth, and is used only for coarse placement.
- **Rectangular geometry is an approximation.** When a usable visible-floor
  quadrilateral is available, a homography maps it to straight-edged plan
  coordinates while preserving its estimated aspect ratio. This improves the
  schematic and keeps object projections in the same coordinate system, but
  does not verify that the real room is rectangular or recover unseen corners.
  If homography is rejected but the ground-plane bounds remain usable, the
  boundary and object coordinates share an axis-aligned rectangular
  approximation with a bounded fallback aspect ratio. This can distort their
  relative scale; if the bounds are degenerate, room geometry is unavailable.
- **Normals are derived.** Surface normals are computed from the relative depth map
  under an assumed pinhole camera model. They are estimates, not ground truth.
- **Object labels are limited and uncertain.** The current pipeline uses the
  pretrained COCO detector when available and explicitly marked shape-based
  furniture candidates otherwise; it does not invoke an LLM reasoning step.
  Detector scores are not calibrated probabilities, and estimated positions
  remain uncertain.
- **Detector class coverage is limited.** The bundled COCO detector can identify
  classes such as beds, couches, chairs, dining tables, and TVs. It does not
  recognize wardrobes, cabinets, doors, or windows as those classes; geometric
  edge candidates are not confirmation that an opening exists.
- **Generated views (optional) are synthetic.** If the optional novel-view
  provider is enabled, generated images and anything derived from them are marked
  `GENERATED` and are never treated as real photographs of the room.

## Operational limits

- One RGB image per request; size and pixel limits enforced server-side.
- In-memory job state: no restart persistence, no multi-worker deployment.
- Artifacts are local, temporary, and deleted after the configured TTL.
- Groq configuration is optional, but the current `run_pipeline` does not invoke
  Groq; it reports the deterministic fallback as the reasoning provider.
- The optional novel-view provider is disabled by default and capped at 4
  candidate views per request.

## Safe usage

Treat the output as a starting sketch: verify any measurement with a tape measure
before cutting, buying, or building anything.
