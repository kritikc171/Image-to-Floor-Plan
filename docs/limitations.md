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
- **Normals are derived.** Surface normals are computed from the relative depth map
  under an assumed pinhole camera model. They are estimates, not ground truth.
- **Object labels are probabilistic.** Labels come from deterministic shape/color
  heuristics and, when configured, an LLM reasoning over that evidence. They carry
  an `INFERRED` or `UNCERTAIN` status and are not calibrated probabilities.
- **Generated views (optional) are synthetic.** If the optional novel-view
  provider is enabled, generated images and anything derived from them are marked
  `GENERATED` and are never treated as real photographs of the room.

## Operational limits

- One RGB image per request; size and pixel limits enforced server-side.
- In-memory job state: no restart persistence, no multi-worker deployment.
- Artifacts are local, temporary, and deleted after the configured TTL.
- Groq reasoning is optional; without a key the system uses the deterministic
  fallback and says so in `provenance.reasoning_provider`.
- The optional novel-view provider is disabled by default and capped at 4
  candidate views per request.

## Safe usage

Treat the output as a starting sketch: verify any measurement with a tape measure
before cutting, buying, or building anything.
