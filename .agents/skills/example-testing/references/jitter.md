# End-of-example jitter and settling

The agent sees discrete images, not a continuous live video stream. A single
final screenshot cannot establish smoothness or settling. Use a consecutive
frame sequence plus measured motion where practical; inspect the sequence in
time order rather than selecting only attractive start/end frames.

1. Define the expected terminal behavior from the user's criteria: stationary
   objects, damped motion, or ongoing smooth motion. Do not assume every example
   should stop (a driven system or pendulum may legitimately keep moving).
   Define the terminal simulation time/event and tolerances per example. The
   windowed GL viewer has no automatic end at `--num-frames`.
2. Run through that event and an agreed settling interval. Keep physics running
   during the observation interval, with rendering enabled. Release all picks,
   mouse buttons and movement keys. Fix the camera and choose close-up views of
   contact points, resting bodies, cable/cloth tips, or other likely trouble spots.
   Test another angle in a separate observation interval; never pan during a
   motion measurement. Retain an overview for context.
3. As a starting observation window, capture 5–10 seconds of late-run output,
   then repeat a later window to detect persistent chatter or slow drift. Extend
   this for slow oscillations. These durations are a proposed protocol, not a
   universal settling threshold or evidence that an example passed.
4. For live GUI capture, use a single persistent `mss.MSS(backend="xlib")`
   session in a loop, saving numbered lossless frames with monotonic capture
   timestamps. Aim for 30–60 captures/second only if rendering and storage sustain
   it; report measured intervals, duplicates and gaps. The initial reference desktop rendered
   around 20–24 FPS, so requesting 60 Hz does not produce 60 unique
   simulation images. Avoid spawning a new Python process for every frame.
5. For suspected fast jitter, use a controlled sequence: pause, advance one
   simulation frame, wait for that frame to render, and capture it; repeat over
   the terminal interval. Verify step acceptance via simulation time/frame count
   when available. A loop of keypresses with fixed sleeps alone does not prove
   every step was accepted. If the GUI cannot expose reliable timing, use a
   separate instrumented replay/harness without modifying the installed wheel,
   recording state and one render per simulation frame. Record this change in
   methodology. Substep jitter needs substep state sampling; frame images alone
   cannot exclude it. Do not infer simulation seconds from wall time or render FPS.
6. Inspect consecutive, full-resolution object crops in ordered batches/contact
   sheets, retaining frame IDs and timestamps. Look for alternating positions,
   contact chatter, small repeated rotations, buzzing cloth/cable ends, sudden
   jumps, residual bouncing, or slow creep. A GIF/video can aid human review but
   does not give the agent continuous perception. Do not rely on a downsampled
   montage alone to rule out subpixel or one-frame motion.
7. Quantify suspicious motion when useful: track object landmarks/edges and
   report peak-to-peak displacement, RMS displacement about the local mean,
   frame-to-frame deltas and long-term drift. For smooth continuing motion,
   examine residuals from its expected trajectory and unexpected changes in
   velocity instead of demanding zero displacement. Prefer body position,
   orientation, linear/angular velocity, or particle-state traces from a replay
   to distinguish physics jitter from image artifacts. State tolerances in world
   units when available; any pixel tolerance depends on camera distance and
   resolution. Choose thresholds from user requirements, not after seeing results.
8. Exclude the cursor, FPS text, UI, animated shadows and unrelated objects from
   image difference regions. Capture a paused baseline to measure renderer noise,
   but never call the paused baseline a settling pass. Image differences alone
   are not proof of physical motion. Conversely, unchanged pixels can conceal
   subpixel motion or occluded objects. State traces and multiple fixed viewpoints
   can resolve those ambiguities.
9. Reproduce anomalies from reset or a fresh process, save the complete sequence
   and logs, and report observed amplitude, frequency only when sampling supports
   it, persistence, and affected objects. Distinguish physical jitter, rendering
   flicker, capture stutter and legitimate oscillation. Sampling can alias fast
   motion; qualify a clean result as “no jitter observed within this interval,
   viewpoint, sampling rate and sensitivity,” not an unconditional absence claim.

For each result record: example/solver/device, terminal event, simulation interval
(or explicitly unknown), wall-clock capture interval, actual sampling cadence,
camera/view, inspected objects, metric/tolerance, observations, and evidence paths.
This is a testing procedure, not evidence that any example meets its settling
criteria. There is no bundled sequence-capture or motion-analysis script yet;
record and validate the capture method used for each campaign.
