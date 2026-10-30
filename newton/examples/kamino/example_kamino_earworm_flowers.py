# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Animate a garden of twenty-stage Stewart platform flowers using Kamino DVI.

Each plate is the moving platform of one stage and the base of the next.
Universal joints attach both ends of each hydraulic actuator. A passive axial
swivel in each cylinder prevents the universal joints from binding in torsion.
Only piston extensions are driven; all plate poses emerge from the dynamics.
The towers sway, bounce, twist, and bow while a lightweight 1 m Hoberman flower
pumps open and closed on a pinned yoke at each tip. Every flower moves with
small variations in timing, size, and direction.
Seven flowers form a compact hexagonal patch with gently scattered base positions.
Use --flower-resolution to vary scissor cells per 36-degree ring arc; the default
of two reproduces the original twenty-cell rings procedurally.
Use --mute to disable audio.

The custom garden layout is assigned once after model setup. Calling the
viewer's set_visible_worlds() or set_world_offsets() afterward replaces it;
reapply viewer.world_offsets.assign(example.world_offsets) to restore it.

Success criteria:
- Seven articulated flowers form a compact garden, with their bases remaining anchored.
- The stacked platforms sway, bounce, twist, and bow with small timing differences between flowers.
- The spherical scissor mechanisms at the tips open and close while remaining attached and connected.
- Motion stays bounded without joints separating or the flower mechanisms collapsing.

Runs:
- `python -m newton.examples kamino_earworm_flowers`
- `python -m newton.examples kamino_earworm_flowers --mute`
  - No background audio plays.
"""

import argparse
import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.kamino.utils.hoberman import generate_sphere
from newton.solvers import SolverKamino

# Saturated colors distinguish the interlocking flower rings.
FLOWER_COLORS = (
    (0.86, 0.06, 0.02),
    (0.92, 0.22, 0.01),
    (0.94, 0.42, 0.01),
    (0.88, 0.62, 0.01),
    (0.78, 0.72, 0.01),
    (0.46, 0.72, 0.01),
    (0.03, 0.60, 0.12),
    (0.00, 0.58, 0.38),
    (0.00, 0.62, 0.68),
    (0.00, 0.53, 0.88),
    (0.06, 0.30, 0.88),
    (0.22, 0.12, 0.78),
    (0.42, 0.10, 0.76),
    (0.58, 0.08, 0.68),
)


def _transform(values, scale=1.0):
    return wp.transform(wp.vec3(*values[:3]) * scale, wp.quat(*values[3:]))


def _garden_offsets(world_count):
    """Place flowers on a 2 m hexagonal lattice with up to 16 cm of fixed jitter."""
    radius = 0
    while 1 + 3 * radius * (radius + 1) < world_count:
        radius += 1
    cells = [(q, r) for q in range(-radius, radius + 1) for r in range(-radius, radius + 1) if abs(q + r) <= radius]
    # Fill from the center outward to avoid a rectangular boundary.
    cells.sort(key=lambda cell: (cell[0] ** 2 + cell[0] * cell[1] + cell[1] ** 2, math.atan2(cell[1], cell[0])))
    offsets = np.zeros((world_count, 3), dtype=np.float32)
    offsets[:, :2] = [(2.0 * (q + 0.5 * r), math.sqrt(3.0) * r) for q, r in cells[:world_count]]
    rng = np.random.default_rng(2026)
    angles = rng.uniform(0.0, 2.0 * math.pi, world_count)
    lengths = 0.16 * np.sqrt(rng.uniform(0.0, 1.0, world_count))
    offsets[:, 0] += lengths * np.cos(angles)
    offsets[:, 1] += lengths * np.sin(angles)
    offsets -= offsets.mean(axis=0)
    return offsets


def add_hoberman_flower(builder, plate, joints_tree, sphere):
    """Mount a scaled sphere by its outer three-hinge crossing on a driven yoke.

    A bearing supports one strut at a shared outer crossing. It turns by half
    the opening hinge angle, keeping the supported arms symmetric about the
    stem while the ring joints transmit the opening to the sphere.
    """
    scale = 1.0 / sphere.diameter_rest
    mass_scale = 0.18 / sum(body[4] for body in sphere.bodies)
    rotation = wp.quat_between_vectors(wp.vec3(-1.0, 0.0, 0.0), wp.vec3(0.0, 0.0, -1.0))
    pole = wp.vec3(*sphere.pole) * scale
    stem_tip = wp.vec3(0.0, 0.0, 0.055)
    placement = wp.transform_multiply(
        builder.body_q[plate], wp.transform(stem_tip - wp.quat_rotate(rotation, pole), rotation)
    )
    offset = builder.body_count
    for label, pose, com, inertia, mass in sphere.bodies:
        builder.add_link(
            xform=wp.transform_multiply(placement, _transform(pose, scale)),
            com=wp.vec3(*com) * scale,
            inertia=wp.mat33(np.asarray(inertia) * mass_scale * scale**2),
            mass=mass * mass_scale,
            lock_inertia=True,
            label=label,
        )

    opening_joint = None
    mount_child = offset + sphere.mount_body
    mount_frame = _transform(sphere.mount_frame, scale)
    closures = []
    for index, (kind, label, parent, child, xp, xc, axes, coordinates) in enumerate(sphere.joints):
        if kind == "FREE":
            joint = builder.add_joint_free(offset + child, label="flower_root")
            start = builder.joint_q_start[joint]
            builder.joint_q[start : start + 7] = list(builder.body_q[offset + child])
        else:
            driven = label == sphere.opening_label
            kwargs = {
                "joint_type": newton.JointType[kind],
                "parent": offset + parent,
                "child": offset + child,
                "parent_xform": _transform(xp, scale),
                "child_xform": _transform(xc, scale),
                "angular_axes": [
                    newton.ModelBuilder.JointDofConfig(
                        axis=wp.vec3(*axis),
                        target_ke=300.0 if driven else 0.0,
                        target_kd=3.0 if driven else 0.0,
                        actuator_mode=newton.JointTargetMode.POSITION_VELOCITY
                        if driven
                        else newton.JointTargetMode.NONE,
                    )
                    for axis in axes
                ],
                "label": label,
            }
            if index >= sphere.tree_joint_count:
                closures.append(kwargs)
                continue
            joint = builder.add_joint(**kwargs)
            start = builder.joint_q_start[joint]
            builder.joint_q[start : start + len(coordinates)] = coordinates
            if driven:
                opening_joint = joint
        joints_tree.append(joint)

    # Both joint frames coincide at the crossing. Its pin axis is tangential,
    # so the supported strut can rotate as the sphere opens around the pole.
    child = mount_child
    child_frame = mount_frame
    parent_frame = wp.transform_multiply(
        wp.transform_inverse(builder.body_q[plate]),
        wp.transform_multiply(builder.body_q[child], child_frame),
    )
    mount = builder.add_joint_revolute(
        plate,
        child,
        parent_xform=parent_frame,
        child_xform=child_frame,
        axis=newton.Axis.X,
        target_ke=600.0,
        target_kd=6.0,
        actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
        label="flower_mount",
    )
    for kwargs in closures:
        builder.add_joint(**kwargs)

    visual = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False)
    for body, pose, extents in sphere.tiles:
        color = FLOWER_COLORS[body % len(FLOWER_COLORS)]
        builder.add_shape_box(
            offset + body,
            xform=_transform(pose, scale),
            hx=extents[0] * scale,
            hy=extents[1] * scale,
            hz=extents[2] * scale,
            cfg=visual,
            color=color,
        )

    gold = (0.92, 0.65, 0.25)
    builder.add_shape_cylinder(
        plate,
        xform=wp.transform((0.0, 0.0, 0.025), wp.quat_identity()),
        radius=0.009,
        half_height=0.022,
        cfg=visual,
        color=gold,
    )
    pin_axis = wp.transform_vector(parent_frame, wp.vec3(1.0, 0.0, 0.0))
    pin_rotation = wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), pin_axis)
    for sign in (-1.0, 1.0):
        bearing = stem_tip + sign * 0.012 * pin_axis
        builder.add_shape_box(
            plate,
            xform=wp.transform(bearing - wp.vec3(0.0, 0.0, 0.006), pin_rotation),
            hx=0.007,
            hy=0.013,
            hz=0.003,
            cfg=visual,
            color=(0.20, 0.25, 0.30),
        )
    builder.add_shape_cylinder(
        plate,
        xform=wp.transform(stem_tip, pin_rotation),
        radius=0.003,
        half_height=0.018,
        cfg=visual,
        color=gold,
    )
    return opening_joint, mount, offset


@wp.func
def flower_opening(time: float) -> float:
    # Start at the authored pose and ease into a slow, six-second breath.
    phase = 1.04719755 * time
    return (1.0 - wp.exp(-0.35 * time * time)) * (0.2 * (1.0 - wp.cos(phase)) + 0.9 * wp.sin(phase))


# The song: an original chiptune, synthesized on the GPU at startup.
SONG_RATE = 44100
SONG_ROW = 4725  # samples per 16th note at 140 BPM
SONG_BEAT = 4.0 * SONG_ROW / SONG_RATE  # [s]
SONG_BAR = 4.0 * SONG_BEAT  # [s]
SONG_LOOP_BAR = 8  # the loop skips the intro
LEAD, HARMONY, STABS, BASS, DRUMS, HATS = range(6)
KICK, CLAP = range(2)
_CHORDS = {"Am": (45, 69, 72, 76), "F": (41, 69, 72, 77), "C": (48, 67, 72, 76), "G": (43, 67, 71, 74)}
# One "<note>:<sixteenth notes>" token per note, one string per bar.
_RIFF = [
    "E5:2 A5:1 E5:2 C6:2 B5:2 A5:2 E5:1 G5:2 A5:2",
    "F5:2 A5:1 F5:2 C6:2 B5:2 A5:2 F5:1 G5:2 A5:2",
    "G5:2 C6:1 G5:2 E6:2 D6:2 C6:2 G5:1 A5:2 B5:2",
    "D6:2 B5:1 G5:2 D6:2 E6:2 D6:2 B5:1 G5:2 A5:2",
]
_HOOK = [*_RIFF, *_RIFF[:2], "G5:2 C6:1 G5:2 E6:2 G6:2 E6:2 D6:1 C6:2 D6:2", "E6:2 D6:1 B5:2 G5:2 A5:2 B5:2 D6:1 E6:4"]
_FINAL = [*_HOOK[:7], "G6:4 E6:2 D6:2 B5:2 D6:1 E6:1 A6:4"]
_BREAK = [
    "A5:2 C6:2 F6:4 E6:2 C6:2 A5:4",
    "B5:2 D6:2 G6:4 F6:2 D6:2 B5:4",
    "C6:2 E6:2 G6:4 A6:2 G6:2 E6:4",
    "A6:4 G6:2 E6:2 C6:4 E6:4",
    "A5:2 C6:2 F6:4 E6:2 C6:2 A5:4",
    "B5:2 D6:2 G6:4 F6:2 D6:2 B5:4",
    "E6:2 G6:2 C7:4 B6:2 G6:2 E6:4",
    "D6:2 B5:2 G5:2 B5:2 D6:2 G6:2 B6:4",
]
# Sections: chords, melody, style flags, and dance moves. Flags: i/d/D/k drum
# style (intro, drive, drop, break), s chord stabs, h harmony, r rolling bass.
# Moves: sway, bounce, twist, head-bang, bow, bloom.
_SONG = (
    ("Am F C G Am F C G", None, "is", (0.05, 0.0, 0.0, 0.0, 0.0, 0.0)),
    ("Am F C G Am F C G", _HOOK, "d", (0.06, 1.0, 0.0, 0.0, 0.0, 0.4)),
    ("Am F C G Am F C G", _HOOK, "ds", (0.05, 1.0, 0.07, 0.0, 0.0, 0.4)),
    ("F G C Am F G C G", _BREAK, "ks", (0.0, 0.0, 0.0, 0.0, 1.0, 0.0)),
    ("Am F C G Am F C G", _HOOK, "Dshr", (0.03, 1.0, 0.06, 0.05, 0.0, 0.7)),
    ("Am F C G Am F C G", _FINAL, "Dshr", (0.03, 1.0, 0.06, 0.05, 0.0, 0.7)),
)
SONG_BARS = sum(len(section[0].split()) for section in _SONG)


def _song_score():
    """Expand the sections into note events and per-bar dance moves."""

    def note(name):
        return 12 * int(name[-1]) + 12 + {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[name[0]]

    def third_below(midi):  # diatonic in A minor
        return [p for p in range(midi - 5, midi) if p % 12 in (0, 2, 4, 5, 7, 9, 11)][-2]

    events, moves = [], []  # event: (voice, start row, rows, note x3, velocity)
    for progression, melody, flags, move in _SONG:
        chords = progression.split()
        for bar, chord in enumerate(chords):
            row, last = 16 * len(moves), bar == len(chords) - 1
            ramp = bar / (len(chords) - 1) if "i" in flags else 1.0
            moves.append([m * ramp for m in move])
            root, *triad = _CHORDS[chord]
            r = row
            for token in (melody[bar] if melody else "").split():
                name, rows = token.split(":")
                midi, rows = note(name), int(rows)
                played = 1 if rows < 4 and "k" not in flags else rows  # staccato hook
                velocity = 1.0 if (r - row) % 4 == 0 else 0.85
                events.append((LEAD, r, played, midi, midi, midi, velocity))
                if "h" in flags:
                    events.append((HARMONY, r, played, *[third_below(midi)] * 3, 0.7 * velocity))
                r += rows
            for r, gate in enumerate((1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1)):
                if gate and "s" in flags:
                    velocity = (1.0 if r % 4 == 2 else 0.7) * min(1.0, 0.5 + 0.1 * bar)
                    events.append((STABS, row + r, 1, *triad, velocity))
            if not ("i" in flags and bar < 2):  # octave-bouncing bass
                step = 1 if "r" in flags else 2
                for r in range(0, 16, step):
                    interval = 12 if r % 4 == 2 else (7 if r >= 12 else 0)
                    events.append((BASS, row + r, step, *[root + interval] * 3, 1.0 - 0.3 * (r % 2)))
            hits = {}
            if flags[0] in "dD" or ("i" in flags and bar >= 4):
                hits = dict.fromkeys((0, 4, 8, 12), (KICK, 1.0))
            if flags[0] in "dD" or ("i" in flags and bar >= 6):
                hits.update({4: (CLAP, 0.8), 12: (CLAP, 0.8)})
                if last:
                    hits.update({r: (CLAP, 0.35 + 0.15 * (r - 12)) for r in (13, 14, 15)})
            if "k" in flags and bar == len(chords) - 2:
                hits = {r: (CLAP, 0.25 + 0.01 * r) for r in (0, 4, 8, 12)}
            if "k" in flags and last:  # clap roll into the drop
                hits = {r: (CLAP, 0.3 + 0.03 * r) for r in (0, 2, 4, 6, *range(8, 16))}
            for r, (kind, velocity) in sorted(hits.items()):
                events.append((DRUMS, row + r, 4, kind, kind, kind, velocity))
            if flags[0] in "dDk" or ("i" in flags and bar >= 6):
                for r in range(1, 16) if flags[0] == "D" else range(2, 16, 4):
                    open_hat = float(flags[0] in "dD" and r % 4 == 2)
                    events.append((HATS, row + r, 2, open_hat, open_hat, open_hat, 0.7 if r % 4 == 2 else 0.3))
    return events, moves


@wp.func
def _pulse(phase: float, dt: float, duty: float):
    # Band-limited (PolyBLEP) pulse wave without DC offset
    p = phase - wp.floor(phase)
    q = p - duty + 1.0
    q -= wp.floor(q)
    v = wp.where(p < duty, 1.0, -1.0) - (2.0 * duty - 1.0)
    for edge in range(2):
        x = wp.where(edge == 0, p, q)
        b = float(0.0)
        if x < dt:
            b = 2.0 * x / dt - (x / dt) * (x / dt) - 1.0
        elif x > 1.0 - dt:
            b = ((x - 1.0) / dt) * ((x - 1.0) / dt) + 2.0 * (x - 1.0) / dt + 1.0
        v += wp.where(edge == 0, b, -b)
    return v


@wp.func
def _noise(index: int, seed: int):
    return 2.0 * wp.randf(wp.rand_init(seed, index)) - 1.0


@wp.kernel
def song_voices_kernel(
    row_event: wp.array2d[int],
    start: wp.array[int],
    length: wp.array[int],
    note: wp.array[wp.vec3],
    velocity: wp.array[float],
    out: wp.array2d[float],
):
    voice, i = wp.tid()
    e = row_event[voice, i / SONG_ROW]
    v = float(0.0)
    if e >= 0:
        t = float(i - start[e]) / float(SONG_RATE)
        n = float(length[e]) / float(SONG_RATE)
        p = note[e]
        f = 440.0 * wp.pow(2.0, (p[int(t * 50.0) % 3] - 69.0) / 12.0)  # stabs arpeggiate at 50 Hz
        dt = f / float(SONG_RATE)
        phase = f * t
        tri = 4.0 * wp.abs(phase - wp.floor(phase) - 0.5) - 1.0
        if voice == BASS:
            v = (0.9 * tri + 0.15 * _pulse(phase, dt, 0.5)) * (0.5 + 0.5 * wp.exp(-t / 0.1))
            n *= 0.8
        elif voice == DRUMS:
            if int(p[0]) == KICK:
                sweep = 48.0 * t + 110.0 * (1.0 - wp.exp(-30.0 * t)) / 30.0
                v = wp.sin(2.0 * wp.pi * sweep) * wp.exp(-10.0 * t) + 0.25 * _noise(i, 11) * wp.exp(-300.0 * t)
            else:
                burst = wp.where(t < 0.025, wp.exp(-400.0 * wp.mod(t, 0.008)), wp.exp(-18.0 * (t - 0.025)))
                v = 0.8 * burst * _noise(int(t * 18000.0 * wp.pow(0.9, wp.floor(t * 50.0))), 13)
        elif voice == HATS:
            v = _noise(int(t * 30000.0), 17) * wp.exp(-t * wp.where(p[0] > 0.5, 14.0, 60.0))
        elif voice == STABS:
            v = _pulse(phase, dt, 0.25) * (0.5 + 0.5 * wp.exp(-t / 0.1))
        else:
            # Lead and harmony: near-square pulse with a slow width sweep, softened
            # by a triangle and turned down for high notes that the ear finds harsh
            duty = 0.44 + 0.06 * wp.sin(0.6 * wp.pi * float(i) / float(SONG_RATE) + 2.1 * float(voice))
            tone = 0.7 * _pulse(phase, dt, duty) + 0.5 * tri
            v = tone * wp.sqrt(wp.min(1.0, 660.0 / f)) * (0.6 + 0.4 * wp.exp(-t / 0.12))
        v *= wp.min(t / 0.002, 1.0) * wp.clamp((n - t) / 0.012, 0.0, 1.0) * velocity[e]
    out[voice, i] = v


@wp.kernel
def song_mix_kernel(voices: wp.array2d[float], gains: wp.array2d[float], loop_start: int, pcm: wp.array[wp.int16]):
    i = wp.tid()
    left = float(0.0)
    right = float(0.0)
    for c in range(6):
        left += gains[0, c] * voices[c, i]
        right += gains[1, c] * voices[c, i]
    j = i - 3 * SONG_ROW  # dotted-eighth echo of the lead, wrapping at the loop point
    if j < loop_start and i >= loop_start:
        j += voices.shape[1] - loop_start
    if j >= 0:
        right += 0.05 * voices[LEAD, j]
    pcm[2 * i] = wp.int16(32000.0 * wp.tanh(left))
    pcm[2 * i + 1] = wp.int16(32000.0 * wp.tanh(right))


def render_song():
    """Synthesize the song with Warp; return interleaved 16-bit stereo PCM and per-bar dance moves."""
    events, moves = _song_score()
    rows = 16 * SONG_BARS
    table = np.full((6, rows), -1, dtype=np.int32)
    for index, event in enumerate(events):  # a later note cuts an earlier one on the same voice
        table[event[0], event[1] : min(event[1] + event[2], rows)] = index
    ev = np.array(events, dtype=np.float32)
    voices = wp.empty((6, rows * SONG_ROW), dtype=float)
    wp.launch(
        song_voices_kernel,
        dim=voices.shape,
        inputs=[
            wp.array(table, dtype=int),
            wp.array(ev[:, 1].astype(np.int32) * SONG_ROW, dtype=int),
            wp.array(ev[:, 2].astype(np.int32) * SONG_ROW, dtype=int),
            wp.array(ev[:, 3:6], dtype=wp.vec3),
            wp.array(ev[:, 6], dtype=float),
        ],
        outputs=[voices],
    )
    gain = 1.4 * np.array((0.1, 0.055, 0.05, 0.17, 0.22, 0.03))  # lead, harmony, stabs, bass, drums, hats
    pan = np.array((-0.1, 0.3, -0.4, 0.0, 0.0, 0.35))
    gains = wp.array(np.stack((gain * np.sqrt(0.5 - 0.5 * pan), gain * np.sqrt(0.5 + 0.5 * pan))), dtype=float)
    pcm = wp.empty(2 * voices.shape[1], dtype=wp.int16)
    wp.launch(
        song_mix_kernel, dim=voices.shape[1], inputs=[voices, gains, 16 * SONG_LOOP_BAR * SONG_ROW], outputs=[pcm]
    )
    return pcm.numpy().tobytes(), moves


def play_song(data: bytes):
    """Loop the song with pyglet, returning the player, or None if no audio device is available."""
    loop_byte = 4 * 16 * SONG_LOOP_BAR * SONG_ROW
    try:
        import pyglet  # noqa: PLC0415
        from pyglet.media.codecs.base import AudioData, AudioFormat, Source  # noqa: PLC0415

        class LoopingSource(Source):
            def __init__(self):
                self.audio_format = AudioFormat(channels=2, sample_size=16, sample_rate=SONG_RATE)
                self.offset = 0

            def seek(self, timestamp):
                offset = 4 * int(max(timestamp, 0.0) * SONG_RATE)
                if offset >= len(data):
                    offset = loop_byte + (offset - loop_byte) % (len(data) - loop_byte)
                self.offset = offset

            def get_audio_data(self, num_bytes, compensation_time=0.0):
                chunk = bytearray()
                while len(chunk) < num_bytes:
                    take = min(num_bytes - len(chunk), len(data) - self.offset) & ~3
                    chunk += data[self.offset : self.offset + take]
                    self.offset = self.offset + take if self.offset + take < len(data) else loop_byte
                return AudioData(bytes(chunk), len(chunk))

        player = pyglet.media.Player()
        player.volume = 0.6
        player.queue(LoopingSource())
        player.play()
        return player
    except Exception as error:
        print(f"Audio playback unavailable ({error}); dancing without sound.")
        return None


@wp.func
def song_position(time: float) -> float:
    # Map the dance clock onto the song, which loops back to the end of the intro
    loop = float(SONG_LOOP_BAR) * SONG_BAR
    end = float(SONG_BARS) * SONG_BAR
    if time >= end:
        time = loop + wp.mod(time - loop, end - loop)
    return time


@wp.func
def dance_move(moves: wp.array2d[float], time: float, move: int) -> float:
    # Each bar's move amplitude, cross-faded over the first beat of the bar
    position = song_position(time)
    bar = int(position / SONG_BAR)
    previous = int(song_position(wp.max(time - SONG_BEAT, 0.0)) / SONG_BAR)
    blend = wp.smoothstep(0.0, SONG_BEAT, position - float(bar) * SONG_BAR)
    return wp.lerp(moves[previous, move], moves[bar, move], blend)


@wp.func
def dance_pose(stage: int, stage_count: int, time: float, variation: wp.vec3, moves: wp.array2d[float]) -> wp.transform:
    # Bar-long sways, beat bounces, two-beat twists, beat-rate head-bang waves
    # running up the tower, and slow two-bar bows in the breakdown. Each flower
    # dances a little differently: its own timing (up to 45 ms early or late),
    # move size (+-12%), wave phase, and sway direction.
    time += 0.15 * variation[0]
    gain = 1.0 + 3.0 * (variation[1] - 1.0)
    wave = 0.6 * wp.sin(1.7 * variation[2])
    turn = 0.4 * wp.cos(variation[2])
    position = song_position(time)
    bar = 2.0 * wp.pi * position / SONG_BAR
    beat = 2.0 * wp.pi * position / SONG_BEAT
    s = float(stage)
    bend = wp.sin(wp.pi * s / float(wp.max(stage_count - 1, 1)))
    sway = dance_move(moves, time, 0)
    bang = dance_move(moves, time, 3)
    bow = 0.06 * bend * dance_move(moves, time, 4)
    x = sway * wp.sin(bar - 0.35 * s + wave) + bang * wp.sin(beat - 0.5 * s + wave) + bow * wp.sin(0.5 * bar)
    y = 0.7 * sway * wp.cos(bar - 0.35 * s + wave) + 0.5 * bang * wp.cos(0.5 * beat - 0.5 * s) + bow * wp.cos(0.5 * bar)
    roll = gain * (wp.cos(turn) * x - wp.sin(turn) * y)
    pitch = gain * (wp.sin(turn) * x + wp.cos(turn) * y)
    twist = gain * dance_move(moves, time, 2) * wp.sin(0.5 * beat - 0.3 * s + wave)
    height = 0.1 - 0.004 * gain * dance_move(moves, time, 1) * wp.cos(beat)
    return wp.transform(wp.vec3(0.0, 0.0, height), wp.quat_rpy(roll, pitch, twist))


@wp.func
def dance_opening(time: float, moves: wp.array2d[float]) -> float:
    # The flower pumps open every other beat; in the breakdown it breathes slowly
    bloom = dance_move(moves, time, 5)
    return (1.0 - bloom) * flower_opening(time) + bloom * 0.8 * wp.cos(wp.pi * song_position(time) / SONG_BEAT)


@wp.kernel
def update_flower(
    time: wp.array[float],
    dt: float,
    opening_slope: float,
    reference_slope: float,
    adapt_opening: bool,
    variations: wp.array[wp.vec3],
    moves: wp.array2d[float],
    coordinates: wp.array[int],
    dofs: wp.array[int],
    target_q: wp.array[float],
    target_qd: wp.array[float],
):
    actuator = wp.tid()
    motor = actuator % 2
    variation = variations[actuator / 2]
    ratio = 1.0
    if motor == 1:
        ratio = 0.5
    angle = variation[1] * dance_opening(time[0] + 0.15 * variation[0], moves)
    angle_next = variation[1] * dance_opening(time[0] + 0.15 * variation[0] + dt, moves)
    # Finer rings reach full compaction at smaller hinge angles. Invert their
    # aperture law to retain the original radial breathing motion safely.
    if adapt_opening:
        gain = wp.cos(0.5 * angle) + reference_slope * wp.sin(0.5 * angle)
        gain_next = wp.cos(0.5 * angle_next) + reference_slope * wp.sin(0.5 * angle_next)
        magnitude = wp.sqrt(1.0 + opening_slope * opening_slope)
        rest = wp.atan2(1.0, opening_slope)
        angle = 2.0 * (wp.asin(wp.clamp(gain / magnitude, -1.0, 1.0)) - rest)
        angle_next = 2.0 * (wp.asin(wp.clamp(gain_next / magnitude, -1.0, 1.0)) - rest)
    target_q[coordinates[actuator]] = ratio * angle
    target_qd[dofs[actuator]] = ratio * (angle_next - angle) / dt


@wp.kernel
def update_actuators(
    time: wp.array[float],
    dt: float,
    stage_count: int,
    variations: wp.array[wp.vec3],
    moves: wp.array2d[float],
    anchors_base: wp.array[wp.vec3],
    anchors_top: wp.array[wp.vec3],
    lengths_rest: wp.array[float],
    actuator_dofs: wp.array[int],
    actuator_coordinates: wp.array[int],
    target_q: wp.array[float],
    target_qd: wp.array[float],
):
    actuator = wp.tid()
    stage = (actuator / 6) % stage_count
    variation = variations[actuator / (6 * stage_count)]
    pose = dance_pose(stage, stage_count, time[0], variation, moves)
    pose_next = dance_pose(stage, stage_count, time[0] + dt, variation, moves)
    length = wp.length(wp.transform_point(pose, anchors_top[actuator]) - anchors_base[actuator])
    length_next = wp.length(wp.transform_point(pose_next, anchors_top[actuator]) - anchors_base[actuator])
    dof = actuator_dofs[actuator]
    target_q[actuator_coordinates[actuator]] = length - lengths_rest[actuator]
    target_qd[dof] = (length_next - length) / dt


@wp.kernel
def advance_time(time: wp.array[float], dt: float):
    time[0] += dt


class Example:
    def __init__(self, viewer: newton.viewer.ViewerBase, args=None):
        self.viewer = viewer
        self.stage_count = args.stage_count if args else 20
        self.world_count = getattr(args, "world_count", 7) if args else 7
        if self.stage_count < 1:
            raise ValueError("--stage-count must be positive")
        if self.world_count < 1:
            raise ValueError("--world-count must be positive")
        self.frame_dt = 1.0 / 60.0
        # One implicit step per frame retains loop closure and actuator tracking;
        # each step's cost is dominated by factorizing the dense joint graph.
        self.sim_substeps = getattr(args, "substeps", 1) if args else 1
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        # Follow playback while running so the choreography stays on the beat
        # regardless of the viewer's frame rate.
        self.time = wp.zeros(1, dtype=float)
        self.player = None

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        SolverKamino.register_custom_attributes(builder)
        # The mechanism is attached to ground. Contact between the small, closely
        # spaced fittings is excluded, as in the Kamino test-mechanism example.
        builder.default_shape_cfg.has_shape_collision = False
        builder.default_shape_cfg.density = 1600.0
        builder.add_shape_cylinder(
            -1,
            xform=wp.transform((0.0, 0.0, 0.005), wp.quat_identity()),
            radius=0.075,
            half_height=0.005,
            color=(0.22, 0.26, 0.30),
        )

        def passive(axis):
            return newton.ModelBuilder.JointDofConfig(axis=axis, actuator_mode=newton.JointTargetMode.NONE)

        joints_tree = []
        closures = []
        self.plates = []
        anchors_base, anchors_top, lengths_rest, actuator_joints = [], [], [], []

        def add_plate(level):
            body = builder.add_link(
                xform=wp.transform((0.0, 0.0, 0.012 + 0.1 * level), wp.quat_identity()),
                label=f"plate_{level}",
            )
            color = FLOWER_COLORS[level % len(FLOWER_COLORS)]
            builder.add_shape_cylinder(body, radius=0.05, half_height=0.002, color=color)
            builder.add_shape_cylinder(body, radius=0.047, half_height=0.0024, color=(0.12, 0.16, 0.20))
            self.plates.append(body)
            return body

        base = add_plate(0)
        joints_tree.append(builder.add_joint_fixed(-1, base, parent_xform=builder.body_q[base], label="ground_anchor"))
        for stage in range(self.stage_count):
            top = add_plate(stage + 1)
            for leg in range(6):
                sector = 2.0 * math.pi * (leg // 2) / 3.0
                sign = -1.0 if leg % 2 == 0 else 1.0
                angle_base = sector + sign * math.radians(12.0)
                angle_top = sector + sign * math.radians(48.0)
                a = wp.vec3(0.0425 * math.cos(angle_base), 0.0425 * math.sin(angle_base), 0.004)
                b = wp.vec3(0.0425 * math.cos(angle_top), 0.0425 * math.sin(angle_top), -0.004)
                delta = b + wp.vec3(0.0, 0.0, 0.1) - a
                length = wp.length(delta)
                rotation = wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), delta / length)
                origin_base = wp.transform_point(builder.body_q[base], a)
                origin_top = wp.transform_point(builder.body_q[top], b)
                barrel = builder.add_link(xform=wp.transform(origin_base, rotation), label=f"barrel_{stage}_{leg}")
                rod = builder.add_link(xform=wp.transform(origin_top, rotation), label=f"rod_{stage}_{leg}")
                builder.add_shape_cylinder(
                    barrel,
                    xform=wp.transform((0.0, 0.0, 0.039), wp.quat_identity()),
                    radius=0.0032,
                    half_height=0.035,
                    color=(0.16, 0.23, 0.29),
                )
                for z in (0.008, 0.070):
                    builder.add_shape_cylinder(
                        barrel,
                        xform=wp.transform((0.0, 0.0, z), wp.quat_identity()),
                        radius=0.0038,
                        half_height=0.002,
                        color=(0.9, 0.52, 0.16),
                    )
                builder.add_shape_cylinder(
                    rod,
                    xform=wp.transform((0.0, 0.0, -0.04), wp.quat_identity()),
                    radius=0.0017,
                    half_height=0.036,
                    color=(0.75, 0.82, 0.88),
                )
                # Cross pins make the two orthogonal universal-joint axes visible.
                for body in (barrel, rod):
                    for axis in (wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0)):
                        builder.add_shape_cylinder(
                            body,
                            xform=wp.transform(wp.vec3(), wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), axis)),
                            radius=0.0018,
                            half_height=0.0042,
                            color=(0.9, 0.65, 0.24),
                        )
                lower = builder.add_joint_d6(
                    base,
                    barrel,
                    parent_xform=wp.transform(a, rotation),
                    angular_axes=[passive(newton.Axis.X), passive(newton.Axis.Y)],
                    label=f"universal_lower_{stage}_{leg}",
                )
                # A cylindrical joint combines driven sliding with passive rod spin.
                # Six U-P-U legs without that spin would overconstrain a Stewart platform.
                actuator = builder.add_joint_d6(
                    barrel,
                    rod,
                    parent_xform=wp.transform((0.0, 0.0, length), wp.quat_identity()),
                    linear_axes=[
                        newton.ModelBuilder.JointDofConfig(
                            axis=newton.Axis.Z,
                            limit_lower=-0.025,
                            limit_upper=0.025,
                            target_ke=300000.0,
                            target_kd=150.0,
                            actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
                        )
                    ],
                    angular_axes=[passive(newton.Axis.Z)],
                    label=f"piston_{stage}_{leg}",
                )
                joints_tree.extend((lower, actuator))
                upper_args = {
                    "parent": rod,
                    "child": top,
                    "child_xform": wp.transform(b, rotation),
                    "angular_axes": [passive(newton.Axis.X), passive(newton.Axis.Y)],
                    "label": f"universal_upper_{stage}_{leg}",
                }
                if leg == 0:
                    joints_tree.append(builder.add_joint_d6(**upper_args))
                else:
                    closures.append(upper_args)
                anchors_base.append(a)
                anchors_top.append(b)
                lengths_rest.append(length)
                actuator_joints.append(actuator)
            base = top

        # The remaining upper universal joints close the six parallel leg loops.
        builder.add_articulation(joints_tree, label="stewart_tower")
        for upper_args in closures:
            builder.add_joint_d6(**upper_args)
        flower_tree = []
        self.sphere = generate_sphere(getattr(args, "flower_resolution", 2) if args else 2)
        self.adapt_flower_opening = self.sphere.opening_slope > 1.001 * self.sphere.reference_opening_slope
        opening, mount, flower_body_start = add_hoberman_flower(builder, top, flower_tree, self.sphere)
        builder.add_articulation(flower_tree, label="hoberman_flower")

        # Keep simulation coordinates near the origin; only the viewer spaces
        # the worlds apart. Each world gets its own constraints and controls.
        prototype = builder
        body_offsets = np.arange(self.world_count) * prototype.body_count
        joint_offsets = np.arange(self.world_count) * prototype.joint_count
        self.plates = np.asarray(self.plates)[None, :] + body_offsets[:, None]
        self.flower_body_starts = flower_body_start + body_offsets
        self.flower_joint_indices = (np.array((opening, mount))[None, :] + joint_offsets[:, None]).reshape(-1)
        actuator_joints = (np.asarray(actuator_joints)[None, :] + joint_offsets[:, None]).reshape(-1)
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        for _ in range(self.world_count):
            builder.add_world(prototype)
        builder.add_ground_plane(color=(0.18, 0.21, 0.25))
        self.model = builder.finalize(skip_validation_joints=True)
        q_starts = self.model.joint_q_start.numpy()
        dof_starts = self.model.joint_qd_start.numpy()
        target_starts = self.model.joint_target_q_start.numpy()
        self.flower_q_indices = q_starts[self.flower_joint_indices]
        self.flower_coords_host = target_starts[self.flower_joint_indices]
        self.flower_coordinates = wp.array(self.flower_coords_host, dtype=int)
        self.flower_dofs = wp.array(dof_starts[self.flower_joint_indices], dtype=int)
        self.anchors_base = wp.array(anchors_base * self.world_count, dtype=wp.vec3)
        self.anchors_top = wp.array(anchors_top * self.world_count, dtype=wp.vec3)
        self.lengths_rest = wp.array(lengths_rest * self.world_count, dtype=float)
        self.actuator_q_indices = q_starts[actuator_joints]
        self.actuator_coords_host = target_starts[actuator_joints]
        self.actuator_dofs = wp.array(dof_starts[actuator_joints], dtype=int)
        self.actuator_coordinates = wp.array(self.actuator_coords_host, dtype=int)
        phases = np.arange(self.world_count) * 2.39996323
        self.variations_host = np.column_stack((0.3 * np.sin(phases), 1.0 + 0.04 * np.sin(phases), phases))
        self.variations = wp.array(self.variations_host, dtype=wp.vec3)

        config = SolverKamino.Config.from_model(
            self.model, dynamics_solver="dvi", sparse_dynamics=True, sparse_jacobian=True
        )
        config.use_fk_solver = False
        config.use_collision_detector = False
        config.constraints.alpha = 0.1
        config.constraints.beta = 0.011
        config.constraints.gamma = 0.015
        config.dynamics.preconditioning = False
        config.dynamics.linear_solver_type = "CR"
        config.dynamics.linear_solver_kwargs = {"maxiter": 9}
        config.dvi.bilateral_solver_type = "LLTBRCM"
        config.dvi.bilateral_solver_kwargs = {"parallel_factorization": True}
        config.dvi.tolerance = 1e-5
        config.dvi.regularization = 1e-8
        config.dvi.max_alternating_iterations = 4
        config.dvi.inequality_sweeps_per_iteration = 3
        config.dvi.bilateral_solve_interval = 1
        self.solver = SolverKamino(self.model, config=config)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.viewer.set_model(self.model)
        self.world_offsets = _garden_offsets(self.world_count)
        self.viewer.set_world_offsets((0.0, 0.0, 0.0))
        # Assign in place so rendering and picking retain the same offset array.
        # Visibility or spacing changes replace this custom layout; reassign it afterward.
        self.viewer.world_offsets.assign(self.world_offsets)
        if hasattr(self.viewer, "set_camera"):
            height = 0.1 * self.stage_count + 1.1
            span = float(np.ptp(self.world_offsets[:, :2], axis=0).max())
            distance = height + 0.45 * span
            elevation = 0.55 * height + 0.75 * span
            pitch = -math.degrees(math.atan2(elevation - 0.5 * height, math.hypot(0.8, 1.38) * distance))
            self.viewer.set_camera(wp.vec3(0.8 * distance, -1.38 * distance, elevation), pitch=pitch, yaw=120.0)
        # The song starts playing on the first step, once setup is complete.
        muted = args is None or getattr(args, "test", False) or getattr(args, "mute", False)
        self.song, moves = (None, _song_score()[1]) if muted else render_song()
        self.moves = wp.array(moves, dtype=float)
        # An odd substep count ends a frame on the other state buffer, so it
        # alternates between two graphs, one starting from each buffer.
        self.graphs = []
        if self.model.device.is_cuda and not wp.config.verify_cuda:
            for _ in range(2 if self.sim_substeps % 2 else 1):
                with wp.ScopedCapture() as capture:
                    self.simulate()
                self.graphs.append(capture.graph)
        self.frame = 0

    def simulate(self):
        for _ in range(self.sim_substeps):
            wp.launch(
                update_actuators,
                dim=6 * self.stage_count * self.world_count,
                inputs=[
                    self.time,
                    self.sim_dt,
                    self.stage_count,
                    self.variations,
                    self.moves,
                    self.anchors_base,
                    self.anchors_top,
                    self.lengths_rest,
                    self.actuator_dofs,
                    self.actuator_coordinates,
                    self.control.joint_target_q,
                    self.control.joint_target_qd,
                ],
            )
            wp.launch(
                update_flower,
                dim=2 * self.world_count,
                inputs=[
                    self.time,
                    self.sim_dt,
                    self.sphere.opening_slope,
                    self.sphere.reference_opening_slope,
                    self.adapt_flower_opening,
                    self.variations,
                    self.moves,
                    self.flower_coordinates,
                    self.flower_dofs,
                    self.control.joint_target_q,
                    self.control.joint_target_qd,
                ],
            )
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            wp.launch(advance_time, dim=1, inputs=[self.time, self.sim_dt])

    def step(self):
        if self.song is not None:
            self.player, self.song = play_song(self.song), None
        if self.player is not None:
            if self.viewer.is_paused():
                self.player.pause()
            else:
                self.player.play()
            self.time.fill_(self.player.time)
        if self.graphs:
            wp.capture_launch(self.graphs[self.frame % len(self.graphs)])
            if len(self.graphs) == 2:
                self.state_0, self.state_1 = self.state_1, self.state_0
        else:
            self.simulate()
        self.frame += 1
        self.sim_time += self.frame_dt
        if self.player is not None and self.viewer.is_paused():
            # Single stepping advances the silent audio cursor as well, so
            # resuming cannot rewind the choreography to the pause position.
            self.player.seek(self.player.time + self.frame_dt)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()
        if self.player is not None and self.viewer.is_paused():
            self.player.pause()

    def close(self):
        """Stop playback and release its audio worker."""
        player = getattr(self, "player", None)
        self.player = None
        self.song = None
        if player is not None:
            player.pause()
            player.delete()

    def __del__(self):
        # Reset and browser switching discard the previous example.
        self.close()

    def test_final(self):
        """Check tower tracking, flower breathing, and the mounted mechanism's loop closure."""
        poses = self.state_0.body_q.numpy()
        assert np.isfinite(poses).all(), "Non-finite body poses"
        assert self.model.world_count == self.world_count, "Flower worlds were not replicated"
        for plates in self.plates:
            # One implicit step per frame leaves the anchored base within a few tenths of a millimeter.
            np.testing.assert_allclose(poses[plates[0], :3], (0.0, 0.0, 0.012), atol=5e-4)
            np.testing.assert_allclose(poses[plates[0], 3:], (0.0, 0.0, 0.0, 1.0), atol=5e-4)
        parent = self.model.joint_parent.numpy()
        child = self.model.joint_child.numpy()
        xp = self.model.joint_X_p.numpy()
        xc = self.model.joint_X_c.numpy()
        for joint, label in enumerate(self.model.joint_label):
            if not (label.startswith("universal_") or label.startswith("/hoberman_sphere/") or label == "flower_mount"):
                continue
            a = wp.transform_point(wp.transform(*poses[parent[joint]]), wp.vec3(*xp[joint, :3]))
            b = wp.transform_point(wp.transform(*poses[child[joint]]), wp.vec3(*xc[joint, :3]))
            assert wp.length(a - b) < 0.001, f"Joint opened: {label}"
        joint_q = self.state_0.joint_q.numpy()
        target_q = self.control.joint_target_q.numpy()
        q = joint_q[self.actuator_q_indices]
        target = target_q[self.actuator_coords_host]
        assert np.max(np.abs(q)) < 0.026, "Piston exceeded its stroke"
        np.testing.assert_allclose(q, target, atol=0.002)
        flower_q = joint_q[self.flower_q_indices].reshape(self.world_count, 2)
        flower_target = target_q[self.flower_coords_host].reshape(self.world_count, 2)
        np.testing.assert_allclose(flower_q, flower_target, atol=0.01)
        assert np.max(np.abs(flower_q[:, 1] - 0.5 * flower_q[:, 0])) < 0.005, (
            "Flower mount lost opening synchronization"
        )
        if self.world_count > 1 and self.sim_time > 1.0:
            spread = np.max(np.ptp(q.reshape(self.world_count, -1), axis=0))
            assert 1e-5 < spread < 0.012, "Worlds lost their subtle motion variations"
        for world in range(self.world_count):
            start = self.flower_body_starts[world]
            points = poses[start : start + len(self.sphere.bodies), :3]
            radius = np.mean(np.linalg.norm(points - points.mean(axis=0), axis=1))
            radius_rest = self.sphere.radius_rest / self.sphere.diameter_rest
            assert 0.25 * radius_rest < radius < 1.8 * radius_rest, "Flower lost its spherical shape"
            opening = flower_q[world, 0]
            if abs(opening) > 0.04:
                assert (radius - radius_rest) * np.sign(opening) > 0.003, "Flower did not expand or contract"

    @staticmethod
    def create_parser():
        def positive_int(value: str) -> int:
            try:
                result = int(value)
            except ValueError:
                raise argparse.ArgumentTypeError("must be a positive integer") from None
            if result < 1:
                raise argparse.ArgumentTypeError("must be a positive integer")
            return result

        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(world_count=7)
        parser.add_argument(
            "--flower-resolution",
            type=positive_int,
            default=2,
            help="Scissor cells per 36-degree ring arc (2 reproduces the original sphere).",
        )
        parser.add_argument("--mute", action="store_true", help="Disable audio.")
        parser.add_argument("--stage-count", type=positive_int, default=20, help="Number of stacked Stewart platforms.")
        parser.add_argument(
            "--substeps",
            type=int,
            choices=(1, 2, 4, 6, 12),
            default=1,
            help="Physics steps per frame; presets give 60, 120, 240, 360 or 720 Hz at 60 fps.",
        )
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
