import os
import sys
import json
import math
import random
import hashlib
import argparse
import bpy
import mathutils
from bpy_extras.object_utils import world_to_camera_view

# Output resolution. The camera framing depends on the aspect ratio, so both live here.
RES_X = 800
RES_Y = 600
# Empty space kept around the tank + shadow, on both image axes (1.35 = 35% wider than the content)
FRAME_MARGIN = 1.35

def parse_args():
    # Since Blender executes this script with custom arguments after "--"
    # we need to find the index of "--" and parse everything after it.
    if "--" in sys.argv:
        args_idx = sys.argv.index("--") + 1
        args_list = sys.argv[args_idx:]
    else:
        args_list = []

    parser = argparse.ArgumentParser(description="Render synthetic oil tank and shadows in Blender.")
    parser.add_argument("--radius", type=float, default=20.0, help="Tank radius in meters")
    parser.add_argument("--height", type=float, default=15.0, help="Tank shell height in meters")
    parser.add_argument("--fill-level", type=float, default=0.5, help="Floating roof fill level (0.0 = empty, 1.0 = full)")
    parser.add_argument("--sun-elevation", type=float, default=45.0, help="Solar elevation angle in degrees")
    parser.add_argument("--sun-azimuth", type=float, default=135.0, help="Solar azimuth angle in degrees (0 = North, 90 = East)")
    parser.add_argument("--camera-elevation", type=float, default=90.0, help="Camera elevation angle in degrees (90 = nadir)")
    parser.add_argument("--camera-azimuth", type=float, default=0.0, help="Direction from the tank to the satellite in degrees (0 = North, 90 = East)")
    parser.add_argument("--ground-color", type=str, default="0.75,0.72,0.65", help="Ground RGB in 0-1, e.g. '0.80,0.70,0.52' for desert sand")
    parser.add_argument("--engine", type=str, default="EEVEE", choices=["EEVEE", "CYCLES"], help="Rendering engine to use")
    parser.add_argument("--atmosphere", type=str, default="CLEAR", choices=["CLEAR", "HAZE", "THIN_CLOUD", "OVERCAST"], help="Atmospheric condition ('CLEAR', 'HAZE', 'THIN_CLOUD'; 'OVERCAST' is an old alias of 'THIN_CLOUD')")
    parser.add_argument("--output", type=str, required=True, help="Absolute path for the output render file")
    parser.add_argument("--realism", type=str, default="FULL", choices=["FULL", "CLEAN"],
                        help="FULL: tank farm scene (textured ground, bunds, roads, pipes, neighbouring tanks, "
                             "roof details, rolling ladder). CLEAN: the single plain tank only.")
    parser.add_argument("--seed", type=int, default=-1,
                        help="Seed for the scene variation; -1 derives it from the output file name")
    parser.add_argument("--samples", type=int, default=64, help="Cycles samples (denoised)")
    parser.add_argument("--device", type=str, default="CPU", choices=["CPU", "CUDA", "OPTIX"],
                        help="Cycles device for the main render (masks always render on the CPU)")
    parser.add_argument("--zoom", type=float, default=1.0,
                        help="1 = tight frame around the tank and its shadow; larger shows more of the farm")
    parser.add_argument("--shift-x", type=float, default=0.0, help="Off-centre framing, -1..1 (image right)")
    parser.add_argument("--shift-y", type=float, default=0.0, help="Off-centre framing, -1..1 (image up)")
    parser.add_argument("--max-mpp", type=float, default=0.8,
                        help="Upper limit on metres per pixel when zooming out (0 = no limit)")
    parser.add_argument("--mpp", type=float, default=0.0,
                        help="> 0: fixed metres per pixel on the 800 x 600 frame (e.g. NAIP 0.6), instead of --zoom. "
                             "Never tighter than the frame that holds the tank and its shadow")
    parser.add_argument("--max-tanks", type=int, default=150, help="Most neighbouring tanks to build")
    parser.add_argument("--layout", type=str, default="random",
                        help="Tank-farm layout: random, dike_rows, staggered_rows, clusters, refinery_mixed, "
                             "single_row or isolated")
    parser.add_argument("--mask-output", type=str, default="",
                        help="Also write an instance/part segmentation mask PNG here")
    parser.add_argument("--vegetation", type=float, default=-1.0,
                        help="0 = bare ground (desert) .. 1 = mostly grassed with many trees; -1 = random")
    parser.add_argument("--neighbour-height-range", type=str, default="",
                        help="'lo,hi' in m: neighbouring tanks get a uniform height in this range, independent of "
                             "the labelled tank. Empty = legacy (scaled from the labelled tank's height)")
    parser.add_argument("--tank-density", type=float, default=-1.0,
                        help="Share of the layout's neighbouring tanks that are built, 0..1 (0 = the labelled tank "
                             "alone); -1 = all of them")
    # --- v3 physical variables (synthetic_data/src/v3). Defaults reproduce the legacy scene exactly. ---
    parser.add_argument("--phi-r", type=float, default=-1.0,
                        help="Normalised roof state phi_R = (z_R - z_F) / (z_C - z_F), 0 = roof deck on the floor, "
                             "1 = deck level with the rim. -1 = use --fill-level (legacy mapping)")
    parser.add_argument("--z-floor", type=float, default=0.2,
                        help="Tank floor elevation z_F in metres (lowest deck height); exterior terrain z_G is 0")
    parser.add_argument("--seal-gap", type=float, default=0.05, help="Roof-to-inner-wall gap g in metres")
    parser.add_argument("--roof-offset-x", type=float, default=0.0, help="Roof centre offset from the tank axis, east (m)")
    parser.add_argument("--roof-offset-y", type=float, default=0.0, help="Roof centre offset from the tank axis, north (m)")
    parser.add_argument("--roof-tilt", type=float, default=0.0, help="Roof deck tilt in degrees")
    parser.add_argument("--roof-tilt-azimuth", type=float, default=0.0, help="Downhill direction of the roof tilt (deg)")
    parser.add_argument("--terrain-slope", type=float, default=0.0, help="Exterior terrain slope in degrees")
    parser.add_argument("--terrain-slope-azimuth", type=float, default=0.0, help="Downhill direction of the terrain (deg)")
    parser.add_argument("--edge-type", type=str, default="none",
                        choices=["none", "rim_plate", "seal", "foam_dam", "pontoon_edge"],
                        help="Visible roof-edge structure eta_edge")
    parser.add_argument("--edge-height", type=float, default=0.0, help="Height of the edge structure above the deck (m)")
    parser.add_argument("--sun-angle", type=float, default=-1.0,
                        help="Sun disc angle in degrees (shadow softness); -1 = the atmosphere's default")
    parser.add_argument("--gsd", type=float, default=-1.0,
                        help="Fixed ground sample distance (m/px). The chip size then follows the framed extent "
                             "(tank + shadow, times --zoom). -1 = legacy 800 x 600 framing")
    parser.add_argument("--export-json", type=str, default="",
                        help="Write the exact camera and scene geometry (Layer A / B inputs) to this JSON file")
    parser.add_argument("--masks-dir", type=str, default="",
                        help="Also render an object-ID pass and a direct-sun shadow pass into this folder")

    return parser.parse_args(args_list)

def setup_render_engine(engine_name, samples=64, device="CPU"):
    scene = bpy.context.scene
    if engine_name == "CYCLES":
        scene.render.engine = 'CYCLES'
        # Set CPU rendering by default to avoid GPU dependency issues in headless containers
        scene.cycles.device = 'CPU'
        scene.cycles.samples = max(1, int(samples))
        scene.cycles.use_denoising = True
        if device != "CPU":
            prefs = bpy.context.preferences.addons["cycles"].preferences
            prefs.compute_device_type = device
            prefs.get_devices()
            gpus = [d for d in prefs.devices if d.type == device]
            if not gpus:
                raise RuntimeError(f"--device {device}: no {device} device found")
            for d in prefs.devices:
                d.use = d.type == device
            scene.cycles.device = 'GPU'
            scene.cycles.denoiser = 'OPENIMAGEDENOISE'   # the same denoiser as the CPU renders
    else:
        # Use Eevee/Blender Internal for fast rasterization
        # Blender 4.2-4.x names it BLENDER_EEVEE_NEXT; 5.x (and <4.2) use BLENDER_EEVEE
        engines = {e.identifier for e in scene.render.bl_rna.properties['engine'].enum_items}
        scene.render.engine = 'BLENDER_EEVEE_NEXT' if 'BLENDER_EEVEE_NEXT' in engines else 'BLENDER_EEVEE'
        if hasattr(scene, "eevee"):
            try:
                scene.eevee.use_shadows = True
            except AttributeError:
                pass
            try:
                scene.eevee.shadow_method = 'VSM'
            except AttributeError:
                pass
            try:
                # Enable ambient occlusion for clear depth contact shadows
                scene.eevee.use_gtao = True
                scene.eevee.gtao_distance = 1.5
            except AttributeError:
                try:
                    scene.eevee.use_ambient_occlusion = True
                except AttributeError:
                    pass

def clear_default_scene():
    # Remove default objects if they exist
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def _ground_with_hole(size, hole_radius, n=128):
    """Square ground plane with a round hole under the tank (v3 scenes). With the floor at grade
    (z_F = z_G = 0) an empty tank's roof deck would otherwise be coplanar with the ground and the two
    surfaces would shadow each other."""
    import bmesh
    mesh = bpy.data.meshes.new("GroundMesh")
    bm = bmesh.new()
    half = size / 2.0
    inner, outer = [], []
    for i in range(n):
        a = 2.0 * math.pi * i / n
        c, s_ = math.cos(a), math.sin(a)
        inner.append(bm.verts.new((hole_radius * c, hole_radius * s_, 0.0)))
        k = half / max(abs(c), abs(s_))
        outer.append(bm.verts.new((k * c, k * s_, 0.0)))
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new((inner[i], outer[i], outer[j], inner[j]))
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new("Ground", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    return obj


def create_environment(ground_color, radius, height, sun_elevation_deg, rng=None, vegetation=0.0, greenness=0.5,
                       slope_deg=0.0, slope_azimuth_deg=0.0, hole_radius=0.0):
    # Ground plane for external shadows. Low suns cast long shadows, so the plane grows with the
    # shadow length to keep the whole shadow (and the camera frame around it) on the ground.
    tan_elev = math.tan(math.radians(sun_elevation_deg))
    shadow_length = height / tan_elev if tan_elev > 0 else 0.0
    size = max(500.0, 4.0 * (radius + shadow_length))
    if hole_radius > 0:
        ground = _ground_with_hole(size, hole_radius)
    else:
        bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))
        ground = bpy.context.active_object
    ground.name = "Ground"
    if slope_deg:
        # Plane through the tank centre at z_G = 0, tilted down towards slope_azimuth
        s, a = math.radians(slope_deg), math.radians(slope_azimuth_deg)
        normal = mathutils.Vector((math.sin(s) * math.sin(a), math.sin(s) * math.cos(a), math.cos(s)))
        ground.rotation_mode = 'QUATERNION'
        ground.rotation_quaternion = mathutils.Vector((0.0, 0.0, 1.0)).rotation_difference(normal)

    if rng is None:
        # Flat colour (CLEAN mode)
        mat = bpy.data.materials.new(name="GroundMaterial")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs['Base Color'].default_value = (*ground_color, 1.0)  # set per location (sand, soil, concrete)
            bsdf.inputs['Roughness'].default_value = 0.95
    else:
        mat = ground_material(ground_color, rng, vegetation, greenness)
    ground.data.materials.append(mat)

WALL_THICKNESS = 0.35  # metres; the solidify modifier thickens the shell inwards
ROOF_THICKNESS = 0.2   # metres; the floating roof slab

# Part classes in the mask's green channel (see render_mask). 6 and 7 are not materials: they are
# painted from a direct-sun pass where the floating roof (6) or the ground around the tanks (7) lies in
# a shadow cast by a tank.
PART_IDS = {"shell_outer": 1, "shell_inner": 2, "floating_roof": 3, "fixed_roof": 4, "ladder": 5}
INNER_SHADOW_PART, GROUND_SHADOW_PART = 6, 7
_MATERIAL_CACHE = {}


def _shared(key, rng, make, variants=6):
    """
    Reuse one of a few material variants per kind, so a farm of 100+ tanks does not create hundreds of
    shader trees. The first tanks create the variants; later ones pick among them.
    """
    pool = _MATERIAL_CACHE.setdefault(key, [])
    if len(pool) < variants and (not pool or rng.random() < 0.5):
        pool.append(make(f"{key}{len(pool)}"))
    return rng.choice(pool)


def _tag(mat, part):
    mat["part"] = PART_IDS[part]
    return mat


def create_oil_tank(radius, height, fill_level, rng=None, center=(0.0, 0.0), name="Tank", kind="floating", tank_id=1,
                    phi_r=None, z_floor=0.2, seal_gap=0.05, roof_offset=(0.0, 0.0), roof_tilt=(0.0, 0.0),
                    edge=("none", 0.0), shell_bottom=0.0, floor_disc=False):
    """
    One tank. Geometry (shell, wall thickness, roof height) is the same as before, so the
    metadata ground truth still holds; rng adds realistic colours, roof details and a rolling ladder.
    kind "fixed_cone" (neighbours only) has a closed cone roof and no visible fill level.
    Every object of the tank carries tank_id (object colour), which the instance mask is rendered from.
    v3 options (labelled tank only): phi_r / z_floor set the deck height, plus seal gap, roof offset and
    tilt, a visible edge structure, a shell that continues below grade, and a floor disc under the roof.
    """
    before = set(bpy.data.objects)
    cx, cy = center
    # 1. Outer cylinder shell (no caps, to see the inside)
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=96 if rng else 64,
        radius=radius,
        depth=height - shell_bottom,
        end_fill_type='NGON' if kind == "fixed_cone" else 'NOTHING',
        location=(cx, cy, (height + shell_bottom) / 2.0)
    )
    tank_shell = bpy.context.active_object
    tank_shell.name = f"{name}Shell"

    # Add thickness to the shell wall using a solidify modifier
    solidify = tank_shell.modifiers.new(name="Solidify", type='SOLIDIFY')
    solidify.thickness = WALL_THICKNESS

    if rng is None:
        outer = _tag(flat_material(f"{name}ShellOuter", (0.9, 0.9, 0.92), roughness=0.4, metallic=0.6), "shell_outer")
        inner = _tag(flat_material(f"{name}ShellInner", (0.42, 0.21, 0.12), roughness=0.85, metallic=0.1), "shell_inner")
        rim_offset = 1
    else:
        # Shells are painted steel: mostly white, sometimes grey, beige or green-grey
        outer = _shared("ShellOuter", rng, lambda n: _tag(grimy_material(
            n, jitter(rng.choices(SHELL_COLOURS, weights=SHELL_WEIGHTS)[0], rng, 0.04), rng,
            roughness=0.5, grime=0.12, scale=0.25), "shell_outer"))
        inner = _shared("ShellInner", rng, lambda n: _tag(flat_material(
            n, jitter(rng.choice(INNER_WALL_COLOURS), rng, 0.08), roughness=0.85, metallic=0.0), "shell_inner"))
        rim_offset = 0  # the top edge is painted like the outside: a thin bright ring from above
    tank_shell.data.materials.append(outer)  # Index 0
    tank_shell.data.materials.append(inner)  # Index 1
    # Index 2: the rim faces get their own slot (same material as before) so the object-ID pass can
    # label the wall top separately from the inner and outer wall
    tank_shell.data.materials.append(inner if rim_offset == 1 else outer)
    solidify.material_offset = 1
    solidify.material_offset_rim = 2

    info = {"kind": kind, "center_m": [round(cx, 3), round(cy, 3)], "radius_m": round(radius, 3),
            "height_m": round(height, 3)}

    if kind == "fixed_cone":
        cone_h = radius * rng.uniform(0.06, 0.12)
        bpy.ops.mesh.primitive_cone_add(vertices=96, radius1=radius, radius2=0.6, depth=cone_h,
                                        location=(cx, cy, height + cone_h / 2.0))
        cone = bpy.context.active_object
        cone.name = f"{name}Cone"
        cone.data.materials.append(_shared("FixedRoof", rng, lambda n: _tag(grimy_material(
            n, jitter(rng.choices(SHELL_COLOURS, weights=SHELL_WEIGHTS)[0], rng, 0.05), rng,
            roughness=0.5, grime=0.15, scale=0.3), "fixed_roof")))
        info["fill_level"] = None
    else:
        # 2. Floating roof inside. Legacy: slab centre at 0.1 + fill * (height - 0.3), i.e. deck top
        #    z_R = 0.2 + fill * (height - 0.3). v3: deck top z_R = z_F + phi_R * (z_C - z_F).
        if phi_r is None:
            roof_z = 0.1 + fill_level * (height - 0.3)
        else:
            roof_z = z_floor + phi_r * (height - z_floor) - ROOF_THICKNESS / 2.0
        deck_z = roof_z + ROOF_THICKNESS / 2.0
        roof_radius = radius - WALL_THICKNESS - seal_gap  # legacy: radius - 0.4 (gap 0.05 m)
        rox, roy = roof_offset
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=96 if rng else 64,
            radius=roof_radius,
            depth=ROOF_THICKNESS,
            end_fill_type='NGON',
            location=(cx + rox, cy + roy, roof_z)
        )
        floating_roof = bpy.context.active_object
        floating_roof.name = f"{name}FloatingRoof"
        tilt_q = None
        if roof_tilt[0]:
            t, a = math.radians(roof_tilt[0]), math.radians(roof_tilt[1])
            normal = mathutils.Vector((math.sin(t) * math.sin(a), math.sin(t) * math.cos(a), math.cos(t)))
            tilt_q = mathutils.Vector((0.0, 0.0, 1.0)).rotation_difference(normal)
            floating_roof.rotation_mode = 'QUATERNION'
            floating_roof.rotation_quaternion = tilt_q
        if rng is None:
            floating_roof.data.materials.append(_tag(
                flat_material(f"{name}Roof", (0.75, 0.75, 0.78), roughness=0.6, metallic=0.7), "floating_roof"))
        else:
            floating_roof.data.materials.append(
                _shared("Roof", rng, lambda n: _tag(roof_material(n, rng), "floating_roof")))
            info.update(add_rolling_ladder(name, cx, cy, radius, height, roof_z + 0.1, rng))
        edge_type, edge_h = edge
        if edge_type != "none":
            add_roof_edge(name, cx + rox, cy + roy, deck_z, roof_radius, seal_gap, edge_type, edge_h,
                          floating_roof.data.materials[0], tilt_q=tilt_q, pivot_z=roof_z)
        if floor_disc:
            # Tank bottom below the deck, so the seal gap never shows the empty world through the
            # ground hole (v3 scenes) or past a below-grade floor
            bpy.ops.mesh.primitive_circle_add(vertices=96, radius=radius - WALL_THICKNESS / 2.0, fill_type='NGON',
                                              location=(cx, cy, min(z_floor, deck_z) - ROOF_THICKNESS - 0.05))
            floor = bpy.context.active_object
            floor.name = f"{name}Floor"
            floor.data.materials.append(_tag(flat_material(f"{name}FloorMat", (0.12, 0.11, 0.10), roughness=0.9),
                                             "shell_inner"))
        info["fill_level"] = round(fill_level, 3)
        info.update({"deck_z_m": round(deck_z, 4), "roof_radius_m": round(roof_radius, 4),
                     "roof_offset_m": [rox, roy], "edge_type": edge_type, "edge_height_m": edge_h})

    for obj in set(bpy.data.objects) - before:
        obj.color = (tank_id / 255.0, 0.0, 0.0, 1.0)
    info["instance_id"] = tank_id
    return info


def add_roof_edge(name, cx, cy, deck_z, roof_radius, seal_gap, edge_type, edge_h, roof_mat, tilt_q=None, pivot_z=None):
    """
    Visible roof-edge structure (eta_edge) on the floating roof: a ring whose top sits edge_h above the
    deck at radius roof_radius (foam dam: 0.6 m further in), as in src/v3/physics.py.
    - seal: a dark rubber ring that fills only the rim space, from the roof edge (roof_radius) out to the
      inner wall (roof_radius + seal_gap). (It used to be a solid dark disc over the whole roof, which made
      the roof black, or half black with a straight edge when the deck was tilted.)
    - rim_plate / foam_dam: thin plates; pontoon_edge: a 2.5 m wide pontoon ring.
    The ring is built level around (cx, cy) and then tilted with the deck: tilt_q is the deck's rotation and
    pivot_z the height of the deck slab's centre it rotates about.
    """
    if edge_type == "seal":
        # From 0.1 m below the deck (hides the gap edge) up to edge_h above it; width = the seal gap
        mat = _tag(flat_material(f"{name}SealMat", (0.08, 0.08, 0.08), roughness=0.9), "floating_roof")
        outer, width, top, bottom = roof_radius + seal_gap, seal_gap, deck_z + max(edge_h, 0.005), deck_z - 0.1
    else:
        mat = roof_mat
        outer = {"rim_plate": roof_radius, "foam_dam": roof_radius - 0.6, "pontoon_edge": roof_radius}[edge_type]
        width = 2.5 if edge_type == "pontoon_edge" else 0.08
        top, bottom = deck_z + max(edge_h, 0.02), deck_z
    bpy.ops.mesh.primitive_cylinder_add(vertices=96, radius=outer, depth=top - bottom, end_fill_type='NOTHING',
                                        location=(cx, cy, (top + bottom) / 2.0))
    obj = bpy.context.active_object
    # Solidify thickens inwards (against the outward normals): the ring spans outer - width .. outer
    mod = obj.modifiers.new(name="Solidify", type='SOLIDIFY')
    mod.thickness = width
    obj.data.materials.append(mat)
    obj.name = f"{name}Edge"
    if tilt_q is not None:
        pivot = mathutils.Vector((cx, cy, pivot_z if pivot_z is not None else deck_z))
        obj.location = pivot + tilt_q @ (obj.location - pivot)
        obj.rotation_mode = 'QUATERNION'
        obj.rotation_quaternion = tilt_q
    return obj


# ---------------------------------------------------------------------------------------------
# Scene realism (--realism FULL). Real very-high-resolution images of crude tanks show tank farms:
# rows of tanks in earth-walled bunds separated by roads, pipe runs along the roads, textured soil or
# gravel, painted shells with a bright rim, floating roofs with a dark seal ring, a pontoon ring and a
# rolling ladder. The labelled tank's geometry is unchanged; neighbours are placed so their shadows and
# their off-nadir lean never touch the labelled tank or its shadow.
# ---------------------------------------------------------------------------------------------
SHELL_COLOURS = [(0.88, 0.88, 0.86), (0.80, 0.81, 0.80), (0.82, 0.78, 0.68), (0.58, 0.64, 0.57), (0.66, 0.67, 0.68)]
SHELL_WEIGHTS = [0.55, 0.15, 0.12, 0.08, 0.10]
ROOF_COLOURS = [(0.80, 0.80, 0.78), (0.70, 0.71, 0.70), (0.58, 0.58, 0.56), (0.66, 0.62, 0.55), (0.86, 0.86, 0.84)]
# Weathered grey / light-rust steel. Real crescents (NAIP, 22 tanks) are only ~2x darker than the lit roof,
# because the sunlit inner wall bounces light into them; the old dark rust walls made them ~2.7x darker.
INNER_WALL_COLOURS = [(0.68, 0.69, 0.71), (0.62, 0.63, 0.66), (0.71, 0.71, 0.70), (0.64, 0.62, 0.59)]


def jitter(rgb, rng, amount):
    b = 1.0 + rng.uniform(-amount, amount)
    return tuple(min(1.0, max(0.0, c * b * (1.0 + rng.uniform(-amount / 3, amount / 3)))) for c in rgb)


def scale_rgb(rgb, k):
    return tuple(min(1.0, max(0.0, c * k)) for c in rgb)


def _new_material(name):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    return mat, mat.node_tree.nodes, mat.node_tree.links, mat.node_tree.nodes.get("Principled BSDF")


def flat_material(name, rgb, roughness=0.6, metallic=0.0):
    mat, _, _, bsdf = _new_material(name)
    bsdf.inputs['Base Color'].default_value = (*rgb, 1.0)
    bsdf.inputs['Roughness'].default_value = roughness
    bsdf.inputs['Metallic'].default_value = metallic
    return mat


def _ramp(nodes, stops, constant=False):
    """ColorRamp from [(position, value_or_rgb), ...]. Scalars become grey levels."""
    node = nodes.new('ShaderNodeValToRGB')
    cr = node.color_ramp
    cr.interpolation = 'CONSTANT' if constant else 'LINEAR'
    while len(cr.elements) < len(stops):
        cr.elements.new(0.5)
    for el, (pos, val) in zip(cr.elements, sorted(stops, key=lambda s: s[0])):
        el.position = pos
        rgb = (val, val, val) if isinstance(val, (int, float)) else val
        el.color = (*rgb, 1.0)
    return node


def _noise(nodes, links, coord, scale, detail=4.0, roughness=0.55):
    n = nodes.new('ShaderNodeTexNoise')
    n.inputs['Scale'].default_value = scale
    n.inputs['Detail'].default_value = detail
    n.inputs['Roughness'].default_value = roughness
    links.new(coord, n.inputs['Vector'])
    return n.outputs['Fac']


def _math(nodes, links, op, a, b):
    m = nodes.new('ShaderNodeMath')
    m.operation = op
    for i, v in enumerate((a, b)):
        if isinstance(v, (int, float)):
            m.inputs[i].default_value = v
        else:
            links.new(v, m.inputs[i])
    return m.outputs['Value']


def _object_coords(nodes, offset_rng):
    """Object-space coordinates (metres) with a random offset so every texture is different."""
    tc = nodes.new('ShaderNodeTexCoord')
    add = nodes.new('ShaderNodeVectorMath')
    add.operation = 'ADD'
    add.inputs[1].default_value = tuple(offset_rng.uniform(-500, 500) for _ in range(3))
    return tc, add


def grimy_material(name, rgb, rng, roughness=0.5, grime=0.12, scale=0.25):
    """Painted steel with uneven weathering."""
    mat, nodes, links, bsdf = _new_material(name)
    tc, add = _object_coords(nodes, rng)
    links.new(tc.outputs['Object'], add.inputs[0])
    colour = _ramp(nodes, [(0.3, scale_rgb(rgb, 1.0 - grime)), (0.7, rgb)])
    links.new(_noise(nodes, links, add.outputs['Vector'], scale, detail=6.0), colour.inputs['Fac'])
    links.new(colour.outputs['Color'], bsdf.inputs['Base Color'])
    bsdf.inputs['Roughness'].default_value = roughness
    bsdf.inputs['Metallic'].default_value = 0.0
    return mat


def roof_material(name, rng):
    """
    Floating-roof deck seen from above: weathered steel, a faint pontoon ring, a dark rim seal next
    to the wall and a central drain, as visible in sub-metre imagery.
    """
    mat, nodes, links, bsdf = _new_material(name)
    base = jitter(rng.choice(ROOF_COLOURS), rng, 0.05)
    tc, add = _object_coords(nodes, rng)
    # Distance from the roof centre as a fraction of its radius, from the object's own bounding box,
    # so the same material works for every tank size
    centred = nodes.new('ShaderNodeVectorMath')
    centred.operation = 'MULTIPLY_ADD'
    links.new(tc.outputs['Generated'], centred.inputs[0])
    centred.inputs[1].default_value = (2.0, 2.0, 0.0)
    centred.inputs[2].default_value = (-1.0, -1.0, 0.0)
    length = nodes.new('ShaderNodeVectorMath')
    length.operation = 'LENGTH'
    links.new(centred.outputs['Vector'], length.inputs[0])
    d = length.outputs['Value']
    seal = rng.uniform(0.95, 0.975)
    pontoon = rng.uniform(0.72, 0.86)
    pdark = rng.uniform(0.70, 0.88)
    ring = _ramp(nodes, [(0.0, 0.35), (0.02, 0.35), (0.025, 1.0), (pontoon - 0.012, 1.0), (pontoon, pdark),
                         (pontoon + 0.012, 1.0), (seal - 0.005, 1.0), (seal, 0.35), (1.0, 0.30)])
    links.new(d, ring.inputs['Fac'])
    links.new(tc.outputs['Object'], add.inputs[0])
    grime = _ramp(nodes, [(0.3, rng.uniform(0.72, 0.88)), (0.7, 1.0)])
    links.new(_noise(nodes, links, add.outputs['Vector'], rng.uniform(0.15, 0.4), detail=6.0), grime.inputs['Fac'])
    shade = _math(nodes, links, 'MULTIPLY', ring.outputs['Color'], grime.outputs['Color'])
    colour = _ramp(nodes, [(0.0, scale_rgb(base, 0.25)), (1.0, base)])
    links.new(shade, colour.inputs['Fac'])
    links.new(colour.outputs['Color'], bsdf.inputs['Base Color'])
    bsdf.inputs['Roughness'].default_value = 0.6
    bsdf.inputs['Metallic'].default_value = 0.2
    return mat


def add_box(name, p0, p1, width, thickness, mat):
    """Thin box from point p0 to p1 (a beam, ladder or pipe rack)."""
    a, b = mathutils.Vector(p0), mathutils.Vector(p1)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(a + b) / 2.0)
    obj = bpy.context.active_object
    obj.name = name
    direction = (b - a)
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = direction.to_track_quat('X', 'Z')
    obj.scale = (direction.length, width, thickness)
    obj.data.materials.append(mat)
    return obj


def add_rolling_ladder(name, cx, cy, radius, height, roof_top, rng):
    """
    Gauger's platform on the rim and the rolling ladder down to the roof: in satellite images a
    dark line from the rim towards the roof centre whose slope changes with the fill level.
    """
    az = rng.uniform(0.0, 2.0 * math.pi)
    ux, uy = math.sin(az), math.cos(az)
    steel = _shared("Ladder", rng, lambda n: _tag(flat_material(n, jitter((0.32, 0.30, 0.28), rng, 0.2), roughness=0.7), "ladder"), 3)
    top = (cx + ux * (radius - 0.8), cy + uy * (radius - 0.8), height + 0.6)
    add_box(f"{name}Platform", (cx + ux * (radius - 1.6), cy + uy * (radius - 1.6), height + 0.5),
            (cx + ux * (radius + 0.8), cy + uy * (radius + 0.8), height + 0.5), 1.8, 0.15, steel)
    drop = top[2] - roof_top
    length = max(0.85 * radius, 1.1 * drop)
    run = min(math.sqrt(max(length ** 2 - drop ** 2, 1.0)), 1.6 * radius)
    bottom = (top[0] - ux * run, top[1] - uy * run, roof_top + 0.2)
    add_box(f"{name}Ladder", top, bottom, 0.9, 0.35, steel)
    return {"ladder_azimuth_degrees": round(math.degrees(az) % 360.0, 2)}


def add_pad(name, cx, cy, radius, mat):
    """Ring foundation / gravel apron at the tank base."""
    bpy.ops.mesh.primitive_circle_add(vertices=96, radius=radius, fill_type='NGON', location=(cx, cy, 0.015))
    pad = bpy.context.active_object
    pad.name = f"{name}Pad"
    pad.data.materials.append(mat)


def add_pipe(name, p0, p1, radius, mat):
    a, b = mathutils.Vector(p0), mathutils.Vector(p1)
    bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=radius, depth=(b - a).length, location=(a + b) / 2)
    pipe = bpy.context.active_object
    pipe.name = name
    pipe.rotation_mode = 'QUATERNION'
    pipe.rotation_quaternion = (b - a).to_track_quat('Z', 'Y')
    pipe.data.materials.append(mat)


def _seg_dist(p, q, r, s):
    """Shortest distance between 2D segments pq and rs."""
    def point_seg(x, a, b):
        ab = (b[0] - a[0], b[1] - a[1])
        L2 = ab[0] ** 2 + ab[1] ** 2
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((x[0] - a[0]) * ab[0] + (x[1] - a[1]) * ab[1]) / L2))
        return math.hypot(x[0] - a[0] - t * ab[0], x[1] - a[1] - t * ab[1])

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    if (cross(p, q, r) * cross(p, q, s) < 0) and (cross(r, s, p) * cross(r, s, q) < 0):
        return 0.0
    return min(point_seg(p, r, s), point_seg(q, r, s), point_seg(r, p, q), point_seg(s, p, q))


def _footprint(center, height, sun_el, sun_az, cam_el, cam_az):
    """Ground segments a tank occupies in the image: its shadow and its off-nadir lean."""
    cx, cy = center
    sh = height / math.tan(math.radians(sun_el))
    lean = height / math.tan(math.radians(cam_el)) if cam_el < 89.99 else 0.0
    s_az, c_az = math.radians(sun_az), math.radians(cam_az)
    return [((cx, cy), (cx - sh * math.sin(s_az), cy - sh * math.cos(s_az))),
            ((cx, cy), (cx - lean * math.sin(c_az), cy - lean * math.cos(c_az)))]


def _mix(nodes, links, fac, a, b):
    """Colour mix that works on Blender 3.4+ (Mix node) and older (MixRGB)."""
    def put(sock, v):
        if isinstance(v, (int, float)):
            sock.default_value = v
        elif isinstance(v, tuple):
            sock.default_value = (*v[:3], 1.0)
        else:
            links.new(v, sock)
    try:
        n = nodes.new('ShaderNodeMix')
        n.data_type = 'RGBA'
        ins = {s.identifier: s for s in n.inputs}
        put(ins['Factor_Float'], fac)
        put(ins['A_Color'], a)
        put(ins['B_Color'], b)
        return {s.identifier: s for s in n.outputs}['Result_Color']
    except (RuntimeError, KeyError):
        n = nodes.new('ShaderNodeMixRGB')
        put(n.inputs['Fac'], fac)
        put(n.inputs['Color1'], a)
        put(n.inputs['Color2'], b)
        return n.outputs['Color']


GRASS_GREEN = ((0.10, 0.20, 0.06), (0.22, 0.34, 0.12))   # dark / light healthy grass
GRASS_DRY = ((0.30, 0.27, 0.16), (0.52, 0.47, 0.30))     # dark / light dry grass


def ground_material(rgb, rng, vegetation=0.0, greenness=0.5, name="GroundMaterial"):
    """
    Soil / gravel with large patches, fine grain and dark stains, plus grass where vegetation > 0.
    vegetation 0..1 is how much of the ground is grassed; greenness 0..1 goes from dry to green.
    """
    mat, nodes, links, bsdf = _new_material(name)
    tc, add = _object_coords(nodes, rng)
    links.new(tc.outputs['Object'], add.inputs[0])
    coord = add.outputs['Vector']
    patches = _noise(nodes, links, coord, rng.uniform(0.012, 0.03), detail=3.0)
    grain = _noise(nodes, links, coord, rng.uniform(0.6, 1.2), detail=8.0, roughness=0.7)
    mix = _math(nodes, links, 'ADD', _math(nodes, links, 'MULTIPLY', patches, 0.65),
                _math(nodes, links, 'MULTIPLY', grain, 0.35))
    stain_ramp = _ramp(nodes, [(0.0, 1.0), (rng.uniform(0.62, 0.70), rng.uniform(0.55, 0.75))], constant=True)
    links.new(_noise(nodes, links, coord, rng.uniform(0.05, 0.12), detail=5.0), stain_ramp.inputs['Fac'])
    shade = _math(nodes, links, 'MULTIPLY', mix, stain_ramp.outputs['Color'])
    soil = _ramp(nodes, [(0.0, scale_rgb(rgb, 0.40)), (0.36, scale_rgb(rgb, 0.76)),
                         (0.50, rgb), (0.62, scale_rgb(rgb, 1.12))])
    links.new(shade, soil.inputs['Fac'])
    colour = soil.outputs['Color']

    if vegetation > 0.02:
        dark = tuple(d + (g - d) * greenness for d, g in zip(GRASS_DRY[0], GRASS_GREEN[0]))
        light = tuple(d + (g - d) * greenness for d, g in zip(GRASS_DRY[1], GRASS_GREEN[1]))
        grass = _ramp(nodes, [(0.3, jitter(dark, rng, 0.1)), (0.7, jitter(light, rng, 0.1))])
        links.new(_noise(nodes, links, coord, rng.uniform(0.3, 0.8), detail=8.0, roughness=0.7), grass.inputs['Fac'])
        thr = 0.72 - 0.42 * min(vegetation, 1.0)
        soft = rng.uniform(0.02, 0.06)
        mask = _ramp(nodes, [(thr - soft, 0.0), (thr + soft, 1.0)])
        links.new(_noise(nodes, links, coord, rng.uniform(0.015, 0.05), detail=5.0), mask.inputs['Fac'])
        colour = _mix(nodes, links, mask.outputs['Color'], colour, grass.outputs['Color'])
    links.new(colour, bsdf.inputs['Base Color'])
    bsdf.inputs['Roughness'].default_value = 0.95
    return mat


# ---------- bund walls along any closed outline ----------
def add_wall_loop(name, pts, height, base_w, mat):
    """
    One closed earth wall following the outline pts [(x, y), ...]: trapezoid cross-section with sloped
    faces, built as a single mesh with mitred corners (no overlapping faces).
    """
    import bmesh
    # Drop repeated points (a zero corner radius repeats each corner); zero-length edges break the mitres
    clean = []
    for p in pts:
        if not clean or math.hypot(p[0] - clean[-1][0], p[1] - clean[-1][1]) > 0.05:
            clean.append(p)
    if len(clean) > 2 and math.hypot(clean[0][0] - clean[-1][0], clean[0][1] - clean[-1][1]) <= 0.05:
        clean.pop()
    pts = clean
    area = sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1] for i in range(len(pts)))
    if area < 0:
        pts = pts[::-1]
    n = len(pts)
    top_w = base_w * 0.3
    verts = []
    for i in range(n):
        p, a, b = pts[i], pts[i - 1], pts[(i + 1) % n]
        normals = []
        for (x0, y0), (x1, y1) in ((a, p), (p, b)):
            dx, dy = x1 - x0, y1 - y0
            L = math.hypot(dx, dy) or 1.0
            normals.append((dy / L, -dx / L))  # outward for a counter-clockwise outline
        mx, my = normals[0][0] + normals[1][0], normals[0][1] + normals[1][1]
        L = math.hypot(mx, my) or 1.0
        mx, my = mx / L, my / L
        k = 1.0 / max(0.35, mx * normals[0][0] + my * normals[0][1])
        for off, z in ((base_w / 2, 0.0), (top_w / 2, height), (-top_w / 2, height), (-base_w / 2, 0.0)):
            verts.append((p[0] + mx * off * k, p[1] + my * off * k, z))
    faces = []
    for i in range(n):
        j = (i + 1) % n
        for s in range(3):
            faces.append((4 * i + s, 4 * j + s, 4 * j + s + 1, 4 * i + s + 1))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)
    return obj


def outline_rounded_rect(x0, y0, x1, y1, corner, rng, wobble=0.0, arc_steps=6):
    """Rectangle with rounded (or, with arc_steps=1, chamfered) corners and slightly uneven sides."""
    corner = max(0.0, min(corner, 0.45 * min(x1 - x0, y1 - y0)))
    pts = []
    for cx, cy, a0 in ((x1 - corner, y1 - corner, 0.0), (x0 + corner, y1 - corner, 90.0),
                       (x0 + corner, y0 + corner, 180.0), (x1 - corner, y0 + corner, 270.0)):
        cx += rng.uniform(-wobble, wobble)
        cy += rng.uniform(-wobble, wobble)
        for s in range(arc_steps + 1):
            a = math.radians(a0 + 90.0 * s / arc_steps)
            pts.append((cx + corner * math.cos(a), cy + corner * math.sin(a)))
    return pts


def outline_circle(cx, cy, r, steps=72):
    return [(cx + r * math.cos(2 * math.pi * i / steps), cy + r * math.sin(2 * math.pi * i / steps)) for i in range(steps)]


# ---------- trees and shrubs ----------
def _crown_meshes(rng, count=4):
    """A few lumpy crown shapes, shared by all trees (linked duplicates render fast)."""
    meshes = []
    for k in range(count):
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=1.0)
        obj = bpy.context.active_object
        for v in obj.data.vertices:
            v.co *= rng.uniform(0.8, 1.15)
        meshes.append(obj.data)
        bpy.data.objects.remove(obj)
    return meshes


def add_trees(rng, spots, vegetation, greenness, is_free, sun_el):
    """
    Trees and shrubs: clusters, scattered shrubs and lines along roads. spots are candidate (x, y)
    centres for clusters; is_free(x, y, crown, height) rejects places on roads, in bunds, near tanks or
    where the shadow would touch the labelled tank.
    """
    if vegetation < 0.03 or not spots:
        return 0
    meshes = _crown_meshes(rng)
    tones = [jitter(tuple(d + (g - d) * greenness for d, g in zip((0.22, 0.22, 0.12), (0.06, 0.13, 0.04))), rng, 0.15)
             for _ in range(3)]
    mats = [grimy_material(f"Foliage{i}", t, rng, roughness=0.9, grime=0.35, scale=0.6) for i, t in enumerate(tones)]
    for m in meshes:
        m.materials.append(mats[0])
    placed = 0
    budget = int(40 + 360 * vegetation)
    for (sx, sy, spread, big) in spots:
        for _ in range(int(rng.uniform(3, 25) * vegetation) + 1):
            if placed >= budget:
                return placed
            x, y = rng.gauss(sx, spread), rng.gauss(sy, spread)
            if big:
                h, crown = rng.uniform(5.0, 14.0), rng.uniform(2.0, 4.5)
            else:
                h, crown = rng.uniform(1.0, 3.0), rng.uniform(0.8, 1.8)
            if not is_free(x, y, crown, h):
                continue
            obj = bpy.data.objects.new("Tree", rng.choice(meshes))
            bpy.context.collection.objects.link(obj)
            ch = min(h * 0.6, crown * 1.6)
            obj.location = (x, y, h - ch / 2.0)
            obj.scale = (crown * rng.uniform(0.85, 1.15), crown * rng.uniform(0.85, 1.15), ch / 2.0)
            obj.rotation_euler = (0.0, 0.0, rng.uniform(0, 2 * math.pi))
            slot = obj.material_slots[0]
            slot.link = 'OBJECT'      # per-tree colour while the crown mesh stays shared
            slot.material = rng.choice(mats)
            placed += 1
    return placed


def _rot(p, th):
    c, s = math.cos(th), math.sin(th)
    return (p[0] * c - p[1] * s, p[0] * s + p[1] * c)


def _pt_seg(pt, a, b):
    ab = (b[0] - a[0], b[1] - a[1])
    L2 = ab[0] ** 2 + ab[1] ** 2
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((pt[0] - a[0]) * ab[0] + (pt[1] - a[1]) * ab[1]) / L2))
    return math.hypot(pt[0] - a[0] - t * ab[0], pt[1] - a[1] - t * ab[1])


def _in_poly(pt, poly):
    x, y = pt
    inside = False
    for i in range(len(poly)):
        (x0, y0), (x1, y1) = poly[i - 1], poly[i]
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def add_road(name, a, b, width, mat, z):
    dx, dy = b[0] - a[0], b[1] - a[1]
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, z))
    road = bpy.context.active_object
    road.name = name
    road.rotation_euler = (0.0, 0.0, math.atan2(dy, dx))
    road.scale = (math.hypot(dx, dy), width, 1.0)
    road.data.materials.append(mat)


def add_building(name, x, y, w, l, h, angle, mat):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, y, h / 2))
    b = bpy.context.active_object
    b.name = name
    b.rotation_euler = (0.0, 0.0, angle)
    b.scale = (l, w, h)
    b.data.materials.append(mat)


# ---------- real tank-farm layouts ----------
# Each generator returns a layout in its own local frame (metres):
#   tanks: [{"x", "y", "r", "target"}], dikes: [outline], roads: [((x0, y0), (x1, y1), width)],
#   sites: [outline] (the fenced facility area, gravel / concrete instead of natural ground)
# build_tank_farm moves the labelled tank to the origin and rotates the whole site by a random angle.

def _rot(p, ang, off=(0.0, 0.0)):
    c, s = math.cos(ang), math.sin(ang)
    return (p[0] * c - p[1] * s + off[0], p[0] * s + p[1] * c + off[1])


def _transform_layout(L, ang, off=(0.0, 0.0)):
    return {"tanks": [dict(t, x=_rot((t["x"], t["y"]), ang, off)[0], y=_rot((t["x"], t["y"]), ang, off)[1]) for t in L["tanks"]],
            "dikes": [[_rot(p, ang, off) for p in d] for d in L["dikes"]],
            "roads": [(_rot(a, ang, off), _rot(b, ang, off), w) for a, b, w in L["roads"]],
            "sites": [[_rot(p, ang, off) for p in s] for s in L["sites"]],
            "per_tank_dikes": L.get("per_tank_dikes", False)}


def _spacing(rmax, rng):
    return max(6.0, 2.0 * rmax * rng.uniform(0.3, 0.75))   # shell to shell, ~0.3-0.75 D


def gen_dike_rows(R, rng, extent, staggered=False, with_target=True, max_dikes=(5, 5)):
    """Dikes of 1-2 rows x 1-4 tanks, laid out in blocks with roads between them (most terminals)."""
    mixed = rng.random() < 0.45
    rmax = R * (rng.uniform(1.0, 1.3) if mixed else 1.0)
    s = _spacing(rmax, rng)
    pitch = 2 * rmax + s
    per_row, rows = rng.choice([1, 2, 2, 3, 3, 4]), rng.choice([1, 2, 2])
    stag = pitch / 2 if staggered and rows == 2 else 0.0
    margin = s / 2 + rng.uniform(2.0, 6.0)
    dw = 2 * margin + 2 * rmax + (per_row - 1) * pitch + stag
    dh = 2 * margin + 2 * rmax + (rows - 1) * pitch
    road = rng.uniform(6.0, 12.0)
    nx = rng.randint(1, max(1, min(max_dikes[0], int(extent / (dw + road)) + 1)))
    ny = rng.randint(1, max(1, min(max_dikes[1], int(extent / (dh + road)) + 1)))
    present = [(a, b) for a in range(nx) for b in range(ny) if rng.random() < 0.9] or [(0, 0)]
    target_dike = rng.choice(present)
    corner, steps = rng.choice([0.0, 0.0, rng.uniform(2.0, 12.0)]), rng.choice([1, 6])
    L = {"tanks": [], "dikes": [], "roads": [], "sites": []}
    slots_target = []
    for a, b in present:
        ox, oy = a * (dw + road), b * (dh + road)
        rd = R if ((a, b) == target_dike or not mixed) else min(rmax, R * rng.choice([0.6, 0.75, 0.9, 1.0, 1.2]))
        for k in range(rows):
            for c in range(per_row):
                x = ox + margin + rmax + c * pitch + (stag if k % 2 else 0.0)
                y = oy + margin + rmax + k * pitch
                if (a, b) == target_dike:
                    slots_target.append(len(L["tanks"]))
                elif rng.random() < 0.1:
                    continue  # an empty slot (tank demolished / not yet built)
                L["tanks"].append({"x": x, "y": y, "r": rd, "target": False})
        L["dikes"].append(outline_rounded_rect(ox, oy, ox + dw, oy + dh, corner, rng, wobble=0.8, arc_steps=steps))
    if with_target:
        L["tanks"][rng.choice(slots_target)]["target"] = True
    W, Hh = nx * (dw + road) - road, ny * (dh + road) - road
    for a in range(1, nx):
        x = a * (dw + road) - road / 2
        L["roads"].append(((x, -road), (x, Hh + road), road))
    for b in range(1, ny):
        y = b * (dh + road) - road / 2
        L["roads"].append(((-road, y), (W + road, y), road))
    if rng.random() < 0.6:  # perimeter road on one side
        side = rng.choice(["s", "n", "w", "e"])
        y = -road / 2 - 2 if side == "s" else Hh + road / 2 + 2
        x = -road / 2 - 2 if side == "w" else W + road / 2 + 2
        L["roads"].append(((-road * 2, y), (W + road * 2, y), road) if side in "sn" else ((x, -road * 2), (x, Hh + road * 2), road))
    pad = rng.uniform(6.0, 25.0)
    L["sites"].append(outline_rounded_rect(-pad, -pad, W + pad, Hh + pad, rng.uniform(0, 20), rng, wobble=2.0))
    return L


def gen_refinery(R, rng, extent):
    """Mixed sizes packed irregularly (refineries, product terminals)."""
    A = min(extent * 0.6, R * rng.uniform(4.0, 9.0) + 30.0)
    s_min = max(5.0, 0.6 * R * rng.uniform(0.4, 0.8))
    tanks = [{"x": 0.0, "y": 0.0, "r": R, "target": True}]
    goal = rng.randint(3, 30)
    for _ in range(goal * 30):
        if len(tanks) >= goal:
            break
        r = R * rng.choice([0.3, 0.45, 0.6, 0.75, 1.0, 1.2]) * rng.uniform(0.95, 1.05)
        x, y = rng.uniform(-A, A), rng.uniform(-A, A)
        if all(math.hypot(x - t["x"], y - t["y"]) > t["r"] + r + s_min for t in tanks):
            tanks.append({"x": x, "y": y, "r": r, "target": False})
    L = {"tanks": tanks, "dikes": [], "roads": [], "sites": []}
    for _ in range(rng.randint(1, 3)):  # access roads that thread between the tanks
        w = rng.uniform(5.0, 9.0)
        if rng.random() < 0.5:
            y = rng.uniform(-A, A); a, b = (-A - 20, y), (A + 20, y)
        else:
            x = rng.uniform(-A, A); a, b = (x, -A - 20), (x, A + 20)
        if all(_point_seg((t["x"], t["y"]), a, b) > t["r"] + w / 2 + 2 for t in tanks):
            L["roads"].append((a, b, w))
    pad = rng.uniform(8.0, 25.0)
    L["sites"].append(outline_rounded_rect(-A - pad, -A - pad, A + pad, A + pad, rng.uniform(0, 25), rng, wobble=4.0))
    L["per_tank_dikes"] = True
    return L


def gen_linear(R, rng, extent):
    """One row of tanks, straight or following a gentle curve (coastal and pipeline terminals)."""
    k = rng.randint(2, 8)
    s = _spacing(R, rng)
    pitch = 2 * R + s
    bend = rng.uniform(-1, 1) * 0.0025
    tanks = []
    for i in range(k):
        x = i * pitch
        r = R if rng.random() < 0.7 else R * rng.uniform(0.6, 1.0)
        tanks.append({"x": x, "y": bend * (x - k * pitch / 2) ** 2, "r": r, "target": False})
    t = rng.randrange(k)
    tanks[t]["r"] = R
    tanks[t]["target"] = True
    L = {"tanks": tanks, "dikes": [], "roads": [], "sites": []}
    m = s / 2 + rng.uniform(2.0, 6.0)
    if rng.random() < 0.5 and abs(bend) < 0.001:
        L["dikes"].append(outline_rounded_rect(-R - m, -R - m, (k - 1) * pitch + R + m, R + m, rng.uniform(0, 8), rng, wobble=1.0))
    else:
        L["per_tank_dikes"] = True
    w = rng.uniform(6.0, 10.0)
    yr = rng.choice([-1, 1]) * (R + m + w / 2 + 3)
    L["roads"].append(((-pitch, yr), (k * pitch, yr), w))
    pad = rng.uniform(6.0, 20.0)
    ys = [tt["y"] for tt in tanks]
    L["sites"].append(outline_rounded_rect(-R - m - pad, min(ys + [yr]) - R - m - pad, (k - 1) * pitch + R + m + pad,
                                           max(ys + [yr]) + R + m + pad, rng.uniform(0, 15), rng, wobble=2.0))
    return L


def gen_isolated(R, rng, extent):
    """One to three tanks on their own (pump stations, small depots, field storage)."""
    k = rng.choice([1, 1, 2, 2, 3])
    tanks = [{"x": 0.0, "y": 0.0, "r": R, "target": True}]
    for _ in range(200):
        if len(tanks) >= k:
            break
        r = R * rng.uniform(0.5, 1.0)
        a, d = rng.uniform(0, 2 * math.pi), rng.uniform(2.2, 5.0) * R + 10
        x, y = d * math.cos(a), d * math.sin(a)
        if all(math.hypot(x - t["x"], y - t["y"]) > t["r"] + r + 8 for t in tanks):
            tanks.append({"x": x, "y": y, "r": r, "target": False})
    L = {"tanks": tanks, "dikes": [], "roads": [], "sites": [], "per_tank_dikes": True}
    a = rng.uniform(0, math.pi)
    d = R * rng.uniform(1.6, 3.0) + 8
    p = (d * math.cos(a + math.pi / 2), d * math.sin(a + math.pi / 2))
    L["roads"].append(((p[0] - 400 * math.cos(a), p[1] - 400 * math.sin(a)),
                       (p[0] + 400 * math.cos(a), p[1] + 400 * math.sin(a)), rng.uniform(4.0, 8.0)))
    xs, ys = [t["x"] for t in tanks], [t["y"] for t in tanks]
    pad = R + rng.uniform(8.0, 20.0)
    L["sites"].append(outline_rounded_rect(min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad,
                                           rng.uniform(0, 10), rng, wobble=2.0))
    return L


def gen_clusters(R, rng, extent):
    """Several separate tank farms at different angles (large hubs owned by several operators)."""
    L = gen_dike_rows(R, rng, extent * 0.5, staggered=rng.random() < 0.3, max_dikes=(3, 2))

    def radius_of(layout):
        return max(math.hypot(t["x"], t["y"]) + t["r"] for t in layout["tanks"]) + 30.0

    cx = sum(t["x"] for t in L["tanks"]) / len(L["tanks"])
    cy = sum(t["y"] for t in L["tanks"]) / len(L["tanks"])
    L = _transform_layout(L, 0.0, (-cx, -cy))
    placed = [(0.0, 0.0, radius_of(L))]
    for _ in range(rng.randint(1, 3)):
        sub = gen_dike_rows(R * rng.choice([0.5, 0.7, 0.85, 1.0, 1.2]), rng, extent * 0.4,
                            staggered=rng.random() < 0.3, with_target=False, max_dikes=(3, 2))
        sx = sum(t["x"] for t in sub["tanks"]) / len(sub["tanks"])
        sy = sum(t["y"] for t in sub["tanks"]) / len(sub["tanks"])
        sub = _transform_layout(sub, 0.0, (-sx, -sy))
        rs = radius_of(sub)
        for _ in range(30):
            a, d = rng.uniform(0, 2 * math.pi), rng.uniform(0, 150.0)
            ox, oy = (placed[0][2] + rs + d) * math.cos(a), (placed[0][2] + rs + d) * math.sin(a)
            if all(math.hypot(ox - px, oy - py) > pr + rs + 20 for px, py, pr in placed):
                sub = _transform_layout(sub, rng.uniform(-0.6, 0.6), (ox, oy))
                for key in ("tanks", "dikes", "roads", "sites"):
                    L[key] += sub[key]
                placed.append((ox, oy, rs))
                break
    return L


LAYOUTS = {"dike_rows": (lambda R, rng, e: gen_dike_rows(R, rng, e), 0.28),
           "staggered_rows": (lambda R, rng, e: gen_dike_rows(R, rng, e, staggered=True), 0.12),
           "clusters": (gen_clusters, 0.15),
           "refinery_mixed": (gen_refinery, 0.17),
           "single_row": (gen_linear, 0.13),
           "isolated": (gen_isolated, 0.15)}


def _point_seg(p, a, b):
    ab = (b[0] - a[0], b[1] - a[1])
    L2 = ab[0] ** 2 + ab[1] ** 2
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / L2))
    return math.hypot(p[0] - a[0] - t * ab[0], p[1] - a[1] - t * ab[1])


def _inside(p, poly):
    x, y = p
    inside = False
    for i in range(len(poly)):
        (x0, y0), (x1, y1) = poly[i - 1], poly[i]
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def add_polygon_patch(name, pts, z, mat):
    """Flat filled area (facility site) slightly above the ground."""
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([(x, y, z) for x, y in pts], [], [list(range(len(pts)))])
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)


def add_road(name, a, b, width, z, mat):
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, z))
    road = bpy.context.active_object
    road.name = name
    road.scale = (math.hypot(b[0] - a[0], b[1] - a[1]), width, 1.0)
    road.rotation_euler = (0.0, 0.0, math.atan2(b[1] - a[1], b[0] - a[0]))
    road.data.materials.append(mat)


def build_tank_farm(args, ground_color, rng, frame_center, frame_size, vegetation, greenness):
    """
    The tank farm around the labelled tank, from one of several real layout types (see LAYOUTS),
    rotated to a random angle. Neighbours whose shadow or off-nadir lean would touch the labelled tank
    or its shadow are left out. Trees grow outside the facility sites.
    """
    R, H = args.radius, args.height
    nh_range = tuple(float(v) for v in args.neighbour_height_range.split(",")) if args.neighbour_height_range else None
    reach = 0.75 * frame_size + 4 * R + 40.0
    names = list(LAYOUTS)
    layout = rng.choices(names, weights=[LAYOUTS[n][1] for n in names])[0]
    if args.layout in LAYOUTS:
        layout = args.layout
    L = LAYOUTS[layout][0](R, rng, 2.0 * reach)
    tgt = next(t for t in L["tanks"] if t["target"])
    L = _transform_layout(L, 0.0, (-tgt["x"], -tgt["y"]))
    site_angle = rng.uniform(0.0, 2.0 * math.pi)   # real sites are not aligned with north
    L = _transform_layout(L, site_angle)
    per_tank = L.get("per_tank_dikes", False)

    wall_h, wall_w = rng.uniform(0.8, 2.2), rng.uniform(2.5, 4.5)
    if per_tank:
        dike_style = rng.choices(["none", "ring", "square"], weights=[0.4, 0.3, 0.3])[0]
    else:
        dike_style = "group" if rng.random() < 0.85 else "none"
    wall_mat = ground_material(scale_rgb(ground_color, rng.uniform(0.85, 1.0)), rng,
                               min(1.0, vegetation * 1.4), greenness, name="BundMat")
    site_rgb = tuple(c * 0.6 + g * 0.4 for c, g in zip(ground_color, jitter((0.62, 0.61, 0.58), rng, 0.15)))
    site_mat = ground_material(site_rgb, rng, vegetation * rng.uniform(0.0, 0.5), greenness * 0.6, name="SiteMat")
    pad_mat = flat_material("PadMat", jitter(scale_rgb(ground_color, rng.uniform(0.7, 1.15)), rng, 0.08), 0.9)
    paved = rng.random() < 0.5
    road_mat = flat_material("RoadMat", jitter((0.22, 0.22, 0.23) if paved else scale_rgb(ground_color, 1.12), rng, 0.08), 0.9)
    pipe_mat = flat_material("PipeMat", jitter(rng.choice([(0.70, 0.70, 0.70), (0.45, 0.42, 0.40), (0.82, 0.82, 0.80)]), rng, 0.08), 0.5)
    use_pads = rng.random() < 0.6
    fx, fy = frame_center

    for k, site in enumerate(L["sites"]):
        add_polygon_patch(f"Site{k}", site, 0.008, site_mat)

    # Tanks: nearest to the image centre first, capped, kept clear of the labelled tank's shadow and lean
    target_fp = _footprint((0.0, 0.0), H, args.sun_elevation, args.sun_azimuth, args.camera_elevation, args.camera_azimuth)
    built = [((0.0, 0.0), R)]
    neighbours = []
    others = sorted((t for t in L["tanks"] if not t["target"]), key=lambda t: math.hypot(t["x"] - fx, t["y"] - fy))
    # --tank-density thins the layout with its own random stream, so the rest of the scene is unchanged
    drng = random.Random(f"tank-density-{args.seed}-{len(others)}")
    for t in others:
        if len(neighbours) >= min(args.max_tanks, 250):
            break  # instance ids must fit in one 8-bit mask channel
        c, rn = (t["x"], t["y"]), t["r"]
        if math.hypot(c[0] - fx, c[1] - fy) > reach:
            continue
        if args.tank_density >= 0 and drng.random() >= args.tank_density:
            continue
        if nh_range:
            hn = rng.uniform(*nh_range)
        else:
            hn = H * rng.uniform(0.85, 1.05) * (rn / R) ** 0.35
        fp = _footprint(c, hn, args.sun_elevation, args.sun_azimuth, args.camera_elevation, args.camera_azimuth)
        if min(_seg_dist(*a, *b) for a in target_fp for b in fp) < R + rn + 1.5:
            continue
        kind = "fixed_cone" if rng.random() < (0.35 if rn < 0.6 * R else 0.15) else "floating"
        neighbours.append(create_oil_tank(rn, hn, rng.uniform(0.05, 0.95), rng=rng, center=c,
                                          name=f"N{len(neighbours)}", kind=kind, tank_id=len(neighbours) + 2))
        built.append((c, rn))

    if use_pads:
        for k, ((cx, cy), r) in enumerate(built):
            add_pad(f"Pad{k}", cx, cy, r + rng.uniform(1.0, 3.0), pad_mat)

    # Dikes
    dike_outlines = []
    if dike_style == "group":
        dike_outlines = L["dikes"]
    elif dike_style in ("ring", "square"):
        for (cx, cy), r in built:
            gap = min([math.hypot(cx - ox, cy - oy) - r - orr for (ox, oy), orr in built if (ox, oy) != (cx, cy)] + [60.0])
            room = min(rng.uniform(4.0, 12.0), gap / 2 - wall_w / 2)
            if room < 2.5:
                continue
            if dike_style == "ring":
                dike_outlines.append(outline_circle(cx, cy, r + room))
            else:
                h = r + room
                ang = site_angle + (rng.uniform(-0.3, 0.3) if rng.random() < 0.2 else 0.0)
                sq = outline_rounded_rect(-h, -h, h, h, rng.uniform(0, 4), rng, wobble=0.5)
                dike_outlines.append([_rot(p, ang, (cx, cy)) for p in sq])
    for k, d in enumerate(dike_outlines):
        if min(math.hypot(x - fx, y - fy) for x, y in d) < reach:
            add_wall_loop(f"Bund{k}", d, wall_h * rng.uniform(0.9, 1.1), wall_w, wall_mat)

    # Roads (each at its own height so crossings do not z-fight)
    for k, (a, b, w) in enumerate(L["roads"]):
        add_road(f"Road{k}", a, b, w, 0.03 + 0.004 * (k % 8), road_mat)

    # Pipes along the road nearest the labelled tank, and a branch to it if the way is clear
    n_pipes = 0
    if L["roads"]:
        a, b, w = min(L["roads"], key=lambda rd: _point_seg((0.0, 0.0), rd[0], rd[1]))
        L2 = math.hypot(b[0] - a[0], b[1] - a[1]) or 1.0
        ux, uy = (b[0] - a[0]) / L2, (b[1] - a[1]) / L2
        nx_, ny_ = -uy, ux
        if (0.0 - a[0]) * nx_ + (0.0 - a[1]) * ny_ < 0:
            nx_, ny_ = -nx_, -ny_   # normal pointing towards the labelled tank
        pipe_r = rng.uniform(0.25, 0.45)
        n_pipes = rng.randint(1, 4)
        for k in range(n_pipes):
            off = w / 2 + 0.6 + k * (2 * pipe_r + 0.3)
            add_pipe(f"PipeRun{k}", (a[0] + nx_ * off, a[1] + ny_ * off, pipe_r + 0.3),
                     (b[0] + nx_ * off, b[1] + ny_ * off, pipe_r + 0.3), pipe_r, pipe_mat)
        t = (0.0 - a[0]) * ux + (0.0 - a[1]) * uy
        foot = (a[0] + ux * t + nx_ * (w / 2 + 0.6), a[1] + uy * t + ny_ * (w / 2 + 0.6))
        dist = math.hypot(*foot)
        if dist > R + 1:
            start = (foot[0] / dist * R, foot[1] / dist * R)
            if all(_point_seg(c, start, foot) > r + 1.0 for c, r in built[1:]):
                add_pipe("PipeBranch", (*start, 0.6), (*foot, 0.6), pipe_r * 0.8, pipe_mat)

    # Vegetation: outside the facility sites, off the roads, away from tanks
    def is_free(x, y, crown, h):
        for a, b, w in L["roads"]:
            if _point_seg((x, y), a, b) < w / 2 + crown:
                return False
        for site in L["sites"]:
            if _inside((x, y), site):
                return False
        for d in dike_outlines:
            if _inside((x, y), d):
                return False
        for (cx, cy), r in built:
            if math.hypot(x - cx, y - cy) < r + 6.0 + crown:
                return False
        fp = _footprint((x, y), h, args.sun_elevation, args.sun_azimuth, args.camera_elevation, args.camera_azimuth)
        return min(_seg_dist(*a, *b) for a in target_fp for b in fp) > R + crown + 1.5

    spots = []
    for _ in range(int(4 + 24 * vegetation)):
        big = rng.random() < 0.6
        spots.append((fx + rng.uniform(-reach, reach), fy + rng.uniform(-reach, reach),
                      rng.uniform(4.0, 16.0) if big else rng.uniform(3.0, 10.0), big))
    for a, b, w in L["roads"]:
        if rng.random() < vegetation * 0.8:
            L2 = math.hypot(b[0] - a[0], b[1] - a[1]) or 1.0
            nx_, ny_ = -(b[1] - a[1]) / L2, (b[0] - a[0]) / L2
            side = rng.choice([-1, 1]) * (w / 2 + 4.0)
            for k in range(0, int(L2), 7):
                f = k / L2
                spots.append((a[0] + (b[0] - a[0]) * f + nx_ * side, a[1] + (b[1] - a[1]) * f + ny_ * side, 1.0, True))
    trees = add_trees(rng, spots, vegetation, greenness, is_free, args.sun_elevation)

    scene_info = {"layout": layout, "site_rotation_degrees": round(math.degrees(site_angle) % 360.0, 2), "tanks_in_layout": len(L["tanks"]), "tanks_built": len(built),
                  "dike_style": dike_style, "bund_height_m": round(wall_h, 2) if dike_outlines else None,
                  "road_surface": "paved" if paved else "unpaved", "roads": len(L["roads"]), "pipes": n_pipes,
                  "vegetation": round(vegetation, 3), "greenness": round(greenness, 3), "trees_and_shrubs": trees,
                  "neighbour_density": round(args.tank_density, 3) if args.tank_density >= 0 else None}
    return scene_info, neighbours


def setup_lighting(elevation_deg, azimuth_deg, atmosphere="CLEAR", sun_angle_deg=-1.0):
    # Convert angles to radians
    elev = math.radians(elevation_deg)
    az = math.radians(azimuth_deg)
    
    # Calculate Sun direction vector: Y is North, X is East, Z is Up
    # In Blender, light direction is along local -Z of the light object.
    # Therefore, we construct a vector pointing from origin to the Sun:
    sun_dir = mathutils.Vector((
        math.cos(elev) * math.sin(az),
        math.cos(elev) * math.cos(az),
        math.sin(elev)
    ))
    
    # Add Sun light source
    bpy.ops.object.light_add(type='SUN', radius=1.0, location=(0, 0, 50))
    sun = bpy.context.active_object
    sun.name = "Sun"
    
    # Every atmosphere keeps a direct sun, so the shadows that carry the fill information stay in
    # the image. The sun is a 0.53 deg disc (its real apparent size) and gets wider only under thin
    # cloud. Haze and cloud mainly add sky light (brighter, lower-contrast shadows) and a veil of
    # scattered light over the whole image, applied after rendering (see ATMOSPHERE_VEIL).
    # A fully overcast sky would remove the shadows completely; such scenes are normally filtered
    # out of satellite archives and cannot be labelled by shadow, so they are not rendered.
    atmosphere = "THIN_CLOUD" if atmosphere == "OVERCAST" else atmosphere
    if atmosphere == "HAZE":
        sun.data.energy = 4.5
        sun.data.angle = math.radians(0.53)
        sky_color = (0.52, 0.62, 0.82, 1.0)  # grey-blue scattered skylight
        sky_strength = 0.5
    elif atmosphere == "THIN_CLOUD":
        sun.data.energy = 3.5               # sun dimmed by thin cloud, but still direct
        sun.data.angle = math.radians(1.0)  # slightly softer shadow edges
        sky_color = (0.72, 0.76, 0.86, 1.0)  # bright white-grey cloud light, still slightly blue
        sky_strength = 0.6
    else:  # CLEAR
        sun.data.energy = 5.5               # strong direct sunlight
        sun.data.angle = math.radians(0.53) # real apparent size of the sun
        # Clean blue sky. Calibrated on real NAIP tank shadows (73 tanks): shadows are lit by blue
        # skylight only, so their blue/red ratio is ~1.35x that of lit ground (B/R 0.89 -> 1.20)
        sky_color = (0.42, 0.58, 0.95, 1.0)
        sky_strength = 0.44

    # Direct sunlight at the ground is slightly warm (~5500 K); with the blue sky fill this gives the blue
    # shadows seen in real imagery
    sun.data.color = (1.0, 0.96, 0.90)
    if sun_angle_deg >= 0:
        sun.data.angle = math.radians(sun_angle_deg)

    # Point sun along the negative direction vector
    sun.rotation_mode = 'QUATERNION'
    sun.rotation_quaternion = (-sun_dir).to_track_quat('-Z', 'Y')
    
    # Configure World ambient lighting
    if bpy.context.scene.world:
        bpy.context.scene.world.use_nodes = True
        nodes = bpy.context.scene.world.node_tree.nodes
        background = nodes.get("Background")
        if background:
            background.inputs['Color'].default_value = sky_color
            background.inputs['Strength'].default_value = sky_strength

# Light scattered by the atmosphere between the ground and the satellite (path radiance): a veil of
# this colour is blended over the rendered image with this weight.
ATMOSPHERE_VEIL = {
    "CLEAR": None,
    "HAZE": ((0.74, 0.77, 0.82), 0.12),
    "THIN_CLOUD": ((0.86, 0.86, 0.88), 0.20),
}

def apply_atmosphere_veil(path, atmosphere):
    atmosphere = "THIN_CLOUD" if atmosphere == "OVERCAST" else atmosphere
    veil = ATMOSPHERE_VEIL.get(atmosphere)
    if not veil:
        return
    import numpy as np
    color, weight = veil
    img = bpy.data.images.load(path)
    px = np.empty(len(img.pixels), dtype=np.float32)
    img.pixels.foreach_get(px)
    px = px.reshape(-1, 4)
    px[:, :3] = px[:, :3] * (1.0 - weight) + np.array(color, dtype=np.float32) * weight
    img.pixels.foreach_set(px.ravel())
    img.filepath_raw = path
    img.file_format = 'PNG'
    img.save()
    bpy.data.images.remove(img)

def setup_camera(elevation_deg, azimuth_deg, tank_height, tank_radius, sun_elevation_deg, sun_azimuth_deg,
                 zoom=1.0, shift_x=0.0, shift_y=0.0, max_mpp=0.0, gsd=-1.0, mpp=0.0):
    """
    zoom > 1 widens the view beyond the tight frame around the tank and its shadow (more surroundings,
    more tanks, fewer pixels per tank), capped so metres-per-pixel stays <= max_mpp when max_mpp > 0.
    shift_x / shift_y in [-1, 1] move the labelled tank off centre, as far as the frame allows while
    the tank and its whole shadow stay inside. Returns (ortho_scale, ground point at the image centre).
    gsd > 0 fixes the metres per pixel instead: the image size (RES_X, RES_Y) then follows the framed
    extent. At zoom 1 and no shift this is the v3 chip framing (src/v3/physics.Camera.for_scene).
    mpp > 0 (with gsd <= 0) fixes the metres per pixel of the 800 x 600 frame instead of zoom (NAIP scale);
    if the tank and its shadow would not fit, the frame grows to the tight frame.
    """
    # Convert angles to radians
    elev = math.radians(elevation_deg)
    az = math.radians(azimuth_deg)
    sun_elev = math.radians(sun_elevation_deg)
    sun_az = math.radians(sun_azimuth_deg)
    
    # Calculate external shadow length on the ground
    shadow_length = tank_height / math.tan(sun_elev) if math.tan(sun_elev) > 0 else 0
    
    # Calculate shadow direction (opposite to sun direction)
    # Shadow extends from center along vector: dx = -shadow_length * sin(sun_az), dy = -shadow_length * cos(sun_az)
    # The centroid of the tank + its shadow is offset by half the shadow extension
    offset_x = -0.5 * shadow_length * math.sin(sun_az)
    offset_y = -0.5 * shadow_length * math.cos(sun_az)
    
    # Set camera distance
    distance = 250.0
    
    # Calculate camera position relative to the offset target center
    x = offset_x + distance * math.cos(elev) * math.sin(az)
    y = offset_y + distance * math.cos(elev) * math.cos(az)
    z = distance * math.sin(elev) + (tank_height / 2.0)
    
    # Add camera
    bpy.ops.object.camera_add(location=(x, y, z))
    camera = bpy.context.active_object
    camera.name = "SatelliteCamera"
    
    # Configure camera properties
    camera.data.type = 'ORTHO' # Orthographic projection is standard for satellite imagery
    
    # Point the camera at the target and roll it so the image is NORTH-UP: image-up is North
    # projected onto the image plane and image-right is East. to_track_quat('-Z', 'Y') rolled the
    # image with the camera azimuth (south-up at azimuth 0) and left the roll undefined at nadir.
    cam_target = mathutils.Vector((offset_x, offset_y, tank_height / 2.0))
    forward = (cam_target - camera.location).normalized()
    north = mathutils.Vector((0.0, 1.0, 0.0))
    up = north - north.dot(forward) * forward
    if up.length < 1e-6:  # only possible for a horizontal camera looking due North/South
        up = mathutils.Vector((0.0, 0.0, 1.0))
    up.normalize()
    back = -forward
    right = up.cross(back)
    camera.rotation_mode = 'QUATERNION'
    camera.rotation_quaternion = mathutils.Matrix((right, up, back)).transposed().to_quaternion()

    # Size the frame from the projected outline of the tank base, tank rim and ground shadow.
    # A tilted camera shifts the rim sideways, and the frame is only 3/4 as tall as it is wide,
    # so measure the extent along both image axes instead of using the ground span alone.
    rotation = camera.rotation_quaternion.to_matrix()
    cam_right, cam_up = rotation.col[0], rotation.col[1]
    shadow_dx = -shadow_length * math.sin(sun_az)
    shadow_dy = -shadow_length * math.cos(sun_az)
    half_w = half_h = 0.0
    for i in range(128):
        a = 2.0 * math.pi * i / 128
        x, y = tank_radius * math.cos(a), tank_radius * math.sin(a)
        for p in ((x, y, 0.0), (x, y, tank_height), (x + shadow_dx, y + shadow_dy, 0.0)):
            rel = mathutils.Vector(p) - cam_target
            half_w = max(half_w, abs(rel.dot(cam_right)))
            half_h = max(half_h, abs(rel.dot(cam_up)))

    global RES_X, RES_Y
    if gsd > 0:
        # Fixed ground sample distance: the chip grows to hold the tank, its shadow and the margin
        camera.data.sensor_fit = 'HORIZONTAL'
        z = max(1.0, zoom)
        RES_X = max(64, int(math.ceil(2.0 * half_w * FRAME_MARGIN * z / gsd)))
        RES_Y = max(64, int(math.ceil(2.0 * half_h * FRAME_MARGIN * z / gsd)))
        ortho = RES_X * gsd
    else:
        # With the default sensor fit, ortho_scale spans the wider image axis (the width here)
        aspect = RES_X / RES_Y
        tight = 2.0 * max(half_w, half_h * aspect) * FRAME_MARGIN
        if mpp > 0:
            ortho = max(tight, mpp * RES_X)
        else:
            ortho = tight * max(1.0, zoom)
            if max_mpp > 0:
                ortho = min(ortho, max(tight, max_mpp * RES_X))
    camera.data.ortho_scale = ortho
    aspect = RES_X / RES_Y

    # Off-centre framing: slide the camera within the slack around the content
    slack_x = max(0.0, ortho / 2.0 - half_w * 1.05)
    slack_y = max(0.0, ortho / (2.0 * aspect) - half_h * 1.05)
    u = max(-1.0, min(1.0, shift_x)) * slack_x
    v = max(-1.0, min(1.0, shift_y)) * slack_y
    camera.location = camera.location + cam_right * u + cam_up * v
    view_center = cam_target + cam_right * u + cam_up * v

    # Set as active camera for the scene
    bpy.context.scene.camera = camera
    return camera.data.ortho_scale, (view_center.x, view_center.y)

def render_mask(path):
    """
    Segmentation mask through the same camera, pixel-aligned with the image:
      red   = tank instance id (1 = the labelled tank, 2.. = neighbour_tanks[*].instance_id, 0 = not a tank)
      green = part (PART_IDS: 1 shell outside, 2 shell inside, 3 floating roof, 4 fixed roof, 5 ladder/platform;
              6 floating roof in a tank's shadow, 7 ground in a tank's shadow; see _paint_shadow_parts)
      blue  = 0
    Every material becomes a flat emission of those values; one un-filtered sample per pixel and the Raw
    view transform keep the values exact (no anti-aliasing blends, no colour management).
    Call this after the image is rendered: it replaces the scene's materials.
    """
    scene = bpy.context.scene
    # Tank bodies (shell, floating roof, cone, floor) cast the labelled shadows; ladders do not
    casters = {obj.name for obj in scene.objects if obj.type == 'MESH' and obj.color[0] > 0 and any(
        slot.material is not None and int(slot.material.get("part", 0)) in (1, 2, 3, 4) for slot in obj.material_slots)}
    for mat in bpy.data.materials:
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()
        out = nt.nodes.new('ShaderNodeOutputMaterial')
        em = nt.nodes.new('ShaderNodeEmission')
        em.inputs['Strength'].default_value = 1.0
        part = int(mat.get("part", 0))
        if part:
            info = nt.nodes.new('ShaderNodeObjectInfo')
            enc = nt.nodes.new('ShaderNodeVectorMath')
            enc.operation = 'MULTIPLY_ADD'
            nt.links.new(info.outputs['Color'], enc.inputs[0])
            enc.inputs[1].default_value = (1.0, 0.0, 0.0)
            enc.inputs[2].default_value = (0.0, part / 255.0, 0.0)
            nt.links.new(enc.outputs['Vector'], em.inputs['Color'])
        else:
            em.inputs['Color'].default_value = (0.0, 0.0, 0.0, 1.0)
        nt.links.new(em.outputs['Emission'], out.inputs['Surface'])
    for obj in scene.objects:
        if obj.type == 'LIGHT':
            obj.hide_render = True
    if scene.world and scene.world.node_tree:
        bg = scene.world.node_tree.nodes.get("Background")
        if bg:
            bg.inputs['Strength'].default_value = 0.0

    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    scene.cycles.max_bounces = 0
    scene.cycles.filter_width = 0.01
    try:
        scene.cycles.use_adaptive_sampling = False
    except AttributeError:
        pass
    vs = scene.view_settings
    try:
        vs.view_transform = 'Raw'
    except TypeError:
        vs.view_transform = 'Standard'
    vs.look = 'None'
    vs.exposure = 0.0
    vs.gamma = 1.0
    scene.render.dither_intensity = 0.0
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.color_depth = '8'
    scene.render.filepath = path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    bpy.ops.render.render(write_still=True)
    _paint_shadow_parts(path, casters)


def _direct_sun_pass(path, casters, cast):
    """White diffuse scene lit by the sun only (angle 0, no sky, no bounces). With cast=False no object
    casts a shadow, so a pixel is dark only where its own surface faces away from the sun."""
    white = bpy.data.materials.get("ShadowPassWhite")
    if white is None:
        white = bpy.data.materials.new("ShadowPassWhite")
        white.use_nodes = True
        nodes, links = white.node_tree.nodes, white.node_tree.links
        nodes.clear()
        dif = nodes.new('ShaderNodeBsdfDiffuse')
        dif.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
        out = nodes.new('ShaderNodeOutputMaterial')
        links.new(dif.outputs['BSDF'], out.inputs['Surface'])
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            for slot in obj.material_slots:
                slot.material = white
            obj.visible_shadow = cast and obj.name in casters
        elif obj.type == 'LIGHT':
            obj.hide_render = False
            obj.data.angle = 0.0
            obj.data.energy = 10.0
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def _load_pixels(path):
    import numpy as np
    img = bpy.data.images.load(path)
    w, h = img.size
    px = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    return img, px.reshape(h, w, 4)


def _paint_shadow_parts(mask_path, casters):
    """
    Add the shadow classes to the mask's green channel: INNER_SHADOW_PART where a floating roof lies in a
    shadow cast by a tank (its own rim, mostly), GROUND_SHADOW_PART where a non-tank surface (ground,
    road, bund, pad) lies in a tank's shadow. Tank-cast shadow = dark in a pass where only tank shells,
    roofs and cones cast shadows, but lit in a pass where nothing casts. Shadows of trees, pipes and
    ladders are not labelled.
    """
    import numpy as np
    base = os.path.splitext(mask_path)[0]
    lit_tanks, lit_none = base + ".__lit_tanks.png", base + ".__lit_none.png"
    _direct_sun_pass(lit_tanks, casters, cast=True)
    _direct_sun_pass(lit_none, casters, cast=False)
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.visible_shadow = True
    img, mask = _load_pixels(mask_path)
    _, a = _load_pixels(lit_tanks)
    _, b = _load_pixels(lit_none)
    red = np.rint(mask[..., 0] * 255.0)
    green = np.rint(mask[..., 1] * 255.0)
    tank_shadow = (a[..., 0] < 0.02) & (b[..., 0] > 0.02)
    green[(green == PART_IDS["floating_roof"]) & tank_shadow] = INNER_SHADOW_PART
    green[(red == 0) & tank_shadow] = GROUND_SHADOW_PART
    mask[..., 1] = green / 255.0
    img.pixels.foreach_set(mask.ravel())
    img.filepath_raw = mask_path
    img.file_format = 'PNG'
    img.save()
    for p in (lit_tanks, lit_none):
        os.remove(p)


# ---------------------------------------------------------------------------------------------
# v3 exports: exact camera / scene geometry, and mask passes rendered from the Blender scene
# ---------------------------------------------------------------------------------------------
# Object-ID codes in the ID pass. Mask classes are derived from these passes.
ID_CODES = {"ground": 0, "wall_exterior": 1, "wall_top": 2, "roof": 3, "neighbour": 10, "wall_interior": 13,
            "bund": 21, "pipe": 22, "ladder": 23, "tree": 24, "road_pad": 25, "edge": 26}


def export_geometry(args, target_info, scene_info, ortho_scale):
    cam = bpy.context.scene.camera
    m = cam.matrix_world
    data = {
        "blender_version": bpy.app.version_string,
        "camera": {"matrix_world": [list(r) for r in m], "ortho_scale": ortho_scale,
                   "res_x": RES_X, "res_y": RES_Y, "sensor_fit": cam.data.sensor_fit, "type": cam.data.type},
        "target": dict(target_info, radius_m=args.radius, height_m=args.height,
                       wall_thickness_m=WALL_THICKNESS, roof_thickness_m=ROOF_THICKNESS,
                       seal_gap_m=args.seal_gap, z_floor_m=args.z_floor,
                       roof_tilt_deg=args.roof_tilt, roof_tilt_azimuth_deg=args.roof_tilt_azimuth,
                       terrain_slope_deg=args.terrain_slope, terrain_slope_azimuth_deg=args.terrain_slope_azimuth),
        "sun": {"elevation_deg": args.sun_elevation, "azimuth_deg": args.sun_azimuth,
                "angle_deg": math.degrees(bpy.data.objects["Sun"].data.angle),
                "energy": bpy.data.objects["Sun"].data.energy},
        "render": {"engine": bpy.context.scene.render.engine,
                   "samples": getattr(bpy.context.scene.cycles, "samples", None)},
        "scene": scene_info,
        "objects": sorted(o.name for o in bpy.data.objects),
        "materials": sorted(mat.name for mat in bpy.data.materials),
    }
    with open(args.export_json, "w") as f:
        json.dump(data, f, indent=1)


def _id_code(obj_name, slot):
    n = obj_name
    if n == "Ground":
        return ID_CODES["ground"]
    if n.startswith("TargetShell"):
        return (ID_CODES["wall_exterior"], ID_CODES["wall_interior"], ID_CODES["wall_top"])[min(slot, 2)]
    if n.startswith("TargetFloatingRoof"):
        return ID_CODES["roof"]
    if n.startswith("TargetFloor"):
        return ID_CODES["wall_interior"]
    if n.startswith("TargetEdge"):
        return ID_CODES["edge"]
    if n.startswith("Target") and ("Ladder" in n or "Platform" in n):
        return ID_CODES["ladder"]
    if len(n) > 1 and n[0] == "N" and n[1].isdigit():   # neighbouring tanks: N0Shell, N12Cone, ...
        return ID_CODES["neighbour"]
    if n.startswith("Bund"):
        return ID_CODES["bund"]
    if n.startswith("Pipe"):
        return ID_CODES["pipe"]
    if n.startswith(("Road", "Pad", "Site")):
        return ID_CODES["road_pad"]
    return ID_CODES["tree"]


def _emission(name, value):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    em = nodes.new('ShaderNodeEmission')
    em.inputs['Color'].default_value = (value, value, value, 1.0)
    em.inputs['Strength'].default_value = 1.0
    out = nodes.new('ShaderNodeOutputMaterial')
    links.new(em.outputs['Emission'], out.inputs['Surface'])
    return mat


def _render_to(path, samples=1):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = samples
    scene.cycles.use_denoising = False
    scene.render.filter_size = 0.01  # one ray per pixel centre: no anti-aliased mixed labels
    scene.render.dither_intensity = 0.0  # 8-bit dithering would add +/-1 noise to the codes
    scene.view_settings.view_transform = 'Raw'
    scene.view_settings.look = 'None'
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_depth = '8'
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def render_mask_passes(masks_dir, sun_elevation, sun_azimuth):
    """
    Two extra renders through the same camera, written to masks_dir:
    - id.png:  every surface emits its ID code (value code/255), one ray per pixel, no lighting
    - lit.png: white diffuse surfaces, direct sunlight only (no sky, no bounces, sun angle 0), so a
               pixel is 0 exactly where the surface seen is in geometric shadow
    """
    os.makedirs(masks_dir, exist_ok=True)
    world = bpy.context.scene.world
    if world and world.node_tree:
        bg = world.node_tree.nodes.get("Background")
        if bg:
            bg.inputs['Strength'].default_value = 0.0
    sun = bpy.data.objects["Sun"]
    cache = {}
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue
        n_slots = max(1, len(obj.material_slots))
        if not obj.material_slots:
            obj.data.materials.append(None)
        for slot in range(n_slots):
            code = _id_code(obj.name, slot)
            if code not in cache:
                cache[code] = _emission(f"ID{code}", code / 255.0)
            obj.material_slots[slot].material = cache[code]
    sun.hide_render = True
    _render_to(os.path.join(masks_dir, "id.png"), samples=1)

    white = bpy.data.materials.new("LitWhite")
    white.use_nodes = True
    nodes, links = white.node_tree.nodes, white.node_tree.links
    nodes.clear()
    dif = nodes.new('ShaderNodeBsdfDiffuse')
    dif.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    out = nodes.new('ShaderNodeOutputMaterial')
    links.new(dif.outputs['BSDF'], out.inputs['Surface'])
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            for slot in obj.material_slots:
                slot.material = white
    sun.hide_render = False
    sun.data.angle = 0.0
    sun.data.energy = 10.0
    scene = bpy.context.scene
    scene.cycles.max_bounces = 0
    _render_to(os.path.join(masks_dir, "lit.png"), samples=1)


def main():
    args = parse_args()
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    clear_default_scene()
    setup_render_engine(args.engine, args.samples, args.device)
    ground_color = tuple(float(c) for c in args.ground_color.split(","))
    # Same file name -> same scene, so a re-render reproduces the image exactly
    seed = args.seed if args.seed >= 0 else int.from_bytes(
        hashlib.sha256(os.path.basename(args.output).encode()).digest()[:4], "little")
    rng = random.Random(seed) if args.realism == "FULL" else None
    vegetation = greenness = 0.0
    if rng is not None:
        vegetation = args.vegetation if args.vegetation >= 0 else rng.uniform(0.05, 0.7)
        # Wet, green sites vs dry, brown ones (season and climate)
        greenness = rng.uniform(0.4, 1.0) if vegetation > 0.3 else rng.uniform(0.0, 0.5)
    args.seed = seed
    v3_roof = args.phi_r >= 0
    create_environment(ground_color, args.radius, args.height, args.sun_elevation, rng=rng,
                       vegetation=vegetation, greenness=greenness,
                       slope_deg=args.terrain_slope, slope_azimuth_deg=args.terrain_slope_azimuth,
                       # v3 scenes: no ground inside the tank (legacy scenes keep the full plane)
                       hole_radius=(args.radius - WALL_THICKNESS / 2.0) if v3_roof else 0.0)
    # On sloped terrain, or with the floor below grade, the shell continues below z_G = 0
    shell_bottom = 0.0
    if args.terrain_slope:
        shell_bottom = -(args.radius * math.tan(math.radians(args.terrain_slope)) + 0.5)
    if v3_roof:
        shell_bottom = min(shell_bottom, args.z_floor - ROOF_THICKNESS - 0.5)
    target_info = create_oil_tank(args.radius, args.height, args.fill_level, rng=rng, name="Target",
                                  phi_r=args.phi_r if v3_roof else None, z_floor=args.z_floor,
                                  seal_gap=args.seal_gap, roof_offset=(args.roof_offset_x, args.roof_offset_y),
                                  roof_tilt=(args.roof_tilt, args.roof_tilt_azimuth),
                                  edge=(args.edge_type, args.edge_height), shell_bottom=shell_bottom,
                                  floor_disc=v3_roof)
    setup_lighting(args.sun_elevation, args.sun_azimuth, args.atmosphere, args.sun_angle)
    ortho_scale, view_center = setup_camera(
        args.camera_elevation,
        args.camera_azimuth,
        args.height,
        args.radius,
        args.sun_elevation,
        args.sun_azimuth,
        zoom=args.zoom, shift_x=args.shift_x, shift_y=args.shift_y, max_mpp=args.max_mpp, gsd=args.gsd,
        mpp=args.mpp,
    )
    # Read by render.py to record the image scale in metadata.json
    print(f"CAMERA_ORTHO_SCALE={ortho_scale:.6f}")
    bpy.context.scene.camera.data.clip_end = 10000.0
    # Wide, off-centre frames can reach past the ground plane; grow its mesh (not the object scale, so
    # textures stay in metres) until it covers the whole view with room to spare
    ground = bpy.data.objects.get("Ground")
    if ground:
        half_now = max(abs(v.co.x) for v in ground.data.vertices)
        half_needed = math.hypot(*view_center) + ortho_scale * 1.5 + 50.0
        if half_needed > half_now:
            hole = (args.radius - WALL_THICKNESS / 2.0) + 1e-3 if v3_roof else -1.0
            for v in ground.data.vertices:
                if math.hypot(v.co.x, v.co.y) > hole:  # keep the hole under a v3 tank its own size
                    v.co.x *= half_needed / half_now
                    v.co.y *= half_needed / half_now

    scene_info, neighbours = {"realism": args.realism, "seed": seed}, []
    if rng is not None:
        farm, neighbours = build_tank_farm(args, ground_color, rng, view_center, ortho_scale, vegetation, greenness)
        scene_info.update(farm)
        scene_info["target_details"] = {k: v for k, v in target_info.items() if k.startswith("ladder")}

    if rng is not None:
        # Linear colours, one stop darker: AgX/Filmic squash bright sand, white shells and pale roofs
        # into the same near-white grey, while real satellite products keep them apart.
        try:
            bpy.context.scene.view_settings.view_transform = 'Standard'
            bpy.context.scene.view_settings.look = 'None'
            bpy.context.scene.view_settings.exposure = -0.75
        except TypeError:
            pass

    # Set resolution
    bpy.context.scene.render.resolution_x = RES_X
    bpy.context.scene.render.resolution_y = RES_Y
    bpy.context.scene.render.resolution_percentage = 100

    # Exact pixel positions (x right, y down, from the top-left corner) through the real camera,
    # read by render.py so downstream code never has to re-derive the projection
    bpy.context.view_layer.update()  # apply the camera rotation to matrix_world before projecting
    rox, roy = target_info.get("roof_offset_m", [0.0, 0.0])
    for name, pt in (("BASE", (0.0, 0.0, 0.0)), ("RIM", (0.0, 0.0, args.height)),
                     ("ROOF", (rox, roy, target_info["deck_z_m"]))):
        p = world_to_camera_view(bpy.context.scene, bpy.context.scene.camera, mathutils.Vector(pt))
        print(f"PIXEL_{name}={p.x * RES_X:.3f},{(1.0 - p.y) * RES_Y:.3f}")
    print(f"IMAGE_SIZE={RES_X},{RES_Y}")

    # Other tanks in the image, so a detector is not trained to treat them as background
    m_per_px = ortho_scale / RES_X
    labelled = []
    for nb in neighbours:
        cx, cy = nb["center_m"]
        px = {}
        for key, z in (("base_center_pixel", 0.0), ("rim_center_pixel", nb["height_m"])):
            p = world_to_camera_view(bpy.context.scene, bpy.context.scene.camera, mathutils.Vector((cx, cy, z)))
            px[key] = [round(p.x * RES_X, 2), round((1.0 - p.y) * RES_Y, 2)]
        r_px = nb["radius_m"] / m_per_px
        xs = [px[k][0] for k in px]
        ys = [px[k][1] for k in px]
        box = [min(xs) - r_px, min(ys) - r_px, max(xs) + r_px, max(ys) + r_px]
        if box[2] < 0 or box[3] < 0 or box[0] > RES_X or box[1] > RES_Y:
            continue  # entirely outside the image
        nb.update(px)
        nb["radius_pixels"] = round(r_px, 2)
        nb["bbox_pixels"] = [round(max(0.0, box[0]), 1), round(max(0.0, box[1]), 1),
                             round(min(RES_X, box[2]), 1), round(min(RES_Y, box[3]), 1)]
        nb["fully_in_frame"] = box[0] >= 0 and box[1] >= 0 and box[2] <= RES_X and box[3] <= RES_Y
        labelled.append(nb)
    scene_info["neighbour_tanks"] = labelled
    scene_info["tanks_in_image"] = 1 + len(labelled)
    print("SCENE_JSON=" + json.dumps(scene_info), flush=True)
    
    # Configure render output file path
    bpy.context.scene.render.filepath = args.output
    
    # Render and write to file
    bpy.ops.render.render(write_still=True)
    apply_atmosphere_veil(args.output, args.atmosphere)
    print(f"Successfully rendered scene to: {args.output}")

    if args.export_json:
        export_geometry(args, target_info, scene_info, ortho_scale)
    # Both mask renders replace the scene's materials, so they come after the image and the export
    if args.mask_output:
        render_mask(args.mask_output)
        print(f"MASK_OUTPUT={args.mask_output}")
    if args.masks_dir:
        render_mask_passes(args.masks_dir, args.sun_elevation, args.sun_azimuth)

if __name__ == "__main__":
    main()
