"""Render the same candidate-only, neutral-material evidence for both modes."""

import argparse
import hashlib
import json
import math
import runpy
import sys
from pathlib import Path


def main() -> None:
    import bpy
    from mathutils import Vector

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blend", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--views",
        help="Comma-separated fixed view indices for rendering-only diagnostics",
    )
    parser.add_argument("--save-display", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    args.output.mkdir(parents=True, exist_ok=False)
    bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()), load_ui=False)
    scene = bpy.context.scene
    for obj in list(scene.objects):
        if obj.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    framing = []
    bounds_module = runpy.run_path(str(Path(__file__).with_name("scene.py")))
    center, radius, bounds = bounds_module["scene_bounds"](
        scene, depsgraph, include_volume=False, diagnostics=framing
    )
    material = bpy.data.materials.new("VerifierNeutralOpaque")
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = (0.55, 0.55, 0.55, 1.0)
    principled.inputs["Roughness"].default_value = 0.7
    scene.view_layers[0].material_override = material
    world = bpy.data.worlds.new("VerifierNeutralWorld")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (1, 1, 1, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.5
    scene.world = world
    for index, direction in enumerate([(4, -6, 8), (-6, -3, 4), (3, 6, 6)]):
        data = bpy.data.lights.new(f"VerifierLight{index}", "AREA")
        data.energy = 500 * radius * radius
        data.size = radius * 3
        obj = bpy.data.objects.new(data.name, data)
        scene.collection.objects.link(obj)
        obj.location = center + Vector(direction).normalized() * radius * 4
        obj.rotation_euler = (center - obj.location).to_track_quat("-Z", "Y").to_euler()
    camera_data = bpy.data.cameras.new("VerifierCamera")
    camera = bpy.data.objects.new(camera_data.name, camera_data)
    scene.collection.objects.link(camera)
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = radius * 2.3
    camera_data.clip_start = max(radius * 1e-5, 1e-6)
    camera_data.clip_end = radius * 20
    scene.camera = camera
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 128
    scene.cycles.seed = 0
    scene.render.resolution_x = 768
    scene.render.resolution_y = 768
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    images = []
    views = [(angle, 20) for angle in range(0, 360, 45)] + [(0, 89.9)]
    selected = (
        set(range(1, 10))
        if args.views is None
        else {int(value) for value in args.views.split(",")}
    )
    if not selected or not selected <= set(range(1, 10)):
        raise ValueError("View indices must be between1 and9")
    for index, (azimuth, elevation) in enumerate(views, 1):
        if index not in selected:
            continue
        a, e = math.radians(azimuth), math.radians(elevation)
        direction = Vector(
            (math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e))
        )
        camera.location = center + direction * radius * 4
        camera.rotation_euler = (-direction).to_track_quat("-Z", "Y").to_euler()
        path = args.output / f"view-{index:02d}.png"
        scene.render.filepath = str(path.resolve())
        bpy.ops.render.render(write_still=True)
        images.append(
            {
                "id": f"sample-V{index}",
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "azimuth": azimuth,
                "elevation": elevation,
            }
        )
    # Fixed center crops in two opposite views; no score-dependent view selection.
    crop_sources = [images[0], images[4]] if selected == set(range(1, 10)) else []
    for index, source in enumerate(crop_sources, 1):
        image = bpy.data.images.load(
            str((args.output / source["path"]).resolve()), check_existing=False
        )
        width, height = image.size
        import numpy as np

        pixels = np.array(image.pixels[:], dtype=np.float32).reshape(height, width, 4)
        crop = pixels[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4].copy()
        zoom = bpy.data.images.new(
            f"VerifierZoom{index}",
            width=crop.shape[1],
            height=crop.shape[0],
            alpha=True,
        )
        zoom.pixels.foreach_set(crop.ravel())
        path = args.output / f"zoom-{index:02d}.png"
        zoom.filepath_raw = str(path.resolve())
        zoom.file_format = "PNG"
        zoom.save()
        images.append(
            {
                "id": f"sample-Z{index}",
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source_view": source["id"],
                "bbox": [0.25, 0.25, 0.75, 0.75],
            }
        )
    manifest = {
        "schema": "blender-overfit-render/v1",
        "images": images,
        "candidate_sha256": hashlib.sha256(args.blend.read_bytes()).hexdigest(),
        "cycles_samples": 128,
        "neutral_material": True,
        "original_candidate_modified": False,
        "framing": {
            "center": list(center),
            "radius": radius,
            "particle_systems": framing,
        },
    }
    if args.save_display:
        # Export after evidence rendering so display settings cannot affect scores.
        direction = Vector((math.cos(math.radians(20)), 0, math.sin(math.radians(20))))
        camera.location = center + direction * radius * 4
        camera.rotation_euler = (-direction).to_track_quat("-Z", "Y").to_euler()
        scene.view_layers[0].material_override = None
        scene.cycles.samples = 32
        display = args.output / "display-scene.blend"
        bpy.ops.wm.save_as_mainfile(
            filepath=str(display.resolve()), check_existing=False
        )
        manifest["display_scene"] = {
            "path": display.name,
            "source_candidate_sha256": manifest["candidate_sha256"],
            "sha256": hashlib.sha256(display.read_bytes()).hexdigest(),
        }
    (args.output / "render-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
