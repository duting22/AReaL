"""Blender-side conversion; run with Blender's --python option."""

import argparse
import json
import math
import runpy
import sys
from itertools import product
from pathlib import Path
from typing import Any


def particle_hair_bounds(instance: Any, diagnostics: list | None = None) -> list:
    """Conservatively enclose static legacy PATH hairs beyond their emitter mesh."""
    obj = instance.object
    systems = [
        modifier.particle_system
        for modifier in getattr(obj, "modifiers", [])
        if modifier.type == "PARTICLE_SYSTEM" and modifier.show_render
    ]
    active = [
        system
        for system in systems
        if system.settings.type == "HAIR"
        and system.settings.render_type == "PATH"
        and system.settings.count > 0
    ]
    if not active:
        return []
    import numpy as np

    matrix = np.asarray([list(row) for row in instance.matrix_world], dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Nonfinite particle emitter transform")
    scale = max(1.0, float(np.linalg.norm(matrix[:3, :3], ord=2)))
    original = getattr(obj, "matrix_world", None)
    if original is not None:
        original = np.asarray([list(row) for row in original], dtype=np.float64)
        relative = matrix @ np.linalg.inv(original)
        scale = max(scale, float(np.linalg.norm(relative[:3, :3], ord=2)))
    corners = np.asarray([list(corner) for corner in obj.bound_box], dtype=np.float64)
    world = corners @ matrix[:3, :3].T + matrix[:3, 3]
    result = []
    for system in active:
        settings = system.settings
        if getattr(system, "use_hair_dynamics", False) or getattr(
            system, "is_edited", False
        ):
            raise ValueError(
                "Dynamic or edited legacy hair requires evaluated path bounds"
            )
        nominal_length = float(settings.hair_length)
        unsupported = (
            "object_factor",
            "particle_factor",
            "brownian_factor",
            "drag_factor",
            "damping",
        )
        if any(float(getattr(settings, name, 0.0)) != 0.0 for name in unsupported):
            raise ValueError("Forced legacy hair requires evaluated path bounds")
        weights = getattr(settings, "effector_weights", None)
        if getattr(weights, "apply_to_hair_growing", False) or getattr(
            settings, "animation_data", None
        ):
            raise ValueError("Animated or forced hair requires evaluated path bounds")
        if any(slot is not None for slot in getattr(settings, "texture_slots", [])):
            raise ValueError("Textured hair velocity requires evaluated path bounds")
        tangent = abs(float(getattr(settings, "tangent_factor", 0.0)))
        random_velocity = abs(float(getattr(settings, "factor_random", 0.0)))
        aligned = sum(
            abs(float(value))
            for value in getattr(settings, "object_align_factor", (0.0, 0.0, 0.0))
        )
        timestep = float(getattr(settings, "timestep", 0.04))
        # Blender exposes hair_length = 4 * normal_factor, but tangent/random
        # velocities are independent, normalized world-space contributions.
        length = (nominal_length + 4 * (tangent + random_velocity + aligned)) * max(
            1.0, timestep / 0.04
        )
        width = (
            2
            * max(float(settings.root_radius), float(settings.tip_radius))
            * float(settings.radius_scale)
        )
        extra = 0.0
        children = settings.child_type != "NONE" and settings.rendered_child_count > 0
        if children:
            if settings.child_type == "SIMPLE":
                extra += abs(float(settings.child_radius))
            if settings.kink not in {"NO", "NONE"}:
                extra += 2 * abs(float(settings.kink_amplitude))
            extra += math.sqrt(3.0) * sum(
                abs(float(getattr(settings, name, 0.0)))
                for name in ("roughness_1", "roughness_2", "roughness_endpoint")
            )
            extra += length * max(0.0, -float(settings.clump_factor))
        padding = (length + width + extra) * scale
        if (
            nominal_length < 0
            or timestep < 0
            or not math.isfinite(timestep)
            or width < 0
            or not math.isfinite(padding)
        ):
            raise ValueError("Invalid legacy particle hair extent")
        low, high = world.min(axis=0) - padding, world.max(axis=0) + padding
        result.extend(
            tuple(float(value) for value in point) for point in product(*zip(low, high))
        )
        if diagnostics is not None:
            diagnostics.append(
                {
                    "object": obj.name,
                    "count": settings.count,
                    "hair_length": nominal_length,
                    "parent_length_bound": length,
                    "tangent_factor": tangent,
                    "factor_random": random_velocity,
                    "object_align_factor_l1": aligned,
                    "world_padding": padding,
                    "child_type": settings.child_type,
                    "method": "static_generated_hair_velocity_envelope",
                }
            )
    return result


def scene_bounds(
    scene: Any,
    depsgraph: Any,
    *,
    include_volume: bool = True,
    include_particles: bool = True,
    diagnostics: list | None = None,
) -> tuple[Any, float, list[Any]]:
    from mathutils import Vector

    points = []
    for instance in depsgraph.object_instances:
        obj = instance.object
        types = {"MESH", "CURVE", "SURFACE", "META", "FONT"}
        if include_volume:
            types.add("VOLUME")
        if obj.type not in types:
            continue
        if obj.hide_render:
            continue
        points.extend(
            instance.matrix_world @ Vector(corner) for corner in obj.bound_box
        )
        if include_particles:
            points.extend(
                Vector(point) for point in particle_hair_bounds(instance, diagnostics)
            )
    if not points or not all(
        math.isfinite(value) for point in points for value in point
    ):
        raise ValueError("The generated scene has no finite, renderable geometry")
    lower = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
    upper = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
    center = (lower + upper) * 0.5
    radius = max((point - center).length for point in points)
    if radius <= 1e-8:
        raise ValueError("The generated geometry has degenerate bounds")
    return center, radius, points


def add_presentation(scene: Any) -> dict[str, Any]:
    """Add missing display equipment without changing model geometry/materials."""
    import bpy
    from mathutils import Vector

    bpy.context.view_layer.update()
    center, radius, points = scene_bounds(scene, bpy.context.evaluated_depsgraph_get())
    added = {"camera": False, "lights": [], "world": False}
    if scene.camera is None:
        cameras = [
            obj for obj in scene.objects if obj.type == "CAMERA" and not obj.hide_render
        ]
        if cameras:
            scene.camera = cameras[0]
        else:
            data = bpy.data.cameras.new("OverfitReviewCamera")
            camera = bpy.data.objects.new("OverfitReviewCamera", data)
            scene.collection.objects.link(camera)
            direction = Vector((6.0, -8.0, 6.0)).normalized()
            camera.location = center + direction * radius * 4.0
            camera.rotation_euler = (-direction).to_track_quat("-Z", "Y").to_euler()
            data.type = "ORTHO"
            rotation = camera.rotation_euler.to_matrix()
            right = rotation @ Vector((1.0, 0.0, 0.0))
            up = rotation @ Vector((0.0, 1.0, 0.0))
            half_width = max(abs((point - center).dot(right)) for point in points)
            half_height = max(abs((point - center).dot(up)) for point in points)
            # Square render; this is a framing operation, not geometry normalization.
            data.ortho_scale = 2.0 * max(half_width, half_height) * 1.2
            data.clip_start = max(radius * 1e-5, 1e-6)
            data.clip_end = radius * 20.0
            scene.camera = camera
            added["camera"] = True
    if not any(obj.type == "LIGHT" and not obj.hide_render for obj in scene.objects):
        for name, offset, power in [
            ("Key", (4.0, -6.0, 8.0), 650.0),
            ("Fill", (-6.0, -3.0, 4.0), 350.0),
            ("Rim", (3.0, 6.0, 6.0), 500.0),
        ]:
            data = bpy.data.lights.new("OverfitReview" + name, "AREA")
            data.energy = power * radius * radius
            data.shape = "DISK"
            data.size = radius * 3.0
            light = bpy.data.objects.new(data.name, data)
            scene.collection.objects.link(light)
            light.location = center + Vector(offset).normalized() * radius * 4.0
            light.rotation_euler = (
                (center - light.location).to_track_quat("-Z", "Y").to_euler()
            )
            added["lights"].append(light.name)
    if scene.world is None:
        scene.world = bpy.data.worlds.new("OverfitReviewWorld")
        scene.world.use_nodes = True
        background = scene.world.node_tree.nodes.get("Background")
        background.inputs["Color"].default_value = (0.8, 0.8, 0.8, 1.0)
        background.inputs["Strength"].default_value = 0.3
        added["world"] = True
    scene.render.resolution_x = 768
    scene.render.resolution_y = 768
    scene.render.resolution_percentage = 100
    return {"added": added, "center": list(center), "radius": radius}


def main() -> None:
    import bpy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", type=Path)
    parser.add_argument("--blend", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    if bool(args.code) == bool(args.blend):
        raise ValueError("Choose exactly one of --code or --blend")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "scene.blend").exists():
        raise FileExistsError("Refusing to overwrite an existing scene.blend")
    if args.code:
        # Execute the unchanged generated script as a standalone program.
        argv = sys.argv
        try:
            sys.argv = [str(args.code.resolve())]
            runpy.run_path(str(args.code.resolve()), run_name="__main__")
        finally:
            sys.argv = argv
    else:
        bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()), load_ui=False)
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    scene = bpy.context.scene
    presentation = add_presentation(scene)
    bpy.ops.wm.save_as_mainfile(
        filepath=str(output / "scene.blend"), check_existing=False
    )
    if args.render:
        # This image is for human review. Verifier views are generated independently.
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = 32
        scene.cycles.seed = 0
        scene.render.image_settings.file_format = "PNG"
        scene.render.filepath = str(output / "preview.png")
        bpy.ops.render.render(write_still=True)
    (output / "presentation.json").write_text(json.dumps(presentation, indent=2) + "\n")


if __name__ == "__main__":
    main()
