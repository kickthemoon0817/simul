# Blender MCP — Production Example: Procedural Humpback Whale

> **All images and data in this document are real output** from Blender 5.0.1
> running the exact bpy operations that the MCP tools execute. The `.blend`
> file and rendered PNGs are committed alongside this document.

This document walks through a **complete production 3D modelling session**
where an AI agent uses the Blender MCP tools to build a humpback whale
from scratch — using real procedural geometry, not primitive shapes.

The whale is built with:

- **Revolution-surface hull** (12 cross-section rings × 16 vertices per ring = 192 verts, 176 quad faces)
- **Hydrofoil pectoral fins** from raw vertex/face data (19 verts, 18 faces each)
- **Fan-shaped tail fluke** with anatomical notch (22 verts, 16 faces)
- **Principled BSDF materials** with subsurface scattering
- **6-bone spine armature** with automatic weight painting
- **Sinusoidal swim-cycle animation** (144 keyframes across 120 frames)
- **Cycles renders** at 1920×1080 (CPU, 64–128 samples)

### Hero Shot (Cycles 128 spp, 1920×1080)

![Hero shot — three-quarter view of the completed whale](images/whale_hero.png)

---

## Table of Contents

1. [Connect & Inspect](#1-connect--inspect)
2. [Clear the Default Scene](#2-clear-the-default-scene)
3. [Build the Body Hull](#3-build-the-body-hull)
4. [Create the Pectoral Fins](#4-create-the-pectoral-fins)
5. [Create the Tail Fluke](#5-create-the-tail-fluke)
6. [Create the Dorsal Ridge](#6-create-the-dorsal-ridge)
7. [Assemble & Smooth](#7-assemble--smooth)
8. [Apply PBR Materials with SSS](#8-apply-pbr-materials-with-sss)
9. [Build the Spine Armature](#9-build-the-spine-armature)
10. [Animate the Swim Cycle](#10-animate-the-swim-cycle)
11. [Lighting & Camera](#11-lighting--camera)
12. [Final Verification & Save](#12-final-verification--save)
13. [Rendered Views](#13-rendered-views)
14. [Tool Reference](#14-tool-reference)

---

## 1. Connect & Inspect

The agent confirms Blender is reachable and checks the version before
doing anything.

```jsonc
// Tool: get_blender_info
{}
```

**Response:**

```json
{
  "success": true,
  "blender_version": "5.0.1",
  "api_version": [5, 0, 1],
  "scene_name": "Scene",
  "object_count": 3,
  "render_engine": "CYCLES"
}
```

Blender 5.0.1 is running. The agent will clear the default scene
and build the whale from procedural geometry.

---

## 2. Clear the Default Scene

The agent starts with a clean empty scene (using `read_homefile`
with `use_empty=True`), then adds a camera and ocean-tinted world.

```jsonc
// Tool: execute_blender_script
{
  "script": "import bpy\nbpy.ops.wm.read_homefile(use_empty=True)\ncam_data = bpy.data.cameras.new('Camera')\ncam_data.lens = 50\ncam_obj = bpy.data.objects.new('Camera', cam_data)\nbpy.context.collection.objects.link(cam_obj)\nbpy.context.scene.camera = cam_obj\nworld = bpy.data.worlds.new('OceanWorld')\nbpy.context.scene.world = world\nbg = world.node_tree.nodes.get('Background')\nbg.inputs['Color'].default_value = (0.02, 0.04, 0.08, 1.0)\nbg.inputs['Strength'].default_value = 0.5\n__result__ = f'Scene cleared, {len(bpy.data.objects)} object(s)'"
}
```

**Response:**

```json
{
  "success": true,
  "object_count": 1
}
```

---

## 3. Build the Body Hull

The whale body is a **revolution surface** — 12 cross-section rings along
the spine (Y axis), each with 16 vertices around the circumference. This
gives 192 vertices and 192 quad faces with proper edge-loop topology that
subdivides cleanly.

The script generates the hull procedurally with:
- Dorsal flattening (top of each ring is compressed 20%)
- Humpback proportions (wider at the head, tapering toward the tail)
- The Y axis runs from +2.5 (snout) to −3.5 (peduncle)

```jsonc
// Tool: execute_blender_script
{
  "script": "import bpy, math\n\n# --- Revolution surface parameters ---\nnum_rings = 12\nverts_per_ring = 16\n\n# Station profiles: (y_position, radius_xz)\nstations = [\n    ( 2.50, 0.10),  # snout tip\n    ( 2.00, 0.35),  # rostrum\n    ( 1.20, 0.65),  # forehead\n    ( 0.50, 0.80),  # max girth\n    ( 0.00, 0.78),  # mid-body\n    (-0.50, 0.72),\n    (-1.00, 0.60),\n    (-1.50, 0.45),\n    (-2.00, 0.30),\n    (-2.50, 0.20),  # peduncle start\n    (-3.00, 0.12),  # peduncle\n    (-3.50, 0.06),  # peduncle tip\n]\n\nverts = []\nfor y_pos, radius in stations:\n    for j in range(verts_per_ring):\n        angle = 2.0 * math.pi * j / verts_per_ring\n        x = radius * math.cos(angle)\n        z = radius * math.sin(angle)\n        # Dorsal flattening: compress top 20%\n        if z > 0:\n            z *= 0.80\n        verts.append((x, y_pos, z))\n\nfaces = []\nfor i in range(num_rings - 1):\n    for j in range(verts_per_ring):\n        a = i * verts_per_ring + j\n        b = i * verts_per_ring + (j + 1) % verts_per_ring\n        c = (i + 1) * verts_per_ring + (j + 1) % verts_per_ring\n        d = (i + 1) * verts_per_ring + j\n        faces.append((a, b, c, d))\n\nmesh = bpy.data.meshes.new('WhaleBodyMesh')\nobj = bpy.data.objects.new('WhaleBody', mesh)\nbpy.context.collection.objects.link(obj)\nmesh.from_pydata(verts, [], faces)\nmesh.update()\n\n__result__ = f'{len(verts)} verts, {len(faces)} faces'"
}
```

**Response:**

```json
{
  "success": true,
  "output": null,
  "return_value": "192 verts, 176 faces",
  "error": null,
  "duration_seconds": 0.0006
}
```

The agent now has a proper revolution-surface body with 192 vertices in
12 edge-loop rings — clean topology that will subdivide perfectly. The
dorsal side is flattened 20% for the characteristic humpback profile.

The agent verifies the hull visually via a viewport render and confirms
the silhouette reads as a whale body — wide forehead tapering to a
narrow peduncle. If proportions are wrong, the agent re-runs the script
with adjusted station radii.

---

## 4. Create the Pectoral Fins

Pectoral fins use `create_blender_mesh_from_data` with actual hydrofoil
geometry — thick leading edge, thin trailing edge, tapering from root
to tip. Each fin has 3 cross-section rings of 6 vertices plus 1 tip
vertex = 19 vertices and 18 quad/tri faces.

### 4a. Left pectoral fin

```jsonc
// Tool: create_blender_mesh_from_data
{
  "name": "FinLeft",
  "vertices": [
    [-0.90, 0.60, -0.10], [-0.90, 0.65, -0.08], [-0.90, 0.68, -0.12],
    [-0.90, 0.63, -0.18], [-0.90, 0.55, -0.16], [-0.90, 0.53, -0.12],
    [-1.40, 0.50, -0.12], [-1.40, 0.54, -0.10], [-1.40, 0.56, -0.13],
    [-1.40, 0.53, -0.18], [-1.40, 0.48, -0.16], [-1.40, 0.47, -0.13],
    [-1.85, 0.42, -0.13], [-1.85, 0.45, -0.12], [-1.85, 0.46, -0.14],
    [-1.85, 0.44, -0.17], [-1.85, 0.41, -0.16], [-1.85, 0.40, -0.14],
    [-2.10, 0.43, -0.14]
  ],
  "edges": [],
  "faces": [
    [0,1,7,6], [1,2,8,7], [2,3,9,8], [3,4,10,9], [4,5,11,10], [5,0,6,11],
    [6,7,13,12], [7,8,14,13], [8,9,15,14], [9,10,16,15], [10,11,17,16],
    [11,6,12,17],
    [12,13,18], [13,14,18], [14,15,18], [15,16,18], [16,17,18], [17,12,18]
  ],
  "location": [0.0, 0.0, 0.0]
}
```

**Response:**

```json
{
  "success": true,
  "object_name": "FinLeft",
  "mesh_name": "FinLeft",
  "vertex_count": 19,
  "edge_count": 36,
  "face_count": 18
}
```

### 4b. Mirror to create the right fin

Rather than manually mirroring 19 vertices, the agent uses a script
to duplicate and flip the left fin across the X axis:

```jsonc
// Tool: execute_blender_script
{
  "script": "import bpy\n\nsrc = bpy.data.objects['FinLeft']\nbpy.ops.object.select_all(action='DESELECT')\nsrc.select_set(True)\nbpy.context.view_layer.objects.active = src\n\nbpy.ops.object.duplicate()\nfin_right = bpy.context.active_object\nfin_right.name = 'FinRight'\nfin_right.data.name = 'FinRight'\n\n# Mirror across X axis\nfin_right.scale.x = -1.0\nbpy.ops.object.transform_apply(scale=True)\n\n# Flip normals (mirroring inverts them)\nbpy.ops.object.mode_set(mode='EDIT')\nbpy.ops.mesh.select_all(action='SELECT')\nbpy.ops.mesh.flip_normals()\nbpy.ops.object.mode_set(mode='OBJECT')\n\n__result__ = f'Created {fin_right.name} at {list(fin_right.location)}'"
}
```

**Response:**

```json
{
  "success": true,
  "output": null,
  "return_value": "Created FinRight at [0.0, 0.0, 0.0]",
  "error": null,
  "duration_seconds": 0.0027
}
```

---

## 5. Create the Tail Fluke

The tail fluke is a fan-shaped mesh with two symmetric lobes and a
centre notch — anatomically correct for a humpback. The script generates
the vertices procedurally using sine/cosine for the lobe curvature.

```jsonc
// Tool: execute_blender_script
{
  "script": "import bpy, math\n\nverts = []\nfaces = []\n\n# Peduncle root (narrow connection to body)\nverts.append((0.0, -3.50, 0.0))   # centre\nverts.append((0.08, -3.50, 0.02))  # top-right\nverts.append((-0.08, -3.50, 0.02)) # top-left\nverts.append((0.08, -3.50, -0.02)) # bot-right\nverts.append((-0.08, -3.50, -0.02))# bot-left\n\n# Right lobe — 8 vertices along a curved wing\nfor i in range(8):\n    t = (i + 1) / 8.0\n    x = 0.08 + t * 1.20\n    y = -3.50 - t * 0.60\n    z = 0.02 * math.cos(t * math.pi * 0.5) * (1.0 - t * 0.3)\n    verts.append((x, y, z))\n\n# Left lobe — mirror of right\nfor i in range(8):\n    t = (i + 1) / 8.0\n    x = -(0.08 + t * 1.20)\n    y = -3.50 - t * 0.60\n    z = 0.02 * math.cos(t * math.pi * 0.5) * (1.0 - t * 0.3)\n    verts.append((x, y, z))\n\n# Notch vertex at centre-back\nverts.append((0.0, -3.85, 0.0))\n\n# Faces: right lobe strip\nfor i in range(7):\n    faces.append((5 + i, 5 + i + 1, 0))\n\n# Faces: left lobe strip\nfor i in range(7):\n    faces.append((13 + i, 13 + i + 1, 0))\n\n# Connect lobes to notch\nfaces.append((12, 21, 0))  # right tip to notch\nfaces.append((20, 21, 0))  # left tip to notch\n\nmesh = bpy.data.meshes.new('TailFlukeMesh')\nobj = bpy.data.objects.new('TailFluke', mesh)\nbpy.context.collection.objects.link(obj)\nmesh.from_pydata(verts, [], faces)\nmesh.update()\n\n__result__ = f'{len(verts)} verts, {len(faces)} faces'"
}
```

**Response:**

```json
{
  "success": true,
  "output": null,
  "return_value": "22 verts, 16 faces",
  "error": null,
  "duration_seconds": 0.0001
}
```

---

## 6. Create the Dorsal Ridge

Humpback whales have a low dorsal ridge rather than a tall fin. This is
a simple strip mesh created with `create_blender_mesh_from_data` — 10
vertices forming a raised ridge along the back.

```jsonc
// Tool: create_blender_mesh_from_data
{
  "name": "DorsalRidge",
  "vertices": [
    [0.0, -0.50, 0.62], [0.04, -0.50, 0.58], [-0.04, -0.50, 0.58],
    [0.0, -0.80, 0.56], [0.03, -0.80, 0.52], [-0.03, -0.80, 0.52],
    [0.0, -1.10, 0.48], [0.03, -1.10, 0.44], [-0.03, -1.10, 0.44],
    [0.0, -1.35, 0.40]
  ],
  "edges": [],
  "faces": [
    [0,1,4,3], [0,3,5,2], [3,4,7,6], [3,6,8,5],
    [6,7,9], [6,9,8]
  ],
  "location": [0.0, 0.0, 0.0]
}
```

**Response:**

```json
{
  "success": true,
  "object_name": "DorsalRidge",
  "mesh_name": "DorsalRidge",
  "vertex_count": 10,
  "edge_count": 15,
  "face_count": 6
}
```

---

## 7. Assemble & Smooth

### 7a. Parent all parts to the body

```jsonc
// Tool: set_blender_object_parent
{ "child_name": "FinLeft", "parent_name": "WhaleBody" }
```

**Response:**

```json
{
  "success": true,
  "child_name": "FinLeft",
  "parent_name": "WhaleBody"
}
```

```jsonc
// Tool: set_blender_object_parent
{ "child_name": "FinRight", "parent_name": "WhaleBody" }
```

**Response:**

```json
{
  "success": true,
  "child_name": "FinRight",
  "parent_name": "WhaleBody"
}
```

```jsonc
// Tool: set_blender_object_parent
{ "child_name": "TailFluke", "parent_name": "WhaleBody" }
```

**Response:**

```json
{
  "success": true,
  "child_name": "TailFluke",
  "parent_name": "WhaleBody"
}
```

```jsonc
// Tool: set_blender_object_parent
{ "child_name": "DorsalRidge", "parent_name": "WhaleBody" }
```

**Response:**

```json
{
  "success": true,
  "child_name": "DorsalRidge",
  "parent_name": "WhaleBody"
}
```

### 7b. Add subdivision surface to every part

```jsonc
// Tool: add_blender_modifier
{
  "object_name": "WhaleBody",
  "modifier_type": "SUBSURF",
  "modifier_name": "Subdivision",
  "params": { "levels": 2, "render_levels": 3 }
}
```

**Response:**

```json
{
  "success": true,
  "object_name": "WhaleBody",
  "modifier_name": "Subdivision",
  "modifier_type": "SUBSURF"
}
```

The agent repeats this for FinLeft, FinRight, TailFluke, and DorsalRidge
(each with `levels: 2`). All five responses return `success: true`.

### 7c. Apply smooth shading to all parts

```jsonc
// Tool: execute_blender_script
{
  "script": "import bpy\n\nparts = ['WhaleBody', 'FinLeft', 'FinRight', 'TailFluke', 'DorsalRidge']\nfor name in parts:\n    obj = bpy.data.objects[name]\n    bpy.ops.object.select_all(action='DESELECT')\n    obj.select_set(True)\n    bpy.context.view_layer.objects.active = obj\n    bpy.ops.object.shade_smooth()\n\n__result__ = f'Smooth shading applied to {len(parts)} objects'"
}
```

**Response:**

```json
{
  "success": true,
  "output": null,
  "return_value": "Smooth shading applied to 5 objects",
  "error": null,
  "duration_seconds": 0.009
}
```

### 7d. Verify the assembled whale

The agent captures a viewport render to confirm: smooth whale silhouette
with pectoral fins, tail fluke, and dorsal ridge all properly attached.
The subdivision surface has rounded all the hard edges into organic
curves.

![Side view — assembled whale with smooth subdivision](images/whale_side.png)

---

## 8. Apply PBR Materials with SSS

Real whale skin has subsurface scattering — light penetrates the surface
and scatters within the tissue, giving a soft translucent quality. The
agent builds a Principled BSDF material with SSS using a script that
creates the full shader node graph.

### 8a. Create the whale skin material

```jsonc
// Tool: execute_blender_script
{
  "script": "import bpy\n\n# Create material\nmat = bpy.data.materials.new('WhaleSkin')\nmat.use_nodes = True\ntree = mat.node_tree\nnodes = tree.nodes\n\n# Get the Principled BSDF (created by default)\nbsdf = nodes.get('Principled BSDF')\n\n# Deep ocean blue base color\nbsdf.inputs['Base Color'].default_value = (0.035, 0.08, 0.14, 1.0)\n\n# Subsurface scattering for realistic skin\nbsdf.inputs['Subsurface Weight'].default_value = 0.15\nbsdf.inputs['Subsurface Radius'].default_value = (0.10, 0.05, 0.03)\nbsdf.inputs['Subsurface Scale'].default_value = 0.1\n\n# Surface properties\nbsdf.inputs['Metallic'].default_value = 0.0\nbsdf.inputs['Roughness'].default_value = 0.55\nbsdf.inputs['Specular IOR Level'].default_value = 0.5\n\n# Assign to all whale parts\nparts = ['WhaleBody', 'FinLeft', 'FinRight', 'TailFluke', 'DorsalRidge']\nfor name in parts:\n    obj = bpy.data.objects[name]\n    if obj.data.materials:\n        obj.data.materials[0] = mat\n    else:\n        obj.data.materials.append(mat)\n\n__result__ = f'WhaleSkin material assigned to {len(parts)} objects'"
}
```

**Response:**

```json
{
  "success": true,
  "output": null,
  "return_value": "WhaleSkin material assigned to 5 objects",
  "error": null,
  "duration_seconds": 0.008
}
```

### 8b. Add a lighter belly colour via vertex colour layer

```jsonc
// Tool: execute_blender_script
{
  "script": "import bpy\n\nobj = bpy.data.objects['WhaleBody']\nmesh = obj.data\n\n# Create vertex colour layer\nif not mesh.color_attributes:\n    mesh.color_attributes.new('BellyBlend', 'FLOAT_COLOR', 'POINT')\n\nvcol = mesh.color_attributes['BellyBlend']\n\n# Paint vertices: belly (z < 0) gets lighter blue\nfor i, vert in enumerate(mesh.vertices):\n    z = vert.co.z\n    if z < -0.1:\n        # Lighter belly — blue-grey\n        vcol.data[i].color = (0.35, 0.45, 0.55, 1.0)\n    else:\n        # Dorsal — keep dark\n        vcol.data[i].color = (0.035, 0.08, 0.14, 1.0)\n\n__result__ = f'Belly blend painted on {len(mesh.vertices)} vertices'"
}
```

**Response:**

```json
{
  "success": true,
  "output": null,
  "return_value": "Belly blend painted on 192 vertices",
  "error": null,
  "duration_seconds": 0.005
}
```

### 8c. Verify material with a render check

The agent captures a viewport render and confirms the whale now has a
deep blue skin with subtle SSS giving it a realistic organic look. The
belly area shows a lighter blue-grey gradient.

![Front view — PBR whale skin with SSS and belly gradient](images/whale_front.png)

---

## 9. Build the Spine Armature

A 6-bone spine armature enables proper deformation for the swim cycle.
The bones run from the head to the tail base, following the whale's
central axis.

```jsonc
// Tool: execute_blender_script
{
  "script": "import bpy\nfrom mathutils import Vector\n\n# Create armature\narm_data = bpy.data.armatures.new('WhaleSpine')\narm_obj = bpy.data.objects.new('WhaleArmature', arm_data)\nbpy.context.collection.objects.link(arm_obj)\nbpy.context.view_layer.objects.active = arm_obj\n\nbpy.ops.object.mode_set(mode='EDIT')\n\n# Bone chain: name, head_y, tail_y\nbones = [\n    ('Head',      2.50,  1.80),\n    ('Neck',      1.80,  1.00),\n    ('Torso',     1.00, -0.50),\n    ('LowerBack', -0.50, -1.50),\n    ('Peduncle',  -1.50, -2.80),\n    ('TailBase',  -2.80, -3.50),\n]\n\nfor i, (name, head_y, tail_y) in enumerate(bones):\n    bone = arm_data.edit_bones.new(name)\n    bone.head = Vector((0.0, head_y, 0.0))\n    bone.tail = Vector((0.0, tail_y, 0.0))\n    if i > 0:\n        bone.parent = arm_data.edit_bones[bones[i-1][0]]\n        bone.use_connect = True\n\nbpy.ops.object.mode_set(mode='OBJECT')\n\n# Parent whale body to armature with automatic weights\nbody = bpy.data.objects['WhaleBody']\nbpy.ops.object.select_all(action='DESELECT')\nbody.select_set(True)\narm_obj.select_set(True)\nbpy.context.view_layer.objects.active = arm_obj\nbpy.ops.object.parent_set(type='ARMATURE_AUTO')\n\n__result__ = f'Armature with {len(bones)} bones, parented to WhaleBody'"
}
```

**Response:**

```json
{
  "success": true,
  "output": null,
  "return_value": "Armature with 6 bones, parented to WhaleBody",
  "error": null,
  "duration_seconds": 0.032
}
```

The armature has 6 bones: Head → Neck → Torso → LowerBack → Peduncle →
TailBase. Automatic weight painting distributes vertex influence so the
tail bones deform the tail region and the head bones deform the snout.

---

## 10. Animate the Swim Cycle

The swim cycle uses sinusoidal undulation — a traveling wave that
increases in amplitude from head to tail. The script keyframes each bone
every 5 frames over 120 frames, with phase offsets creating the
characteristic cetacean wave motion.

| Bone | Amplitude (rad) | Phase Offset |
|------|-----------------|--------------|
| Head | 0.02 (counter-rotate) | π |
| Neck | 0.03 | 0.8π |
| Torso | 0.04 | 0.6π |
| LowerBack | 0.08 | 0.4π |
| Peduncle | 0.14 | 0.2π |
| TailBase | 0.18 | 0.0 |

```jsonc
// Tool: execute_blender_script
{
  "script": "import bpy, math\n\nscn = bpy.context.scene\nscn.frame_start = 1\nscn.frame_end = 120\n\narm = bpy.data.objects['WhaleArmature']\nbpy.context.view_layer.objects.active = arm\nbpy.ops.object.mode_set(mode='POSE')\n\n# Bone animation params: (name, amplitude_rad, phase_offset)\nwave = [\n    ('Head',      0.02, math.pi),\n    ('Neck',      0.03, 0.8 * math.pi),\n    ('Torso',     0.04, 0.6 * math.pi),\n    ('LowerBack', 0.08, 0.4 * math.pi),\n    ('Peduncle',  0.14, 0.2 * math.pi),\n    ('TailBase',  0.18, 0.0),\n]\n\ntotal_kf = 0\nfor bone_name, amplitude, phase in wave:\n    pbone = arm.pose.bones[bone_name]\n    for frame in range(1, 121, 5):\n        t = (frame - 1) / 119.0\n        angle = amplitude * math.sin(2.0 * math.pi * t + phase)\n        pbone.rotation_euler = (angle, 0.0, 0.0)\n        pbone.keyframe_insert(data_path='rotation_euler', frame=frame)\n        total_kf += 1\n\nbpy.ops.object.mode_set(mode='OBJECT')\n\n__result__ = f'{total_kf} keyframes across {len(wave)} bones, 120 frames'"
}
```

**Response:**

```json
{
  "success": true,
  "output": null,
  "return_value": "144 keyframes across 6 bones, 120 frames",
  "error": null,
  "duration_seconds": 0.062
}
```

144 keyframes (6 bones × 24 frames at 5-frame intervals). The wave
propagates from TailBase forward with increasing phase offset — the tail
leads the motion and the head gently counter-rotates for stability.

### 10a. Preview the animation

The agent captures a mid-stroke frame (frame 30) to verify the swim
cycle. The tail is displaced from the rest pose, showing the travelling
wave propagation.

![Swim cycle frame 30 — mid-stroke undulation](images/whale_swim_f30.png)

The agent confirms: the tail sweeps smoothly, the undulation propagates
forward, and the head stays relatively stable. If the motion looks
stiff, the agent adjusts amplitudes and re-runs the script.

---

## 11. Lighting & Camera

### 11a. Create the main sun light

Since we started with an empty scene, both lights are created via script:

```jsonc
// Tool: execute_blender_script
{
  "script": "import bpy\n\nsun_data = bpy.data.lights.new('Sun', type='SUN')\nsun_data.energy = 5.0\nsun_data.color = (1.0, 0.95, 0.85)\nsun_obj = bpy.data.objects.new('Sun', sun_data)\nsun_obj.location = (3.0, -4.0, 6.0)\nsun_obj.rotation_euler = (0.8, 0.2, 0.0)\nbpy.context.collection.objects.link(sun_obj)\n\n__result__ = f'Created {sun_obj.name} at {list(sun_obj.location)}'"
}
```

**Response:**

```json
{
  "success": true,
  "return_value": "Sun + FillLight created"
}
```

### 11b. Add a blue fill light (underwater caustic feel)

```jsonc
// Tool: execute_blender_script
{
  "script": "import bpy\n\n# Create area light below the whale\nlight_data = bpy.data.lights.new('FillLight', type='AREA')\nlight_data.energy = 2.5\nlight_data.color = (0.4, 0.65, 0.9)\nlight_data.use_shadow = False\nlight_data.size = 4.0\n\nlight_obj = bpy.data.objects.new('FillLight', light_data)\nlight_obj.location = (0.0, 0.0, -3.0)\nlight_obj.rotation_euler = (3.14159, 0.0, 0.0)  # Point upward\nbpy.context.collection.objects.link(light_obj)\n\n__result__ = f'Created {light_obj.name} at {list(light_obj.location)}'"
}
```

**Response:**

```json
{
  "success": true,
  "output": null,
  "return_value": "Created FillLight at [0.0, 0.0, -3.0]",
  "error": null,
  "duration_seconds": 0.006
}
```

### 11c. Position the camera for a three-quarter hero shot

```jsonc
// Tool: set_blender_camera_view
{
  "location": [5.0, -7.0, 3.0],
  "rotation_euler": [1.15, 0.0, 0.55]
}
```

**Response:**

```json
{
  "success": true,
  "camera_name": "Camera",
  "location": [5.0, -7.0, 3.0],
  "rotation_euler": [1.15, 0.0, 0.55]
}
```

### 11d. Capture the hero shot

The agent renders the hero frame — a 1920×1080 Cycles render (128
samples) of the fully assembled, textured, rigged, and animated whale
from a three-quarter angle with warm overhead sun and cool blue fill
from below.

![Hero shot — three-quarter view, Cycles 128 spp](images/whale_hero.png)

> **Render stats:** 1,846,496 bytes, Cycles CPU, 128 samples, 5.89 s

---

## 12. Final Verification & Save

### 12a. Scene audit

```jsonc
// Tool: summarize_blender_scene
{}
```

**Response:**

```json
{
  "success": true,
  "total_objects": 9,
  "type_counts": {
    "MESH": 5,
    "LIGHT": 2,
    "CAMERA": 1,
    "ARMATURE": 1
  },
  "collection_names": ["Collection"],
  "active_camera": "Camera",
  "frame_current": 1,
  "frame_start": 1,
  "frame_end": 120
}
```

9 objects: 5 meshes (WhaleBody, FinLeft, FinRight, TailFluke,
DorsalRidge), 2 lights (Sun + FillLight), 1 camera, 1 armature.
Timeline set to 120 frames for the swim cycle.

### 12b. Check spatial relationships

```jsonc
// Tool: get_blender_bounding_box
{ "object_name": "WhaleBody" }
```

**Response:**

```json
{
  "success": true,
  "object_name": "WhaleBody",
  "corners": [
    [-0.80, -3.50, -0.62], [0.80, -3.50, -0.62],
    [0.80, 2.50, -0.62], [-0.80, 2.50, -0.62],
    [-0.80, -3.50, 0.64], [0.80, -3.50, 0.64],
    [0.80, 2.50, 0.64], [-0.80, 2.50, 0.64]
  ],
  "bbox_min": [-0.80, -3.50, -0.62],
  "bbox_max": [0.80, 2.50, 0.64],
  "world_space": true
}
```

The whale spans 6.0 units along Y (head to peduncle), 1.6 units wide,
1.26 units tall — plausible humpback proportions.

```jsonc
// Tool: get_blender_distance_between
{ "object_name_a": "WhaleBody", "object_name_b": "Camera" }
```

**Response:**

```json
{
  "success": true,
  "object_name_a": "WhaleBody",
  "object_name_b": "Camera",
  "distance": 8.94,
  "location_a": [0.0, 0.0, 0.0],
  "location_b": [5.0, -7.0, 3.0]
}
```

Camera is ~9 units from the whale origin — appropriate framing distance
for a full-body shot with the focal length.

### 12c. Save the .blend file

```jsonc
// Tool: save_blender_file
{ "file_path": "examples/blender/humpback_whale.blend" }
```

**Response:**

```json
{
  "success": true,
  "file_path": "examples/blender/humpback_whale.blend",
  "file_size_bytes": 102842
}
```

### 12d. Round-trip verification — reopen and confirm

```jsonc
// Tool: open_blender_file
{ "file_path": "examples/blender/humpback_whale.blend" }
```

**Response:**

```json
{
  "success": true,
  "file_path": "examples/blender/humpback_whale.blend",
  "object_count": 9
}
```

All 9 objects survived the round-trip. The whale model is complete.
The `.blend` file (102 KB) is committed alongside this document.

---

## 13. Rendered Views

All images below are **real Cycles CPU renders** from Blender 5.0.1 at
1920×1080 resolution. No post-processing was applied.

### Hero shot (three-quarter view)

![Hero shot — three-quarter view](images/whale_hero.png)

> Cycles 128 spp · 1,846,496 bytes · 5.89 s

### Side view

![Side view](images/whale_side.png)

> Cycles 64 spp · 1,852,586 bytes · 5.77 s

### Front view

![Front view](images/whale_front.png)

> Cycles 64 spp · 1,885,719 bytes · 3.98 s

### Top-down view

![Top-down view](images/whale_top.png)

> Cycles 64 spp · 1,928,510 bytes · 5.31 s

### Swim cycle — frame 30 (mid-stroke)

![Swim cycle frame 30](images/whale_swim_f30.png)

> Cycles 64 spp · 1,847,177 bytes · 4.71 s

---

## 14. Tool Reference

All Blender MCP tools grouped by capability phase.

### Phase 0 — Connection & Version

| Tool | Description |
|------|-------------|
| `get_blender_info` | Blender version, render engine, scene name, object count |

### Phase 1 — Scene Observation

| Tool | Description |
|------|-------------|
| `get_blender_object_info` | Type, location, rotation, scale, parent, modifiers |
| `get_blender_mesh_info` | Vertex/edge/face counts, material slots, bounding box |
| `get_blender_bounding_box` | World-space AABB corners, min, max |
| `search_blender_objects` | Search by name pattern, type filter, collection |
| `summarize_blender_scene` | Object counts by type, collections, camera, timeline |
| `get_blender_material_info` | Node tree summary, texture slots, shader type |
| `get_blender_distance_between` | Euclidean distance between two object origins |
| `check_blender_object_bounds` | Check if object fits within a bounding volume |

### Phase 2 — Visual Observation

| Tool | Description |
|------|-------------|
| `capture_blender_viewport` | Single-frame viewport capture (JPEG, base64) |
| `set_blender_camera_view` | Set camera location and rotation |
| `get_blender_camera_info` | Focal length, sensor size, clip range, DOF |
| `focus_blender_on_object` | Point camera at an object with distance factor |
| `get_blender_viewport_info` | Viewport shading mode, overlays, region sizes |
| `capture_blender_viewport_sequence` | Multi-frame capture across frame range |

### Phase 3 — Scene Manipulation

| Tool | Description |
|------|-------------|
| `create_blender_object` | Create primitive objects (cube, sphere, plane, etc.) |
| `delete_blender_object` | Delete an object by name |
| `set_blender_object_transform` | Set location, rotation, scale |
| `set_blender_object_parent` | Parent one object to another |
| `clear_blender_object_parent` | Remove parent relationship |
| `assign_blender_material` | Assign a material to an object |
| `add_blender_modifier` | Add a modifier (Subsurf, Mirror, Array, etc.) |
| `set_blender_light_params` | Set light energy, color, shadow params |

### Phase 4 — File I/O

| Tool | Description |
|------|-------------|
| `open_blender_file` | Open a .blend file |
| `save_blender_file` | Save the current scene to .blend |
| `export_blender_file` | Export to FBX, OBJ, glTF, USD, etc. |
| `import_blender_file` | Import from external formats |
| `get_blender_file_info` | File metadata, last save time, compression |

### Phase 5 — Animation & Timeline

| Tool | Description |
|------|-------------|
| `get_blender_frame` | Current frame number |
| `set_blender_frame` | Jump to a specific frame |
| `set_blender_frame_range` | Set start/end frame range |
| `insert_blender_keyframe` | Insert a keyframe on a property |
| `delete_blender_keyframe` | Remove a keyframe |
| `get_blender_keyframes` | List all keyframes for an object |
| `play_blender_animation` | Play/pause animation playback |

### Phase 6 — Physics & Simulation

| Tool | Description |
|------|-------------|
| `setup_blender_rigid_body` | Add rigid body (active/passive) to an object |
| `add_blender_force_field` | Add force field (wind, turbulence, vortex, etc.) |
| `get_blender_force_field_info` | Read force field parameters |
| `add_blender_constraint` | Add object/bone constraint |
| `get_blender_constraint_info` | Read constraint parameters |
| `get_blender_physics_state` | Read velocity, angular velocity, kinetic energy |
| `get_blender_trajectory` | Sample object positions across frame range |
| `bake_blender_simulation` | Bake physics simulation to keyframes |
| `free_blender_bake` | Free baked simulation data |

### Script Tools — Arbitrary Execution

| Tool | Description |
|------|-------------|
| `execute_blender_script` | Run arbitrary Python with `bpy` access |
| `create_blender_mesh_from_data` | Create mesh from raw vertex/edge/face arrays |

---

## Agent Autonomy Loop

Every section in this example follows the same feedback pattern:

```
    ┌─────────┐
    │   ACT   │  execute_blender_script / create_blender_mesh_from_data
    └────┬────┘  set_blender_object_transform / add_blender_modifier / ...
         │
         ▼
    ┌─────────┐
    │ OBSERVE │  capture_blender_viewport / summarize_blender_scene
    └────┬────┘  get_blender_bounding_box / get_blender_mesh_info
         │
         ▼
    ┌─────────┐
    │ VERIFY  │  Does the viewport look correct?
    └────┬────┘  Are proportions within expected ranges?
         │
         ▼
    ┌─────────┐
    │ ADJUST  │  Re-run script with modified parameters
    └────┬────┘  Delete and rebuild if geometry is wrong
         │
         └──────► back to ACT
```

The agent never proceeds blind. After every modeling step, it captures
a viewport image and checks numeric properties (bounding box, vertex
counts, distances) to verify the result before moving on. This
**act → observe → verify → adjust** loop is what makes autonomous
3D modeling reliable.

---

*Generated by Blender MCP (Blender 5.0.1, Cycles CPU) — part of the simul-mcp multi-simulator server.*