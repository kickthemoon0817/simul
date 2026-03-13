# API Quick Reference

| Task | Module | Key Functions |
|------|--------|---------------|
| Play/Pause/Stop | `omni.timeline` | `get_timeline_interface().play/pause/stop()` |
| Step physics | `isaacsim.core.api.World` | `World.instance().step(render=True)` |
| Get stage | `omni.usd` | `get_context().get_stage()` |
| Open/Save stage | `isaacsim.core.utils.stage` | `open_stage()`, `save_stage()` |
| New blank stage | `isaacsim.core.utils.stage` | `create_new_stage()` |
| Traverse prims | `pxr.Usd` | `stage.Traverse()` |
| Create prim | `isaacsim.core.utils.prims` | `create_prim(path, type, position=, scale=, attributes=)` |
| Delete prim | `isaacsim.core.utils.prims` | `delete_prim(path)` |
| Get prim at path | `isaacsim.core.utils.prims` | `get_prim_at_path(path)` |
| List attributes | `isaacsim.core.utils.prims` | `get_prim_attribute_names(path)` |
| Get attribute | `isaacsim.core.utils.prims` | `get_prim_attribute_value(path, attr)` |
| Set attribute | `isaacsim.core.utils.prims` | `set_prim_attribute_value(path, attr, value)` |
| Set visibility | `isaacsim.core.utils.prims` | `set_prim_visibility(prim, visible)` |
| World pose | `isaacsim.core.utils.xforms` | `get_world_pose(path)` → (pos, quat_wxyz) |
| Local pose | `isaacsim.core.utils.xforms` | `get_local_pose(path)` → (pos, quat_wxyz) |
| Camera view | `isaacsim.core.utils.viewports` | `set_camera_view(eye, target, camera_prim_path)` |
| Active viewport cam | `isaacsim.core.utils.viewports` | `set_active_viewport_camera(path)` |
| Rigid body | `pxr.UsdPhysics` | `RigidBodyAPI.Apply(prim)` |
| Collision | `pxr.UsdPhysics` | `CollisionAPI.Apply(prim)` |
| Mass | `pxr.UsdPhysics` | `MassAPI.Apply(prim)` |
| Fixed joint | `pxr.UsdPhysics` | `FixedJoint.Define(stage, path)` |
| Revolute joint | `pxr.UsdPhysics` | `RevoluteJoint.Define(stage, path)` |
| Prismatic joint | `pxr.UsdPhysics` | `PrismaticJoint.Define(stage, path)` |
| Physics scene | `pxr.UsdPhysics` | `Scene.Define(stage, path)` |
| Materials | `pxr.UsdShade` | `Material.Define(stage, path)`, `Shader.Define(stage, path)` |
| Distant light | `pxr.UsdLux` | `DistantLight.Define(stage, path)` |
| Dome light | `pxr.UsdLux` | `DomeLight.Define(stage, path)` |
| Sphere light | `pxr.UsdLux` | `SphereLight.Define(stage, path)` |
| Bounding box | `isaacsim.core.utils.bounds` | `create_bbox_cache()`, `compute_aabb(cache, path)` |
| OBB | `isaacsim.core.utils.bounds` | `compute_obb(cache, path)` |
| Euler to quat | `isaacsim.core.utils.rotations` | `euler_angles_to_quat(angles)` → wxyz |
| Quat to Euler | `isaacsim.core.utils.rotations` | `quat_to_euler_angles(quat, degrees=True)` |
| Raycast | `isaacsim.core.utils.collisions` | `ray_cast(position, orientation, offset, max_dist)` |
| Load USD asset | `isaacsim.core.utils.stage` | `add_reference_to_stage(usd_path, prim_path)` |
| Selection query | `omni.usd` | `get_context().get_selection().get_selected_prim_paths()` |
| Select prim | `omni.usd` | `get_context().get_selection().set_prim_path_selected(path, ...)` |
| Stage up axis | `pxr.UsdGeom` | `GetStageUpAxis(stage)` |
| Stage units | `pxr.UsdGeom` | `GetStageMetersPerUnit(stage)` |
