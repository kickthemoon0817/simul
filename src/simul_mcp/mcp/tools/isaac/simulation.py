"""Simulation Control tools for Isaac Sim."""

import json
import textwrap
from typing import Any, Callable, Dict, List, Optional

from ....adapters import IsaacSocketClient, ScriptResult
from ...schemas.common import ErrorResponse
from ._shared import (
    BULK_GEOMETRY_ATTRIBUTES,
    LOG_SCAN_WINDOW_BYTES,
    MAX_CAPTURE_DIMENSION,
    MAX_INLINE_CAPTURE_BYTES,
    MAX_RETAINED_CAPTURES,
    MAX_SCRIPT_BYTES,
    PRIM_DETAIL_ASPECTS,
    FloatList,
    _pyval,
    logger,
)
from .._meta import tool_meta


class SimulationMixin:
    # ------------------------------------------------------------------
    # Phase 6: Simulation Control
    # ------------------------------------------------------------------

    @tool_meta(
        name="get_isaac_simulation_state",
        description=(
            "Get the current simulation state: playing, paused, or stopped, along with "
            "current time and time-codes-per-second."
        ),
        read_only=True,
        idempotent=True,
    )
    async def get_isaac_simulation_state(self) -> Dict[str, Any]:
        """
        Get the current simulation state (playing, paused, stopped).

        Returns:
            Dict with simulation state, time, and step count.
        """
        bridge_result = await self._execute_bridge_action("get_simulation_state")
        if bridge_result is not None:
            return bridge_result

        script = textwrap.dedent("""\
            import json
            try:
                import omni.timeline
                timeline = omni.timeline.get_timeline_interface()
                is_playing = timeline.is_playing()
                is_stopped = timeline.is_stopped()
                current_time = timeline.get_current_time()
                tps = timeline.get_time_codes_per_seconds()
                if is_playing:
                    state = "playing"
                elif is_stopped:
                    state = "stopped"
                else:
                    state = "paused"
                print(json.dumps({
                    "state": state,
                    "current_time": current_time,
                    "time_codes_per_second": tps,
                    "is_playing": is_playing,
                    "is_stopped": is_stopped,
                }))
            except Exception as e:
                print(json.dumps({"error": f"Failed to get simulation state: {e}"}))
        """)
        mode = self._raw_script_transport_mode
        return await self._execute_json_script(script, transport_mode=mode)

    @tool_meta(
        name="start_isaac_simulation",
        description="Start (play) the simulation timeline.",
        read_only=False,
        idempotent=True,
    )
    async def start_isaac_simulation(self) -> Dict[str, Any]:
        """
        Start (play) the simulation.

        Returns:
            Dict confirming simulation started.
        """
        bridge_result = await self._execute_bridge_action(
            "simulation_control",
            {"command": "start"},
        )
        if bridge_result is not None:
            return bridge_result

        script = textwrap.dedent("""\
            import json
            try:
                import omni.timeline
                timeline = omni.timeline.get_timeline_interface()
                timeline.play()
                print(json.dumps({"state": "playing", "started": True}))
            except Exception as e:
                print(json.dumps({"error": f"Failed to start simulation: {e}"}))
        """)
        mode = self._raw_script_transport_mode
        return await self._execute_json_script(script, transport_mode=mode)

    @tool_meta(
        name="stop_isaac_simulation",
        description="Stop the simulation and reset to initial state.",
        read_only=False,
        idempotent=True,
    )
    async def stop_isaac_simulation(self) -> Dict[str, Any]:
        """
        Stop the simulation and reset to initial state.

        Returns:
            Dict confirming simulation stopped.
        """
        bridge_result = await self._execute_bridge_action(
            "simulation_control",
            {"command": "stop"},
        )
        if bridge_result is not None:
            return bridge_result

        script = textwrap.dedent("""\
            import json
            try:
                import omni.timeline
                timeline = omni.timeline.get_timeline_interface()
                timeline.stop()
                print(json.dumps({"state": "stopped", "stopped": True}))
            except Exception as e:
                print(json.dumps({"error": f"Failed to stop simulation: {e}"}))
        """)
        mode = self._raw_script_transport_mode
        return await self._execute_json_script(script, transport_mode=mode)

    @tool_meta(
        name="pause_isaac_simulation",
        description="Pause the currently running simulation.",
        read_only=False,
        idempotent=True,
    )
    async def pause_isaac_simulation(self) -> Dict[str, Any]:
        """
        Pause the running simulation.

        Returns:
            Dict confirming simulation paused.
        """
        bridge_result = await self._execute_bridge_action(
            "simulation_control",
            {"command": "pause"},
        )
        if bridge_result is not None:
            return bridge_result

        script = textwrap.dedent("""\
            import json
            try:
                import omni.timeline
                timeline = omni.timeline.get_timeline_interface()
                timeline.pause()
                print(json.dumps({"state": "paused", "paused": True}))
            except Exception as e:
                print(json.dumps({"error": f"Failed to pause simulation: {e}"}))
        """)
        mode = self._raw_script_transport_mode
        return await self._execute_json_script(script, transport_mode=mode)

    @tool_meta(
        name="step_isaac_simulation",
        description=(
            "Step the simulation forward by exactly N frames and leave the timeline "
            "paused. Works from a playing, paused or stopped timeline; reports the "
            "frames advanced and the time delta."
        ),
        read_only=False,
    )
    async def step_isaac_simulation(
        self, num_steps: int = 1
    ) -> Dict[str, Any]:
        """
        Step the simulation forward by exactly N frames, then pause.

        Args:
            num_steps: Number of timeline frames to advance (one physics
                step each at the default physics rate). Clamped to [1, 1000].

        Returns:
            Dict with ``steps`` advanced, ``steps_requested``,
            ``start_time``, ``current_time``, ``time_delta`` and ``state``
            ("paused"); ``error`` when fewer frames advanced than requested.
        """
        num_steps = max(1, min(num_steps, 1000))
        bridge_result = await self._execute_bridge_action(
            "simulation_control",
            {"command": "step", "num_steps": num_steps},
        )
        if bridge_result is not None:
            return bridge_result

        script = textwrap.dedent(f"""\
            import json
            try:
                import omni.timeline
                import omni.kit.app

                timeline = omni.timeline.get_timeline_interface()
                app = omni.kit.app.get_app()
                num_steps = {num_steps}

                def _commit():
                    commit = getattr(timeline, "commit", None)
                    if callable(commit):
                        commit()

                # A paused timeline does not move on app updates and a
                # stopped one spends its first updates after play() on
                # physics warm-up, so play and count only the updates that
                # moved the time; then pause. The budget bounds the loop when
                # time cannot advance (end of a non-looping range).
                start_time = timeline.get_current_time()
                if not timeline.is_playing():
                    timeline.play()
                    _commit()
                advanced = 0
                updates = 0
                last = start_time
                try:
                    while advanced < num_steps and updates < num_steps + 60:
                        await app.next_update_async()
                        updates += 1
                        now = timeline.get_current_time()
                        if now != last:
                            advanced += 1
                            last = now
                finally:
                    timeline.pause()
                    _commit()

                current_time = timeline.get_current_time()
                result = {{
                    "steps": advanced,
                    "steps_requested": num_steps,
                    "start_time": start_time,
                    "current_time": current_time,
                    "time_delta": current_time - start_time,
                    "app_updates": updates,
                    "state": "paused",
                }}
                if advanced < num_steps:
                    result["error"] = (
                        f"Timeline advanced {{advanced}} of {{num_steps}} steps in "
                        f"{{updates}} app updates; it may be at the end of a "
                        "non-looping time range."
                    )
                print(json.dumps(result))
            except Exception as e:
                print(json.dumps({{"error": f"Failed to step simulation: {{e}}"}}))
        """)
        mode = self._raw_script_transport_mode
        return await self._execute_json_script(script, transport_mode=mode)

    @tool_meta(
        name="reset_isaac_simulation",
        description="Reset the simulation to initial state and time 0.",
        read_only=False,
        idempotent=True,
    )
    async def reset_isaac_simulation(self) -> Dict[str, Any]:
        """
        Reset the simulation to initial state (stop + set time to 0).

        Returns:
            Dict confirming reset.
        """
        bridge_result = await self._execute_bridge_action(
            "simulation_control",
            {"command": "reset"},
        )
        if bridge_result is not None:
            return bridge_result

        script = textwrap.dedent("""\
            import json
            try:
                import omni.timeline
                timeline = omni.timeline.get_timeline_interface()
                timeline.stop()
                timeline.set_current_time(0)
                print(json.dumps({
                    "state": "stopped",
                    "current_time": 0.0,
                    "reset": True,
                }))
            except Exception as e:
                print(json.dumps({"error": f"Failed to reset simulation: {e}"}))
        """)
        mode = self._raw_script_transport_mode
        return await self._execute_json_script(script, transport_mode=mode)

    @tool_meta(
        name="get_isaac_simulation_time",
        description="Get current simulation time, start/end times, and time-codes-per-second.",
        read_only=True,
        idempotent=True,
    )
    async def get_isaac_simulation_time(self) -> Dict[str, Any]:
        """
        Get current simulation time information.

        Returns:
            Dict with current time, FPS, and time codes per second.
        """
        script = textwrap.dedent("""\
            import json
            try:
                import omni.timeline
                timeline = omni.timeline.get_timeline_interface()
                print(json.dumps({
                    "current_time": timeline.get_current_time(),
                    "start_time": timeline.get_start_time(),
                    "end_time": timeline.get_end_time(),
                    "time_codes_per_second": timeline.get_time_codes_per_seconds(),
                    "is_playing": timeline.is_playing(),
                }))
            except Exception as e:
                print(json.dumps({"error": f"Failed to get simulation time: {e}"}))
        """)
        return await self._execute_json_script(script)

