"""Simulation Control tools for Isaac Sim."""

import textwrap
from typing import Any, Dict

from ._shared import (
    HEAVY_SCRIPT_TIMEOUT_SECONDS,
    PHYSICS_RATES_CORE,
    STEP_BUDGET_BASE_SECONDS,
    STEP_BUDGET_SECONDS_PER_FRAME,
    _compose_script,
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
        description=(
            "Stop the simulation and reset to initial state. With /physics/resetOnStop "
            "on (Isaac Sim's default) simulated prims return to their pre-play state, "
            "which also reverts edits made to them while the timeline was paused."
        ),
        read_only=False,
        # Reverts edits made while paused.
        destructive=True,
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
            "Advance the simulation by exactly N timeline frames and leave the timeline "
            "paused. Works from a playing, paused or stopped timeline. A frame is not one "
            "physics step: PhysX runs frame_dt / physics_dt steps per frame, both reported "
            "under physics with an estimated physics step count. The app is rate-limited "
            "(60 Hz by default), so the call's budget grows with N (up to 110 s) and a "
            "step that runs out of it stops, pauses and reports how far it got. The "
            "timeline stays paused: with /physics/resetOnStop on (Isaac Sim's default), a "
            "later stop or reset restores simulated prims to their pre-play state and "
            "reverts edits made to them in between."
        ),
        read_only=False,
    )
    async def step_isaac_simulation(
        self, num_steps: int = 1
    ) -> Dict[str, Any]:
        """
        Advance the simulation by exactly N timeline frames, then pause.

        Args:
            num_steps: Number of timeline frames to advance. Each frame runs
                as many physics steps as fit its duration (``frame_dt /
                physics_dt``). Clamped to [1, 1000].

        Returns:
            Dict with ``steps`` (frames advanced), ``steps_requested``,
            ``start_time``, ``current_time``, ``time_delta``, ``frame_dt``,
            ``physics`` (``physics_dt``, ``physics_steps_per_second``,
            ``physics_steps_per_frame``, ``physics_steps_estimated``,
            ``timeline_time_codes_per_second``, ``rate_limit_hz``),
            ``elapsed_seconds`` and ``state`` ("paused"); ``error`` when
            fewer frames advanced than requested or the budget ran out.
        """
        num_steps = max(1, min(num_steps, 1000))
        budget = self._step_budget(num_steps)
        bridge_result = await self._execute_bridge_action(
            "simulation_control",
            {"command": "step", "num_steps": num_steps, "timeout": budget},
            read_timeout=budget + 5.0,
        )
        if bridge_result is not None:
            return bridge_result

        script = _compose_script(
            """\
            import json
            import time as _time
            """,
            PHYSICS_RATES_CORE,
            f"""\
            try:
                import omni.timeline
                import omni.kit.app

                timeline = omni.timeline.get_timeline_interface()
                app = omni.kit.app.get_app()
                num_steps = {num_steps}
                budget = {budget!r}

                def _commit():
                    commit = getattr(timeline, "commit", None)
                    if callable(commit):
                        commit()

                # A paused timeline does not move on app updates and a
                # stopped one spends its first updates after play() on
                # physics warm-up, so play and count only the updates that
                # moved the time; then pause. The update budget bounds the
                # loop when time cannot advance (end of a non-looping range)
                # and the wall-clock budget when frames are slow.
                started = _time.monotonic()
                start_time = timeline.get_current_time()
                if not timeline.is_playing():
                    timeline.play()
                    _commit()
                advanced = 0
                updates = 0
                last = start_time
                out_of_time = False
                try:
                    while advanced < num_steps and updates < num_steps + 60:
                        if _time.monotonic() - started >= budget:
                            out_of_time = True
                            break
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
                time_delta = current_time - start_time
                frame_dt = time_delta / advanced if advanced else None
                elapsed = _time.monotonic() - started
                result = {{
                    "steps": advanced,
                    "steps_requested": num_steps,
                    "start_time": start_time,
                    "current_time": current_time,
                    "time_delta": time_delta,
                    "app_updates": updates,
                    "state": "paused",
                    "frame_dt": frame_dt,
                    "physics": _physics_rates(timeline, frame_dt, time_delta),
                    "elapsed_seconds": round(elapsed, 3),
                }}
                if out_of_time:
                    rate = advanced / elapsed if elapsed else 0.0
                    result["error"] = (
                        f"Stepped {{advanced}} of {{num_steps}} frames before the "
                        f"{{budget:g}}s budget ran out ({{rate:.1f}} frames/s; the app is "
                        "rate-limited). Step fewer frames per call."
                    )
                elif advanced < num_steps:
                    result["error"] = (
                        f"Timeline advanced {{advanced}} of {{num_steps}} steps in "
                        f"{{updates}} app updates; it may be at the end of a "
                        "non-looping time range."
                    )
                print(json.dumps(result))
            except Exception as e:
                print(json.dumps({{"error": f"Failed to step simulation: {{e}}"}}))
            """,
        )
        mode = self._raw_script_transport_mode
        # The script stops itself at the budget; the transport waits a little
        # longer so that report, not a transport timeout, is what comes back.
        return await self._execute_json_script(
            script, transport_mode=mode, timeout_seconds=budget + 5.0
        )

    def _step_budget(self, num_steps: int) -> float:
        """Wall-clock budget for stepping ``num_steps`` frames.

        Grows with the frame count from the default script budget up to the
        heavy one, so 1000 frames are not held to the 29 s a single frame
        gets; a longer configured timeout is kept.
        """
        configured = getattr(self._client, "script_timeout_seconds", None)
        if isinstance(configured, bool) or not isinstance(configured, (int, float)):
            configured = 0.0
        scaled = min(
            HEAVY_SCRIPT_TIMEOUT_SECONDS,
            STEP_BUDGET_BASE_SECONDS + num_steps * STEP_BUDGET_SECONDS_PER_FRAME,
        )
        return max(float(configured), scaled)

    @tool_meta(
        name="reset_isaac_simulation",
        description=(
            "Reset the simulation to initial state and time 0. Like stop, this restores "
            "simulated prims to their pre-play state (/physics/resetOnStop), reverting "
            "edits made to them while paused."
        ),
        read_only=False,
        # Reverts edits made while paused.
        destructive=True,
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
