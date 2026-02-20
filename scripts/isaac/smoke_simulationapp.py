"""Smoke test for Isaac Sim MCP integration via SimulationApp."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def _load_simulation_app():
    try:
        from isaacsim.simulation_app import SimulationApp

        return SimulationApp
    except ImportError:
        try:
            from omni.isaac.kit import SimulationApp

            return SimulationApp
        except ImportError:
            from isaacsim import SimulationApp

            return SimulationApp


def main() -> int:
    SimulationApp = _load_simulation_app()
    app = SimulationApp({"headless": True, "width": 1280, "height": 720})

    try:
        repo_root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(repo_root / "src"))

        log_path = Path("/tmp/simul_mcp_smoke.log")
        logger = logging.getLogger("simul_mcp.smoke")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.addHandler(stream_handler)
            logger.propagate = False

        try:
            import carb
        except ImportError:
            carb = None

        def log(message: str) -> None:
            logger.info(message)
            if carb:
                carb.log_warn(message)

        from simul_mcp.adapters import IsaacRuntimeAdapter, is_isaac_available

        log(f"Logging to {log_path}")
        log(f"Isaac runtime available: {is_isaac_available()}")

        adapter = IsaacRuntimeAdapter()
        with adapter.create_session() as session:
            world_ok = session.initialize_world()
            log(f"World initialized: {world_ok}")

            viewport_info = session.get_viewport_info()
            camera_info = session.get_camera_info()
            sim_status = session.get_simulation_status()

            log(f"Viewport info: {viewport_info}")
            log(f"Camera info: {camera_info}")
            log(f"Simulation status: {sim_status}")

        return 0
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
