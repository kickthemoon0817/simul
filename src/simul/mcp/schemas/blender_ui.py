"""Typed inputs for named Blender UI actions."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AgentControl = Literal[
    "inspect",
    "observe",
    "move_cursor",
    "clear_cursor",
    "open_menu",
    "select_object",
    "set_tool",
    "show_properties",
    "set_property",
    "isolate_workspace",
]


class BlenderUIRequest(BaseModel):
    """Action-specific validation also runs inside the attached Blender process."""

    agent_control: AgentControl
    target: str | None = None
    area_id: str | None = None
    position: list[float] | None = Field(None, min_length=2, max_length=2)
    value: list[float] | None = Field(None, min_length=3, max_length=3)
    agent_id: str = Field("agent", min_length=1, max_length=64)


class BlenderUIResponse(BaseModel):
    """UI identity plus action-specific observations from Blender."""

    model_config = ConfigDict(extra="allow")
    success: bool
    agent_control: AgentControl
    execution_method: Literal["blender_ui_api"]
    window_id: str
