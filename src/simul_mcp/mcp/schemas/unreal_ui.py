"""Bounded editor controls available even with arbitrary scripting disabled."""

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class UnrealUIRequest(BaseModel):
    agent_control: Literal[
        "inspect", "select_actor", "clear_selection", "set_property",
        "pilot_actor", "eject_actor", "set_game_view",
    ]
    target: Optional[str] = Field(None, description="Actor path or unique label; defaults to the sole selected actor")
    property_name: Optional[Literal["location", "rotation", "scale"]] = None
    value: Optional[List[float]] = Field(None, description="Three values; cm for location, degrees for rotation")
    enabled: Optional[bool] = None


class UnrealUIResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    success: bool
    agent_control: Optional[str] = None
