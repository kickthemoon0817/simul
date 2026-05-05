"""Isaac instance discovery and routing schemas."""

from typing import List, Optional

from pydantic import BaseModel, Field


class IsaacInstanceInfo(BaseModel):
    """Information about a discovered Isaac Sim instance."""

    name: str = Field(..., description="Instance identifier")
    host: str = Field(..., description="TCP socket host")
    port: int = Field(..., description="TCP socket port")
    reachable: bool = Field(..., description="Whether the instance responded to ping")
    active: bool = Field(False, description="Whether this is the currently active instance")
    stage_url: Optional[str] = Field(None, description="URL of the loaded stage")
    up_axis: Optional[str] = Field(None, description="Stage up axis")
    prim_count: Optional[int] = Field(None, description="Total prim count on stage")
    is_playing: Optional[bool] = Field(None, description="Whether simulation is playing")


class ListIsaacInstancesResponse(BaseModel):
    """Response from list_isaac_instances tool."""

    success: bool = Field(True, description="Whether discovery completed")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    instances: List[IsaacInstanceInfo] = Field(default_factory=list, description="Discovered instances")
    active_instance: Optional[str] = Field(None, description="Name of the currently active instance")
    total_discovered: int = Field(0, description="Number of reachable instances found")


class SetActiveInstanceResponse(BaseModel):
    """Response from set_active_isaac_instance tool."""

    success: bool = Field(..., description="Whether the switch succeeded")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    active_instance: str = Field(..., description="Name of the now-active instance")
    address: str = Field(..., description="host:port of the now-active instance")
    message: str = Field("", description="Status message")

__all__ = [
    "IsaacInstanceInfo",
    "ListIsaacInstancesResponse",
    "SetActiveInstanceResponse",
]
