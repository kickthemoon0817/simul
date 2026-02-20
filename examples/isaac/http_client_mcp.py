#!/usr/bin/env python3
"""
HTTP Client Example for Isaac Sim MCP Server.

This example demonstrates how to interact with the Isaac Sim MCP Server
using HTTP requests (when running with SSE transport).
"""

import json
import asyncio
import aiohttp
import argparse
from typing import Dict, Any, Optional, List
from pathlib import Path


class MCPClient:
    """Simple MCP client for HTTP communication."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize MCP client.
        
        Args:
            base_url: Base URL of the MCP server
        """
        self.base_url = base_url.rstrip('/')
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call an MCP tool.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            
        Returns:
            Tool response
        """
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        url = f"{self.base_url}/tools/{tool_name}"
        
        try:
            async with self.session.post(url, json=arguments) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}: {error_text}",
                        "error_type": "HTTPError"
                    }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": "NetworkError"
            }
    
    async def get_server_info(self) -> Dict[str, Any]:
        """Get server information."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        url = f"{self.base_url}/info"
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"HTTP {response.status}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def list_tools(self) -> List[str]:
        """List available tools."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        url = f"{self.base_url}/tools"
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("tools", [])
                else:
                    return []
        except Exception as e:
            print(f"Error listing tools: {e}")
            return []


async def demonstrate_usd_operations(client: MCPClient, usd_file: str):
    """
    Demonstrate USD operations using the MCP client.
    
    Args:
        client: MCP client instance
        usd_file: Path to USD file
    """
    print("USD Operations Demo")
    print("=" * 50)
    
    # Load USD file
    print(f"Loading USD file: {usd_file}")
    result = await client.call_tool("load_usd_file", {"file_path": usd_file})
    
    if not result.get("success", True):
        print(f"Error loading USD file: {result.get('error', 'Unknown error')}")
        return
    
    stage_id = result.get("stage_id")
    if not stage_id:
        print("No stage ID returned")
        return
    
    print(f"✓ Loaded stage: {stage_id}")
    
    # Display stage info
    print(f"\nStage Information:")
    print(f"  Up Axis: {result.get('up_axis', 'Unknown')}")
    print(f"  Meters per Unit: {result.get('meters_per_unit', 'Unknown')}")
    print(f"  Total Prims: {result.get('total_prims', 'Unknown')}")
    print(f"  Has Animation: {result.get('has_animation', 'Unknown')}")
    
    # Search for mesh prims
    print(f"\nSearching for mesh prims...")
    search_result = await client.call_tool("search_prims", {
        "stage_id": stage_id,
        "search_type": "by_type",
        "query": "Mesh"
    })
    
    if search_result.get("success", True):
        mesh_prims = search_result.get("results", [])
        print(f"✓ Found {len(mesh_prims)} mesh prims")
        
        # Analyze first few meshes
        for i, mesh_path in enumerate(mesh_prims[:3]):
            print(f"\nAnalyzing mesh {i+1}: {mesh_path}")
            
            # Get mesh info
            mesh_result = await client.call_tool("get_mesh_info", {
                "stage_id": stage_id,
                "prim_path": mesh_path
            })
            
            if mesh_result.get("success", True):
                print(f"  Vertices: {mesh_result.get('vertex_count', 'Unknown'):,}")
                print(f"  Faces: {mesh_result.get('face_count', 'Unknown'):,}")
                print(f"  Has Normals: {mesh_result.get('has_normals', 'Unknown')}")
                print(f"  Has UVs: {mesh_result.get('has_uvs', 'Unknown')}")
                print(f"  Surface Area: {mesh_result.get('surface_area', 'Unknown')}")
                print(f"  Is Closed: {mesh_result.get('is_closed', 'Unknown')}")
            
            # Get bounding box
            bbox_result = await client.call_tool("get_bounding_box", {
                "stage_id": stage_id,
                "prim_path": mesh_path,
                "world_space": True
            })
            
            if bbox_result.get("success", True) and bbox_result.get("bbox"):
                bbox = bbox_result["bbox"]
                min_pt = bbox.get("min", [0, 0, 0])
                max_pt = bbox.get("max", [0, 0, 0])
                size = [max_pt[i] - min_pt[i] for i in range(3)]
                print(f"  Bounding Box Size: [{size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}]")
    else:
        print(f"Error searching for meshes: {search_result.get('error', 'Unknown error')}")
    
    # Get scene summary
    print(f"\nGenerating scene summary...")
    summary_result = await client.call_tool("summarize_scene", {
        "stage_id": stage_id,
        "include_meshes": True,
        "format": "text"
    })
    
    if summary_result.get("success", True):
        digest = summary_result.get("digest")
        if digest:
            print("Scene Summary:")
            print("-" * 20)
            print(digest)
    else:
        print(f"Error generating summary: {summary_result.get('error', 'Unknown error')}")


async def demonstrate_isaac_operations(client: MCPClient):
    """
    Demonstrate Isaac Sim specific operations.
    
    Args:
        client: MCP client instance
    """
    print("\nIsaac Sim Operations Demo")
    print("=" * 50)
    
    # Get simulation status
    print("Getting simulation status...")
    status_result = await client.call_tool("get_simulation_status", {})
    
    if status_result.get("success", True):
        print(f"✓ World Initialized: {status_result.get('world_initialized', 'Unknown')}")
        print(f"  Is Playing: {status_result.get('is_playing', 'Unknown')}")
        print(f"  Current Time: {status_result.get('current_time', 'Unknown')}")
    else:
        print(f"Error getting simulation status: {status_result.get('error', 'Unknown error')}")
    
    # Try to capture viewport
    print("\nAttempting viewport capture...")
    capture_result = await client.call_tool("capture_viewport", {
        "width": 512,
        "height": 512,
        "format": "png",
        "save_to_file": True
    })
    
    if capture_result.get("success", True):
        file_path = capture_result.get("file_path")
        print(f"✓ Viewport captured: {file_path}")
        print(f"  Size: {capture_result.get('width')}x{capture_result.get('height')}")
        print(f"  Format: {capture_result.get('format')}")
    else:
        print(f"Viewport capture failed: {capture_result.get('error', 'Unknown error')}")
        print("(This is expected if not running in Isaac Sim)")
    
    # Try simulation control
    print("\nTesting simulation control...")
    control_result = await client.call_tool("control_simulation", {
        "action": "reset"
    })
    
    if control_result.get("success", True):
        print(f"✓ Simulation reset successful")
    else:
        print(f"Simulation control failed: {control_result.get('error', 'Unknown error')}")
        print("(This is expected if not running in Isaac Sim)")


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="HTTP client example for Isaac Sim MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python http_client_mcp.py --server http://localhost:8000
  python http_client_mcp.py --usd-file scene.usd
  python http_client_mcp.py --server http://localhost:8000 --usd-file scene.usd --isaac-demo
        """
    )
    
    parser.add_argument(
        "--server",
        default="http://localhost:8000",
        help="MCP server URL (default: http://localhost:8000)"
    )
    
    parser.add_argument(
        "--usd-file",
        help="USD file to analyze (optional)"
    )
    
    parser.add_argument(
        "--isaac-demo",
        action="store_true",
        help="Run Isaac Sim specific demos"
    )
    
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="List available tools and exit"
    )
    
    args = parser.parse_args()
    
    print(f"Connecting to MCP server: {args.server}")
    
    async with MCPClient(args.server) as client:
        # Get server info
        print("Getting server information...")
        server_info = await client.get_server_info()
        
        if "error" in server_info:
            print(f"Error connecting to server: {server_info['error']}")
            print("Make sure the MCP server is running with SSE transport")
            return
        
        print("✓ Connected to server")
        print(f"Server capabilities: {server_info.get('capabilities', 'Unknown')}")
        
        # List tools if requested
        if args.list_tools:
            print("\nAvailable tools:")
            tools = await client.list_tools()
            for tool in tools:
                print(f"  - {tool}")
            return
        
        # Run USD demo if file provided
        if args.usd_file:
            usd_path = Path(args.usd_file)
            if not usd_path.exists():
                print(f"Error: USD file not found: {args.usd_file}")
                return
            
            await demonstrate_usd_operations(client, str(usd_path.resolve()))
        
        # Run Isaac Sim demo if requested
        if args.isaac_demo:
            await demonstrate_isaac_operations(client)
        
        if not args.usd_file and not args.isaac_demo:
            print("\nNo demos specified. Use --usd-file or --isaac-demo to run demonstrations.")
            print("Use --list-tools to see available tools.")


if __name__ == "__main__":
    asyncio.run(main())
