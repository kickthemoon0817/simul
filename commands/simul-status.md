---
description: Check simul-mcp server and runtime connectivity
allowed-tools: mcp__simul__ping_isaac, mcp__simul__list_isaac_instances
---

Check Isaac Sim server connectivity and discover all running instances.

1. Call `ping_isaac` to verify Isaac Sim is reachable. Note the response time and any error message.

2. Call `list_isaac_instances` to enumerate all running instances.

3. Format the results as a concise status table with these columns:
   - Instance name
   - Host:port
   - Reachable (yes/no)
   - Active (mark the currently active instance with `*`)
   - Stage URL loaded in that instance
   - Total prim count

   Example table format:
   ```
   * default   localhost:8226   reachable   omniverse://localhost/scenes/test.usd   1,243 prims
     remote    10.0.0.5:8226   reachable   omniverse://server/envs/factory.usd     4,891 prims
   ```

4. If `ping_isaac` fails or no instances are found:
   - Show a clear ERROR status
   - Suggest: "Ensure Isaac Sim is running and the VS Code / Embedded Script extension is active on port 8226. Check the Extension Manager for `omni.isaac.mcp_server` or equivalent."

5. After the table, print a one-line summary such as:
   `2 instances found, 2 reachable, active: default`
