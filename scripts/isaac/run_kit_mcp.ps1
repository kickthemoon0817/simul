# Isaac Sim MCP Server Launcher Script (Windows PowerShell)
# This script launches the Isaac Sim MCP server with proper environment setup

param(
    [string]$Host = "localhost",
    [int]$Port = 8765,
    [string]$LogLevel = "INFO",
    [switch]$Headless,
    [string]$IsaacPath = "",
    [string]$Config = "",
    [switch]$Debug,
    [switch]$Profile,
    [switch]$DryRun,
    [switch]$Help
)

# Script configuration
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

# Colors for output
$Colors = @{
    Red = "Red"
    Green = "Green"
    Yellow = "Yellow"
    Blue = "Blue"
    White = "White"
}

# Logging functions
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor $Colors.Blue
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor $Colors.Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor $Colors.Red
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor $Colors.Green
}

# Help function
function Show-Help {
    @"
Isaac Sim MCP Server Launcher (Windows)

Usage: .\run_kit_mcp.ps1 [OPTIONS]

Options:
    -Host HOST              MCP server host (default: localhost)
    -Port PORT              MCP server port (default: 8765)
    -LogLevel LEVEL         Log level (DEBUG, INFO, WARN, ERROR) (default: INFO)
    -Headless               Run in headless mode (no GUI)
    -IsaacPath PATH         Path to Isaac Sim installation
    -Config CONFIG          Path to configuration file
    -Debug                  Enable debug mode
    -Profile                Enable profiling
    -DryRun                 Show commands without executing
    -Help                   Show this help message

Environment Variables:
    ISAAC_SIM_PATH          Path to Isaac Sim installation
    MCP_SERVER_HOST         Default server host
    MCP_SERVER_PORT         Default server port
    LOG_LEVEL               Default log level

Examples:
    .\run_kit_mcp.ps1                                          # Run with defaults
    .\run_kit_mcp.ps1 -Port 9000 -Headless                   # Run headless on port 9000
    .\run_kit_mcp.ps1 -IsaacPath "C:\isaac-sim"              # Use specific Isaac Sim path
    .\run_kit_mcp.ps1 -Config "config\production.yaml"       # Use custom config
"@
}

# Show help if requested
if ($Help) {
    Show-Help
    exit 0
}

# Load environment variables from .env if it exists
$EnvFile = Join-Path $ProjectRoot ".env"
if (Test-Path $EnvFile) {
    Write-Info "Loading environment from .env file"
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match '^([^#][^=]+)=(.*)$') {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim()
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

# Override with environment variables if set
if ($env:MCP_SERVER_HOST) { $Host = $env:MCP_SERVER_HOST }
if ($env:MCP_SERVER_PORT) { $Port = [int]$env:MCP_SERVER_PORT }
if ($env:LOG_LEVEL) { $LogLevel = $env:LOG_LEVEL }

# Determine Isaac Sim path
if ($IsaacPath) {
    $IsaacSimPath = $IsaacPath
} elseif ($env:ISAAC_SIM_PATH) {
    $IsaacSimPath = $env:ISAAC_SIM_PATH
} else {
    # Try to find Isaac Sim in common Windows locations
    $CommonPaths = @(
        "$env:USERPROFILE\AppData\Local\ov\pkg\isaac_sim-*",
        "C:\Program Files\NVIDIA\isaac_sim-*",
        "C:\isaac_sim-*"
    )
    
    $IsaacSimPath = $null
    foreach ($PathPattern in $CommonPaths) {
        $FoundPaths = Get-ChildItem -Path (Split-Path $PathPattern -Parent) -Directory -Name (Split-Path $PathPattern -Leaf) -ErrorAction SilentlyContinue
        if ($FoundPaths) {
            $IsaacSimPath = Join-Path (Split-Path $PathPattern -Parent) $FoundPaths[0]
            break
        }
    }
    
    if (-not $IsaacSimPath) {
        Write-Error "Isaac Sim installation not found!"
        Write-Error "Please set ISAAC_SIM_PATH environment variable or use -IsaacPath parameter"
        exit 1
    }
}

# Validate Isaac Sim path
if (-not (Test-Path $IsaacSimPath)) {
    Write-Error "Isaac Sim directory not found: $IsaacSimPath"
    exit 1
}

$PythonBat = Join-Path $IsaacSimPath "python.bat"
if (-not (Test-Path $PythonBat)) {
    Write-Error "Isaac Sim python.bat not found in: $IsaacSimPath"
    exit 1
}

Write-Info "Using Isaac Sim installation: $IsaacSimPath"

# Build command arguments
$CmdArgs = @()
$CmdArgs += "--host", $Host
$CmdArgs += "--port", $Port.ToString()
$CmdArgs += "--log-level", $LogLevel

if ($Headless) {
    $CmdArgs += "--headless"
}

if ($Debug) {
    $CmdArgs += "--debug"
}

if ($Profile) {
    $CmdArgs += "--profile"
}

if ($Config) {
    if (-not (Test-Path $Config)) {
        Write-Error "Configuration file not found: $Config"
        exit 1
    }
    $CmdArgs += "--config", $Config
}

# Build full command
$ModuleCmd = "-m simul_mcp.cli.main server"
$FullCmd = "& `"$PythonBat`" $ModuleCmd $($CmdArgs -join ' ')"

# Show configuration
Write-Info "Isaac Sim MCP Server Configuration:"
Write-Info "  Host: $Host"
Write-Info "  Port: $Port"
Write-Info "  Log Level: $LogLevel"
Write-Info "  Headless: $Headless"
Write-Info "  Debug: $Debug"
Write-Info "  Profile: $Profile"
Write-Info "  Isaac Sim Path: $IsaacSimPath"
if ($Config) {
    Write-Info "  Config File: $Config"
}

# Execute or show command
if ($DryRun) {
    Write-Info "Dry run - would execute:"
    Write-Host $FullCmd
} else {
    Write-Info "Starting Isaac Sim MCP Server..."
    Write-Info "Command: $FullCmd"
    
    # Change to project root directory
    Set-Location $ProjectRoot
    
    # Execute the command
    try {
        Invoke-Expression $FullCmd
    } catch {
        Write-Error "Failed to start Isaac Sim MCP Server: $_"
        exit 1
    }
}
