#!/bin/bash

# Isaac Sim MCP Server Launcher Script (Linux/macOS)
# This script launches the Isaac Sim MCP server with proper environment setup

set -e  # Exit on any error

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default configuration
DEFAULT_HOST="localhost"
DEFAULT_PORT="8765"
DEFAULT_LOG_LEVEL="INFO"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Isaac Sim MCP Server Launcher

Usage: $0 [OPTIONS]

Options:
    -h, --help              Show this help message
    -p, --port PORT         MCP server port (default: $DEFAULT_PORT)
    -H, --host HOST         MCP server host (default: $DEFAULT_HOST)
    -l, --log-level LEVEL   Log level (DEBUG, INFO, WARN, ERROR) (default: $DEFAULT_LOG_LEVEL)
    --headless              Run in headless mode (no GUI)
    --isaac-path PATH       Path to Isaac Sim installation
    --config CONFIG         Path to configuration file
    --debug                 Enable debug mode
    --profile               Enable profiling
    --dry-run               Show commands without executing

Environment Variables:
    ISAAC_SIM_PATH          Path to Isaac Sim installation
    MCP_SERVER_HOST         Default server host
    MCP_SERVER_PORT         Default server port
    LOG_LEVEL               Default log level

Examples:
    $0                                          # Run with defaults
    $0 --port 9000 --headless                 # Run headless on port 9000
    $0 --isaac-path /opt/isaac-sim             # Use specific Isaac Sim path
    $0 --config config/production.yaml        # Use custom config
EOF
}

# Parse command line arguments
HOST="$DEFAULT_HOST"
PORT="$DEFAULT_PORT"
LOG_LEVEL="$DEFAULT_LOG_LEVEL"
HEADLESS=false
DEBUG=false
PROFILE=false
DRY_RUN=false
ISAAC_PATH=""
CONFIG_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -H|--host)
            HOST="$2"
            shift 2
            ;;
        -l|--log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --headless)
            HEADLESS=true
            shift
            ;;
        --isaac-path)
            ISAAC_PATH="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        --profile)
            PROFILE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Load environment variables from .env if it exists
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    log_info "Loading environment from .env file"
    set -a  # automatically export all variables
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Override with environment variables if set
HOST="${MCP_SERVER_HOST:-$HOST}"
PORT="${MCP_SERVER_PORT:-$PORT}"
LOG_LEVEL="${LOG_LEVEL:-$LOG_LEVEL}"

# Determine Isaac Sim path
if [[ -n "$ISAAC_PATH" ]]; then
    ISAAC_SIM_PATH="$ISAAC_PATH"
elif [[ -z "$ISAAC_SIM_PATH" ]]; then
    # Try to find Isaac Sim in common locations
    COMMON_PATHS=(
        "$HOME/.local/share/ov/pkg/isaac_sim-*"
        "/opt/nvidia/isaac_sim-*"
        "/usr/local/isaac_sim-*"
    )
    
    for path_pattern in "${COMMON_PATHS[@]}"; do
        if ls $path_pattern 1> /dev/null 2>&1; then
            ISAAC_SIM_PATH=$(ls -d $path_pattern | head -n1)
            break
        fi
    done
    
    if [[ -z "$ISAAC_SIM_PATH" ]]; then
        log_error "Isaac Sim installation not found!"
        log_error "Please set ISAAC_SIM_PATH environment variable or use --isaac-path option"
        exit 1
    fi
fi

# Validate Isaac Sim path
if [[ ! -d "$ISAAC_SIM_PATH" ]]; then
    log_error "Isaac Sim directory not found: $ISAAC_SIM_PATH"
    exit 1
fi

if [[ ! -f "$ISAAC_SIM_PATH/python.sh" ]]; then
    log_error "Isaac Sim python.sh not found in: $ISAAC_SIM_PATH"
    exit 1
fi

log_info "Using Isaac Sim installation: $ISAAC_SIM_PATH"

# Build command arguments
CMD_ARGS=()
CMD_ARGS+=("--host" "$HOST")
CMD_ARGS+=("--port" "$PORT")
CMD_ARGS+=("--log-level" "$LOG_LEVEL")

if [[ "$HEADLESS" == true ]]; then
    CMD_ARGS+=("--headless")
fi

if [[ "$DEBUG" == true ]]; then
    CMD_ARGS+=("--debug")
fi

if [[ "$PROFILE" == true ]]; then
    CMD_ARGS+=("--profile")
fi

if [[ -n "$CONFIG_FILE" ]]; then
    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_error "Configuration file not found: $CONFIG_FILE"
        exit 1
    fi
    CMD_ARGS+=("--config" "$CONFIG_FILE")
fi

# Build full command
PYTHON_CMD="$ISAAC_SIM_PATH/python.sh"
MODULE_CMD="-m simul_mcp.cli.main server"
FULL_CMD="$PYTHON_CMD $MODULE_CMD ${CMD_ARGS[*]}"

# Show configuration
log_info "Isaac Sim MCP Server Configuration:"
log_info "  Host: $HOST"
log_info "  Port: $PORT"
log_info "  Log Level: $LOG_LEVEL"
log_info "  Headless: $HEADLESS"
log_info "  Debug: $DEBUG"
log_info "  Profile: $PROFILE"
log_info "  Isaac Sim Path: $ISAAC_SIM_PATH"
if [[ -n "$CONFIG_FILE" ]]; then
    log_info "  Config File: $CONFIG_FILE"
fi

# Execute or show command
if [[ "$DRY_RUN" == true ]]; then
    log_info "Dry run - would execute:"
    echo "$FULL_CMD"
else
    log_info "Starting Isaac Sim MCP Server..."
    log_info "Command: $FULL_CMD"
    
    # Change to project root directory
    cd "$PROJECT_ROOT"
    
    # Execute the command
    exec $FULL_CMD
fi
