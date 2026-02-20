#!/bin/bash

# Development Code Formatting Script
# Runs isort and black on the codebase for consistent formatting

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

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
Development Code Formatting Script

Usage: $0 [OPTIONS]

Options:
    -h, --help      Show this help message
    -c, --check     Check formatting without making changes
    -v, --verbose   Verbose output
    --isort-only    Run only isort
    --black-only    Run only black
    --diff          Show diff of changes

Examples:
    $0                  # Format all code
    $0 --check          # Check formatting without changes
    $0 --diff           # Show what would be changed
    $0 --isort-only     # Run only import sorting
EOF
}

# Parse command line arguments
CHECK_ONLY=false
VERBOSE=false
ISORT_ONLY=false
BLACK_ONLY=false
SHOW_DIFF=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -c|--check)
            CHECK_ONLY=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --isort-only)
            ISORT_ONLY=true
            shift
            ;;
        --black-only)
            BLACK_ONLY=true
            shift
            ;;
        --diff)
            SHOW_DIFF=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Change to project root
cd "$PROJECT_ROOT"

# Directories to format
DIRS_TO_FORMAT=(
    "src/"
    "tests/"
    "cli/"
    "examples/"
)

# Filter directories that exist
EXISTING_DIRS=()
for dir in "${DIRS_TO_FORMAT[@]}"; do
    if [[ -d "$dir" ]]; then
        EXISTING_DIRS+=("$dir")
    fi
done

if [[ ${#EXISTING_DIRS[@]} -eq 0 ]]; then
    log_warn "No directories found to format"
    exit 0
fi

log_info "Formatting directories: ${EXISTING_DIRS[*]}"

# Check if tools are installed
check_tool() {
    local tool=$1
    if ! command -v "$tool" &> /dev/null; then
        log_error "$tool is not installed. Please install it with: pip install $tool"
        return 1
    fi
    return 0
}

# Build command arguments
ISORT_ARGS=()
BLACK_ARGS=()

if [[ "$CHECK_ONLY" == true ]]; then
    ISORT_ARGS+=("--check-only")
    BLACK_ARGS+=("--check")
fi

if [[ "$SHOW_DIFF" == true ]]; then
    ISORT_ARGS+=("--diff")
    BLACK_ARGS+=("--diff")
fi

if [[ "$VERBOSE" == true ]]; then
    ISORT_ARGS+=("--verbose")
    BLACK_ARGS+=("--verbose")
fi

# Run isort
if [[ "$BLACK_ONLY" != true ]]; then
    if check_tool "isort"; then
        log_info "Running isort..."
        if [[ "$VERBOSE" == true ]]; then
            log_info "isort command: isort ${ISORT_ARGS[*]} ${EXISTING_DIRS[*]}"
        fi
        
        if isort "${ISORT_ARGS[@]}" "${EXISTING_DIRS[@]}"; then
            log_success "isort completed successfully"
        else
            log_error "isort failed"
            exit 1
        fi
    else
        exit 1
    fi
fi

# Run black
if [[ "$ISORT_ONLY" != true ]]; then
    if check_tool "black"; then
        log_info "Running black..."
        if [[ "$VERBOSE" == true ]]; then
            log_info "black command: black ${BLACK_ARGS[*]} ${EXISTING_DIRS[*]}"
        fi
        
        if black "${BLACK_ARGS[@]}" "${EXISTING_DIRS[@]}"; then
            log_success "black completed successfully"
        else
            log_error "black failed"
            exit 1
        fi
    else
        exit 1
    fi
fi

if [[ "$CHECK_ONLY" == true ]]; then
    log_success "Code formatting check completed"
else
    log_success "Code formatting completed"
fi
