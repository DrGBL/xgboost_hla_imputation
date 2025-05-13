#!/bin/bash
set -euo pipefail

# Default values
CONTAINER_NAME="xgboost_container"
RECIPE_NAME="xgboost.def"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_PATH="${SCRIPT_DIR}/${RECIPE_NAME}"
SANDBOX_MODE=false
DRY_RUN=false

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        --sandbox)
            SANDBOX_MODE=true
            ;;
        --dry)
            DRY_RUN=true
            ;;
        *)
            echo "❌ Unknown option: $arg"
            echo "Usage: $0 [--sandbox] [--dry]"
            exit 1
            ;;
    esac
done

# Check if recipe file exists in the same directory
if [ ! -f "$RECIPE_PATH" ]; then
    echo "❌ Error: recipe file '$RECIPE_NAME' not found in the same directory as this script ($SCRIPT_DIR)"
    exit 1
fi

# Determine output name and build option
if $SANDBOX_MODE; then
    OUTPUT_NAME="${CONTAINER_NAME}_SANDBOX"
    BUILD_OPTION="--sandbox"
else
    OUTPUT_NAME="${CONTAINER_NAME}.sif"
    BUILD_OPTION=""
fi

# Assemble build command
CMD="apptainer build --fakeroot $BUILD_OPTION $OUTPUT_NAME $RECIPE_PATH"

# Show and (maybe) run the command
echo "🔧 Build command:"
echo "$CMD"

if $DRY_RUN; then
    echo "💡 Dry run mode enabled — command not executed."
else
    echo "🚀 Building container..."
    eval "$CMD"
    echo "✅ Build complete: $OUTPUT_NAME"
fi
