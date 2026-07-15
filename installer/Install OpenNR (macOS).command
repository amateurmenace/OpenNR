#!/bin/bash
# OpenNR installer (fallback for the .pkg) — double-click to install the
# OpenNR.ofx.bundle sitting next to this script into /Library/OFX/Plugins.
DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -d "$DIR/OpenNR.ofx.bundle" ]; then
    echo "OpenNR.ofx.bundle not found next to this script."
    echo "Keep this script in the same folder as OpenNR.ofx.bundle and run it again."
    read -r -p "Press Return to close..."
    exit 1
fi

echo "Installing OpenNR into /Library/OFX/Plugins (you may be asked for your password)..."
osascript -e "do shell script \"mkdir -p '/Library/OFX/Plugins' && rm -rf '/Library/OFX/Plugins/OpenNR.ofx.bundle' && cp -Rf '$DIR/OpenNR.ofx.bundle' '/Library/OFX/Plugins/'\" with administrator privileges"

if [ -d "/Library/OFX/Plugins/OpenNR.ofx.bundle" ]; then
    echo
    echo "Installed. Restart DaVinci Resolve, then find it under:"
    echo "  Color page  ->  Effects  ->  OpenFX  ->  Filters  ->  OpenNR  ->  OpenNR Denoise"
    echo "  Edit page   ->  Effects Library  ->  OpenFX  ->  Filters  ->  OpenNR"
else
    echo "Installation failed."
fi
read -r -p "Press Return to close..."
