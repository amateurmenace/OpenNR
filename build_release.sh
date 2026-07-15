#!/bin/bash
# OpenNR — build a macOS release: universal .ofx bundle, double-clickable .pkg
# installer, and a zip fallback with a .command install script.
set -euo pipefail
cd "$(dirname "$0")"

VERSION="2.1.0"

echo "== building plugin =="
make -C plugin clean >/dev/null
make -C plugin

echo "== staging =="
rm -rf release
STAGE="release/stage"
mkdir -p "$STAGE"
cp -R plugin/OpenNR.ofx.bundle "$STAGE/"

echo "== signing =="
# Prefer a Developer ID Application identity when one is present; fall back to
# ad-hoc. (Notarization additionally requires --timestamp and hardened runtime.)
DEV_ID=$(security find-identity -v -p codesigning 2>/dev/null | grep -o '"Developer ID Application: [^"]*"' | head -1 | tr -d '"') || true
if [ -n "${DEV_ID:-}" ]; then
    echo "signing with: $DEV_ID"
    codesign --force --deep --timestamp --options runtime --sign "$DEV_ID" "$STAGE/OpenNR.ofx.bundle"
else
    echo "no Developer ID found — ad-hoc signing"
    codesign --force --deep --sign - "$STAGE/OpenNR.ofx.bundle"
fi
codesign --verify --deep "$STAGE/OpenNR.ofx.bundle"

echo "== sanity: minimum macOS of the shipped binary =="
otool -arch arm64 -l "$STAGE/OpenNR.ofx.bundle/Contents/MacOS/OpenNR.ofx" | grep -m1 minos

echo "== pkg installer =="
PKG_STAGE="release/pkgroot"
mkdir -p "$PKG_STAGE"
cp -R "$STAGE/OpenNR.ofx.bundle" "$PKG_STAGE/"
pkgbuild --root "$PKG_STAGE" \
         --identifier org.opennr.plugin \
         --version "$VERSION" \
         --install-location "/Library/OFX/Plugins" \
         "release/OpenNR-$VERSION-macOS-unsigned.pkg"

# Sign the installer too when a Developer ID Installer identity exists,
# then notarize when a notarytool keychain profile named "opennr-notary"
# has been stored (xcrun notarytool store-credentials opennr-notary ...).
INST_ID=$(security find-identity -v 2>/dev/null | grep -o '"Developer ID Installer: [^"]*"' | head -1 | tr -d '"') || true
if [ -n "${INST_ID:-}" ]; then
    echo "signing pkg with: $INST_ID"
    productsign --sign "$INST_ID" "release/OpenNR-$VERSION-macOS-unsigned.pkg" "release/OpenNR-$VERSION-macOS.pkg"
    rm "release/OpenNR-$VERSION-macOS-unsigned.pkg"
else
    echo "no Developer ID Installer identity — shipping unsigned pkg (payload is still signed)"
    mv "release/OpenNR-$VERSION-macOS-unsigned.pkg" "release/OpenNR-$VERSION-macOS.pkg"
fi

if xcrun notarytool history --keychain-profile "opennr-notary" >/dev/null 2>&1; then
    echo "notarizing (profile: opennr-notary)..."
    xcrun notarytool submit "release/OpenNR-$VERSION-macOS.pkg" --keychain-profile "opennr-notary" --wait
    xcrun stapler staple "release/OpenNR-$VERSION-macOS.pkg"
else
    echo "no notarytool profile 'opennr-notary' — skipping notarization"
fi

echo "== zip fallback =="
cp "installer/Install OpenNR (macOS).command" "$STAGE/"
cp README.md LICENSE "$STAGE/"
(cd "$STAGE" && zip -qry "../OpenNR-$VERSION-macOS.zip" .)

echo
echo "Release artifacts:"
ls -la release/*.pkg release/*.zip
