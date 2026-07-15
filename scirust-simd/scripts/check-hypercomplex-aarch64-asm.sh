#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "ERREUR : ce contrôle est destiné à AArch64."
    exit 1
fi

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

RUSTFLAGS="-C target-cpu=native" \
cargo rustc -p scirust-simd \
    --features portable-simd \
    --release \
    --example hypercomplex_asm_probe \
    -- --emit=asm

ASM="$(find target/release/examples \
    -maxdepth 1 \
    -type f \
    -name 'hypercomplex_asm_probe-*.s' \
    -printf '%T@ %p\n' |
    sort -n |
    tail -n1 |
    cut -d' ' -f2-)"

if [[ -z "$ASM" || ! -f "$ASM" ]]; then
    echo "ERREUR : assembleur du probe introuvable."
    exit 1
fi

OUT_DIR="${TMPDIR:-/tmp}/scirust-hypercomplex-asm"
mkdir -p "$OUT_DIR"

extract_function() {
    local symbol="$1"
    local output="$2"

    awk -v symbol="${symbol}:" '
        $0 == symbol { capture = 1 }
        capture { print }
        capture && /^\.Lfunc_end[0-9]+:/ { exit }
    ' "$ASM" > "$output"

    if [[ ! -s "$output" ]]; then
        echo "ERREUR : symbole $symbol introuvable dans $ASM."
        exit 1
    fi
}

OCT_ASM="$OUT_DIR/octonion-aarch64.s"
SED_ASM="$OUT_DIR/sedenion-aarch64.s"

extract_function "scirust_octonion_mul_probe" "$OCT_ASM"
extract_function "scirust_sedenion_mul_probe" "$SED_ASM"

echo "Assembleur source : $ASM"
echo

echo "=== OCTONION : accès à la pile ==="
if grep -nE '\[(sp|x29)(,|])|sub[[:space:]]+sp|add[[:space:]]+sp' "$OCT_ASM"; then
    echo "ERREUR : le noyau octonionique isolé utilise la pile."
    exit 1
else
    echo "Aucun accès à la pile."
fi

echo
echo "=== SÉDÉNION : accès à la pile ==="
grep -nE '\[(sp|x29)(,|])|sub[[:space:]]+sp|add[[:space:]]+sp' "$SED_ASM" || true

# AAPCS64 impose la préservation de la moitié basse d8-d15.
# Les sauvegardes/restaurations de d8-d15 en prologue/épilogue sont admises.
# Tout transfert qN/sN vers ou depuis sp révèle en revanche un temporaire
# vectoriel matérialisé sur la pile.
echo
echo "=== SÉDÉNION : spills vectoriels complets qN/sN ==="
if grep -nE '\b(ldp|stp|ldr|str|ldur|stur)\b.*\b(q[0-9]+|s[0-9]+)\b.*\[(sp|x29)' "$SED_ASM"; then
    echo "ERREUR : spill/reload vectoriel complet détecté."
    exit 1
else
    echo "Aucun spill/reload qN ou sN ; seules les sauvegardes ABI dN sont admises."
fi

echo
echo "Validation AArch64 réussie."
echo "Octonion : noyau isolé sans pile."
echo "Sédénion : aucun spill vectoriel complet dans le calcul isolé."
