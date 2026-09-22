#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${V888_DATA_ROOT:-$HOME/datasets/banc_v888}"
BASE_URL="https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888"
ROOT_URL="https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome"

mkdir -p "$DATA_ROOT"

remote_size() {
  local url="$1"
  curl -fsSIL --retry 5 --retry-delay 3 "$url"     | awk 'BEGIN{IGNORECASE=1} /^content-length:/ {gsub("\r","",$2); size=$2} END{print size}'
}

download_one() {
  local url="$1"
  local dest="$2"
  local label="$3"
  local expected local_size tmp

  expected="$(remote_size "$url" || true)"
  if [[ -z "$expected" ]]; then
    echo "ERROR: cannot resolve remote size for $label: $url" >&2
    return 1
  fi

  if [[ -f "$dest" ]]; then
    local_size="$(stat -c '%s' "$dest")"
    if [[ "$local_size" == "$expected" ]]; then
      echo "SKIP $label: already complete ($local_size bytes)"
      return 0
    fi
  fi

  tmp="${dest}.part"
  if [[ -f "$dest" && ! -f "$tmp" ]]; then
    mv "$dest" "$tmp"
  fi

  echo "DOWNLOAD $label"
  echo "  source: $url"
  echo "  target: $dest"
  echo "  expected_bytes: $expected"
  curl     --fail     --location     --retry 10     --retry-all-errors     --retry-delay 5     --connect-timeout 30     --continue-at -     --output "$tmp"     "$url"

  local_size="$(stat -c '%s' "$tmp")"
  if [[ "$local_size" != "$expected" ]]; then
    echo "ERROR: size mismatch for $label: got $local_size expected $expected" >&2
    return 1
  fi
  mv "$tmp" "$dest"
}

download_optional() {
  local url="$1"
  local dest="$2"
  local label="$3"
  if remote_size "$url" >/dev/null 2>&1; then
    download_one "$url" "$dest" "$label"
  else
    echo "OPTIONAL_MISSING $label: $url"
  fi
}

download_one "$BASE_URL/banc_888_meta.feather"   "$DATA_ROOT/banc_888_meta.feather" "metadata"

download_one "$BASE_URL/banc_888_metrics.feather"   "$DATA_ROOT/banc_888_metrics.feather" "metrics"

download_one "$BASE_URL/banc_888_edgelist_simple_v3.feather"   "$DATA_ROOT/banc_888_edgelist_simple_v3.feather" "edgelist-v3"

download_one "$BASE_URL/banc_888_edgelist_split_v2.feather"   "$DATA_ROOT/banc_888_edgelist_split_v2.feather" "compartment-edgelist-v2"

download_one "$BASE_URL/banc_888_synapses_v3_enriched.parquet"   "$DATA_ROOT/banc_888_synapses_v3_enriched.parquet" "individual-synapses-v3"

download_one "$BASE_URL/banc_888_neurotransmitter_prediction_v2.csv"   "$DATA_ROOT/banc_888_neurotransmitter_prediction_v2.csv" "neurotransmitter-summary-v2"

download_optional "$BASE_URL/banc_888_cns_network_spectral_clustering_v2.csv"   "$DATA_ROOT/banc_888_cns_network_spectral_clustering_v2.csv" "cns-network-clusters"

download_optional "$BASE_URL/banc_888_betweenness_all_to_all_v2.csv"   "$DATA_ROOT/banc_888_betweenness_all_to_all_v2.csv" "betweenness-all-to-all"

download_optional "$BASE_URL/banc_888_betweenness_afferent_to_efferent_v2.csv"   "$DATA_ROOT/banc_888_betweenness_afferent_to_efferent_v2.csv" "betweenness-afferent-efferent"

download_optional "$ROOT_URL/neuron_skeletons.zip"   "$DATA_ROOT/neuron_skeletons.zip" "skeletons"

(
  cd "$DATA_ROOT"
  find . -maxdepth 1 -type f ! -name '*.part' ! -name 'manifest.sha256' -printf '%f\n'     | LC_ALL=C sort     | while read -r file; do
        sha256sum "$file"
      done > manifest.sha256
)

printf 'DATA_ROOT=%s\n' "$DATA_ROOT"
du -sh "$DATA_ROOT"
df -h "$DATA_ROOT"
