#!/usr/bin/env bash
# Stage 1: Extract leak archives.
# Skips repo.tar (497GB source) and mesalab_git.tar.zst (63GB git) by default.
# Re-enable below if you later want to index source code.

set -euo pipefail

: "${GEEDGE_RAW:?Set GEEDGE_RAW to the leak source dir, e.g. ~/projects/geedge-docs}"
: "${GEEDGE_WORK:?Set GEEDGE_WORK to your working dir, e.g. ~/geedge-data}"

OUT="${GEEDGE_WORK}/extracted"
mkdir -p "$OUT"

# Archives we care about for prose/policy content
ARCHIVES=(
  "geedge_docs.tar.zst"
  "geedge_jira.tar.zst"
  "mesalab_docs.tar.zst"
)

for archive in "${ARCHIVES[@]}"; do
  src="${GEEDGE_RAW}/${archive}"
  name="$(basename "${archive%.tar*}")"
  dest="${OUT}/${name}"

  if [[ ! -f "$src" ]]; then
    echo "MISSING: $src" >&2
    continue
  fi

  if [[ -d "$dest" && -n "$(ls -A "$dest" 2>/dev/null)" ]]; then
    echo "SKIP (already extracted): $dest"
    continue
  fi

  echo "EXTRACTING: $src -> $dest"
  mkdir -p "$dest"
  if [[ "$archive" == *.tar.zst ]]; then
    zstd -dc "$src" | tar -xf - -C "$dest"
  else
    tar -xf "$src" -C "$dest"
  fi
done

# Loose .docx files at the leak root
echo "Linking loose root files..."
mkdir -p "${OUT}/_root_files"
find "${GEEDGE_RAW}" -maxdepth 1 -type f \( -iname '*.docx' -o -iname '*.pdf' -o -iname '*.pptx' -o -iname '*.xlsx' \) \
  -exec ln -sf {} "${OUT}/_root_files/" \;

# JIRA attachments include zip/rar/gz/xz files (JIRA never unpacks what
# users attach). Stage 2/3 treat those extensions as opaque binary and
# skip them, so anything inside is invisible to the pipeline unless we
# recurse here — otherwise email/JIRA attachments don't unpack
# recursively and their contents never reach Stage 3.
JIRA_ATTACH="${OUT}/geedge_jira/attachment"
if [[ -d "$JIRA_ATTACH" ]]; then
  echo "Unpacking nested archives under $JIRA_ATTACH..."
  UNPACK_LOG="${OUT}/jira_nested_unpack_errors.log"
  : > "$UNPACK_LOG"
  find "$JIRA_ATTACH" -type f \( -iname '*.zip' -o -iname '*.rar' -o -iname '*.7z' -o -iname '*.gz' -o -iname '*.xz' \) -print0 |
  while IFS= read -r -d '' nested; do
    dest="${nested}_unpacked"
    [[ -d "$dest" ]] && continue
    mkdir -p "$dest"
    if ! unar -f -q -o "$dest" "$nested" </dev/null >/dev/null 2>>"$UNPACK_LOG"; then
      echo "FAILED: $nested" >> "$UNPACK_LOG"
      rmdir --ignore-fail-on-non-empty "$dest" 2>/dev/null || true
    fi
  done
  [[ -s "$UNPACK_LOG" ]] && echo "Nested-archive unpack errors logged to $UNPACK_LOG"
fi

# Verify every attachment referenced by an issue's JSON actually landed
# on disk. Catches plain extraction gaps distinct from the nested-archive
# case above.
if [[ -d "${OUT}/geedge_jira/issues" ]]; then
  echo "Checking JIRA attachment completeness..."
  python3 "$(dirname "$0")/check_jira_attachments.py" "${OUT}/geedge_jira"
fi

echo "Done. Extracted to: $OUT"
du -sh "${OUT}"/* 2>/dev/null || true
