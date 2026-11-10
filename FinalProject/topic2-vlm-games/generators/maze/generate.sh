#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "Usage: bash generate.sh OUTPUT SMALL_BOARDS MEDIUM_BOARDS LARGE_BOARDS [EXCLUSION_FILE]" >&2
  exit 2
fi
EXCLUDE="${5:-$ROOT/../../test_data/excluded-states.txt}"
[[ -f "$EXCLUDE" ]] || { echo "Missing exclusion list: $EXCLUDE" >&2; exit 2; }
mkdir -p "$ROOT/build/classes"
find "$ROOT/src/main/java" -name '*.java' -print0 | xargs -0 javac -encoding UTF-8 -cp "$ROOT/lib/gson.jar" -d "$ROOT/build/classes"
exec java -Djava.awt.headless=true -cp "$ROOT/build/classes:$ROOT/lib/gson.jar" com.zjt.Main "$1" "$2" "$3" "$4" "$EXCLUDE"
