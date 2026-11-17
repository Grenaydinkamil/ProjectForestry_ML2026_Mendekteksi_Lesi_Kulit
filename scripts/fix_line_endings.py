"""
Memperbaiki akhir baris CRLF -> LF pada file skrip shell.

Kenapa perlu?
  File .sh yang ditulis di Windows biasanya berakhir dengan CRLF (\r\n).
  Bash membaca \r sebagai karakter biasa, jadi baris jadi:
      set -uo pipefail\r     ->  error: invalid option name
      #!/bin/bash\r          ->  command not found
  Skrip yang tadinya jalan di mesin ini jadi gagal total.

Skrip ini mengubah semua .sh di repo menjadi LF, dan memastikan shebang
memakai LF.

Jalankan:  python scripts/fix_line_endings.py
"""

from __future__ import annotations

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def main() -> int:
    fixed = []
    for pattern in ("scripts/*.sh", "scripts/**/*.sh", "*.sh"):
        for path in REPO_ROOT.glob(pattern):
            if path.is_dir():
                continue
            raw = path.read_bytes()
            if b"\r\n" not in raw and not raw.startswith(b"\r"):
                continue
            new = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            # Pastikan shebang di baris pertama
            if new.startswith(b"#!/") and not new.startswith(b"#!/bin/bash\n"):
                new = b"#!/bin/bash\n" + new.split(b"\n", 1)[1]
            path.write_bytes(new)
            fixed.append(str(path.relative_to(REPO_ROOT)))

    if fixed:
        print("Diperbaiki (CRLF -> LF):")
        for f in fixed:
            print(f"  {f}")
    else:
        print("Semua file .sh sudah memakai LF.")

    # Verifikasi
    bad = []
    for pattern in ("scripts/*.sh", "scripts/**/*.sh"):
        for path in REPO_ROOT.glob(pattern):
            if path.is_dir():
                continue
            if b"\r" in path.read_bytes():
                bad.append(str(path.relative_to(REPO_ROOT)))

    if bad:
        print("\nMASIH ADA CR:")
        for b in bad:
            print(f"  {b}")
        return 1

    print("Verifikasi: semua .sh bebas CR.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
