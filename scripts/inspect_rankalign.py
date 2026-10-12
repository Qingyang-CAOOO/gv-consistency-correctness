from __future__ import annotations

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "external" / "rankalign"


def main() -> None:
    if not (UPSTREAM / ".git").exists():
        raise SystemExit("RankAlign is not cloned. Run: bash scripts/bootstrap_rankalign.sh")

    commit = subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True
    ).strip()
    print(f"RankAlign commit: {commit}")

    for rel in ["scripts", "data", "notebooks", "src"]:
        path = UPSTREAM / rel
        print(f"\n[{rel}]")
        if not path.exists():
            print("  <missing>")
            continue
        for p in sorted(path.rglob("*")):
            if p.is_file():
                print(" ", p.relative_to(UPSTREAM))

    script = UPSTREAM / "scripts" / "consistency_ft.py"
    if script.exists():
        print("\n--- consistency_ft.py --help (best effort) ---")
        proc = subprocess.run(
            ["python", str(script), "--help"],
            cwd=str(UPSTREAM),
            text=True,
            capture_output=True,
        )
        print(proc.stdout)
        if proc.returncode != 0:
            print(proc.stderr)
            print("The script may import dependencies before parsing --help; install upstream requirements first.")


if __name__ == "__main__":
    main()
