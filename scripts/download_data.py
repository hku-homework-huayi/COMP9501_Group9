#!/usr/bin/env python3
"""Fetch public data with pinned SHA-256 checksums; never execute downloaded code."""

import hashlib
import shutil
import subprocess
import urllib.request
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "data/cmupie_1632.rar": (
        "https://raw.githubusercontent.com/Gunther17/CMU_PIE-dataset/master/cmupie_1632.rar",
        "5526a55909b8788d81698ff9922be8eb64ab69f6b69ec5d1eb4ae359e4f7a0bb",
    ),
    "data/ORL_64x64.mat": (
        "http://www.cad.zju.edu.cn/home/dengcai/Data/ORL/ORL_64x64.mat",
        "8235e632b2083a2422cb1c2c5e0eda4272af2aa0b40abce20868ad09e5ba92a5",
    ),
    "papers/xu_ebsl.pdf": (
        "https://www.ee.columbia.edu/ln/dvmm/publications/09/xu_ebsl.pdf",
        "9548936df905405ce49ce6fbfa305480069235f71a6143f41480623c98b3bed6",
    ),
}


def main():
    for name, (url, expected) in SOURCES.items():
        path = ROOT / name
        path.parent.mkdir(exist_ok=True, parents=True)
        if not path.exists():
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"Checksum mismatch for {url}; source may have changed")
            path.write_bytes(data)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Checksum mismatch for existing {path}")
        print("Verified", name, actual)
    target = ROOT / "data/pie_public"
    target.mkdir(exist_ok=True)
    if len(list((target / "cmupie_1632").glob("*/*.jpg"))) == 1632:
        print("PIE images already extracted")
        return
    tar = shutil.which("bsdtar")
    if not tar:
        raise RuntimeError("bsdtar (libarchive, RAR support) is required to extract PIE")
    archive = ROOT / "data/cmupie_1632.rar"
    names = subprocess.check_output([tar, "-tf", str(archive)], text=True).splitlines()
    for name in names:
        p = PurePosixPath(name)
        if p.is_absolute() or ".." in p.parts or p.parts[0] != "cmupie_1632":
            raise ValueError("Unsafe archive entry: " + name)
    subprocess.run([tar, "-xf", str(archive), "-C", str(target), "--no-same-owner"], check=True)
    if len(list((target / "cmupie_1632").glob("*/*.jpg"))) != 1632:
        raise ValueError("Incomplete extraction")


if __name__ == "__main__":
    main()
