#!/usr/bin/env python3
"""Pull the two Docker images, download Gurobi and compile PlaJA.

Examples:
  scripts/setup.py
  scripts/setup.py --roundingsat <dir>   # with PB support; needs <dir>/src/ipasirpb.h and <dir>/build/librsipasir.so
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from common import PLAJA_IMAGE, RRL_IMAGE, ROOT


def run(command):
    subprocess.run(command, check=True)


def setup(roundingsat):
    user = f"{os.getuid()}:{os.getgid()}"

    print("Pulling the Docker images ...")
    run(["docker", "pull", PLAJA_IMAGE])
    run(["docker", "pull", RRL_IMAGE])

    if not os.path.exists(".cache/gurobi/linux/lib/libgurobi_c++.a"):
        print("Downloading Gurobi 12.0.0 ...")
        shutil.rmtree(".cache/gurobi", ignore_errors=True)
        os.makedirs(".cache", exist_ok=True)
        download = (
            "wget -q https://packages.gurobi.com/12.0/gurobi12.0.0_linux64.tar.gz && "
            "tar -xzf gurobi12.0.0_linux64.tar.gz && "
            "rm gurobi12.0.0_linux64.tar.gz && "
            "mkdir gurobi && "
            "mv gurobi1200/linux64 gurobi/linux && "
            "rmdir gurobi1200 && "
            "make -s -C gurobi/linux/src/build && "
            "cp gurobi/linux/src/build/libgurobi_c++.a gurobi/linux/lib/"
        )
        run(["docker", "run", "--rm", "-u", user, "-v", f"{ROOT}/.cache:/cache", "-w", "/cache",
             PLAJA_IMAGE, "bash", "-c", download])

    print("Compiling PlaJA ...")
    os.makedirs("fnn-plaja/build", exist_ok=True)
    roundingsat_file = Path("fnn-plaja/build/roundingsat.path")
    if roundingsat_file.exists():
        roundingsat_file.unlink()

    if roundingsat:
        roundingsat = os.path.abspath(roundingsat)
        build = ("cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_PB_CONSTRAINTS=ON "
                 "-DROUNDINGSAT_DIR=/workspace/dependencies/roundingsat "
                 "-DROUNDINGSAT_BUILD_DIR=/workspace/dependencies/roundingsat/build .. && "
                 "make -j$(nproc) PlaJA")
        run(["docker", "run", "--rm", "-u", user, "-e", "HOME=/tmp",
             "-v", f"{ROOT}/fnn-plaja:/ws", "-w", "/ws/build",
             "-v", f"{ROOT}/.cache/gurobi:/workspace/dependencies/gurobi:ro",
             "-v", f"{roundingsat}:/workspace/dependencies/roundingsat:ro",
             PLAJA_IMAGE, "bash", "-c", build])
        roundingsat_file.write_text(roundingsat + "\n")
    else:
        build = "cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_PB_CONSTRAINTS=OFF .. && make -j$(nproc) PlaJA"
        run(["docker", "run", "--rm", "-u", user, "-e", "HOME=/tmp",
             "-v", f"{ROOT}/fnn-plaja:/ws", "-w", "/ws/build",
             "-v", f"{ROOT}/.cache/gurobi:/workspace/dependencies/gurobi:ro",
             PLAJA_IMAGE, "bash", "-c", build])

    print("Done: fnn-plaja/build/PlaJA")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roundingsat", metavar="DIR", help="RoundingSat build with the IPASIR-PB interface")
    args = parser.parse_args()
    setup(args.roundingsat)
