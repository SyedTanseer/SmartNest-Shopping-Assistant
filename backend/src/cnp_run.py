#!/usr/bin/env python3
"""SmartNest — Assignment 3.1: CNP Demonstration Runner.

Run from the ``backend`` directory:

    python src/cnp_run.py

Starts two independent bidding agent servers, then runs the coordinator
to execute a multi-round Contract Net Protocol negotiation over TCP.
"""

from smartnest.cnp.cnp_demo import run

if __name__ == "__main__":
    run()
