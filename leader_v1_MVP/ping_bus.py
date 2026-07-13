#!/usr/bin/env python3
"""Broadcast-ping the Dynamixel bus through the base adapter and list responders.

Shows every ID on the bus (motors 1-7 and the button node 8) with its model number
— the quickest way to isolate which chain segment kills the bus: add one segment
at a time and re-run.

    uv run python leader_v1_MVP/ping_bus.py --port /dev/tty.usbmodemXXXX
"""
import argparse

from dynamixel_sdk import PortHandler, PacketHandler

ap = argparse.ArgumentParser()
ap.add_argument("--port", required=True)
ap.add_argument("--baud", type=int, default=1_000_000)
args = ap.parse_args()

port = PortHandler(args.port)
pk = PacketHandler(2.0)
if not port.openPort() or not port.setBaudRate(args.baud):
    raise SystemExit(f"Could not open {args.port} @ {args.baud}")

data, comm = pk.broadcastPing(port)   # {id: [model, fw_ver]}
print(f"Responders @ {args.baud}: {sorted(data.keys()) if data else 'NONE'}")
for i in sorted(data):
    print(f"  id {i}: model={data[i][0]} fw={data[i][1]}")

# Explicit ping of the button node in case broadcast missed it (the slave library
# doesn't stagger its broadcast reply, so it can collide and be absent above).
model, comm, err = pk.ping(port, 8)
print(f"\nping(8): comm={comm} (0=OK) err={err} model={model}")
port.closePort()
