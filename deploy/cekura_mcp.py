#!/usr/bin/env python3
"""Minimal Cekura MCP (Streamable HTTP / JSON-RPC) client over urllib (stdlib).

Usage:
  python3 cekura_mcp.py schema <tool>            # print a tool's inputSchema
  python3 cekura_mcp.py call <tool> '<json>'     # call a tool, print result
  python3 cekura_mcp.py tools [filter]           # list tool names
"""
import json
import sys
import urllib.request
from pathlib import Path

URL = "https://api.cekura.ai/mcp"
KEY = (Path(__file__).resolve().parent / "cekura_api_key.txt").read_text().strip()
_HDR = {"Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "X-CEKURA-API-KEY": KEY}
_sid = {"v": None}


def _post(payload):
    headers = dict(_HDR)
    if _sid["v"]:
        headers["mcp-session-id"] = _sid["v"]
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(), headers=headers, method="POST")
    resp = urllib.request.urlopen(req, timeout=120)
    if not _sid["v"]:
        _sid["v"] = resp.headers.get("mcp-session-id")
    body = resp.read().decode()
    # SSE: extract the JSON after the last "data: "
    out = None
    for line in body.splitlines():
        if line.startswith("data: "):
            out = json.loads(line[6:])
    return out


def _init():
    _post({"jsonrpc": "2.0", "id": 0, "method": "initialize",
           "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                      "clientInfo": {"name": "cekura_mcp.py", "version": "1.0"}}})
    _post({"jsonrpc": "2.0", "method": "notifications/initialized"})


def _tools():
    r = _post({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    return r["result"]["tools"]


def call(name, args):
    r = _post({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
               "params": {"name": name, "arguments": args}})
    return r


def main():
    _init()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "tools"
    if cmd == "tools":
        flt = sys.argv[2] if len(sys.argv) > 2 else ""
        for t in _tools():
            if flt in t["name"]:
                print(t["name"])
    elif cmd == "schema":
        name = sys.argv[2]
        for t in _tools():
            if t["name"] == name:
                print(json.dumps(t.get("inputSchema", {}), indent=2))
                return
        print("not found:", name)
    elif cmd == "call":
        name = sys.argv[2]
        args = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
        print(json.dumps(call(name, args), indent=2))


if __name__ == "__main__":
    main()
