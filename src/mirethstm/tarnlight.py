"""Feeds Tarnlight, an optional live decision console, through its drop box (SPEC 10.3).

Tarnlight makes ~/.tarnlight/inbox/ when it is installed and imports every complete line written there. We
only write when that folder already exists (never creating it), write each decision exactly once (Tarnlight
keeps duplicates), and never raise: the decision is already made when this runs.
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path

SOURCE = "mirethstm"
MIN_FREE_BYTES = 1 << 30  # leave the last gigabyte of the disk alone
MAX_LINE_BYTES = 16 << 20  # Tarnlight refuses longer lines


def drop(request, response, request_id, latency_ms):
    """Append one record for a finished decision; True when a line was written.

    `request` is {"model", "state", "questions"}; `response` is the /v1/systemone body.
    """
    try:
        inbox = Path.home() / ".tarnlight" / "inbox"
        if not inbox.is_dir() or shutil.disk_usage(inbox).free < MIN_FREE_BYTES:
            return False
        now = time.time()
        record = {"v": 1, "ts": now, "source": SOURCE, "request_id": request_id, "request": request,
                  "response": response, "latency_ms": latency_ms, "status": 200, "cost_est_micro": 0}
        line = json.dumps(record, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(line) > MAX_LINE_BYTES:
            return False
        # The file is named after the current UTC hour; Tarnlight deletes an hour's file once it has read it all.
        append_line(inbox / f"{SOURCE}-{time.strftime('%Y-%m-%dT%H', time.gmtime(now))}.jsonl", line + b"\n")
        return True
    except Exception:
        return False


if sys.platform == "win32":
    import _winapi

    _FILE_APPEND_DATA = 0x0004
    _SYNCHRONIZE = 0x00100000
    _SHARE_READ_WRITE_DELETE = 0x0007
    _OPEN_ALWAYS = 4
    _FILE_ATTRIBUTE_NORMAL = 0x0080

    def append_line(path, data):
        # Append-only access makes Windows put the whole write at the end of the file, even while another
        # process appends too. Python's "ab" mode seeks to the end first, so two writers could overlap.
        handle = _winapi.CreateFile(str(path), _FILE_APPEND_DATA | _SYNCHRONIZE, _SHARE_READ_WRITE_DELETE, 0,
                                    _OPEN_ALWAYS, _FILE_ATTRIBUTE_NORMAL, 0)
        try:
            _winapi.WriteFile(handle, data)
        finally:
            _winapi.CloseHandle(handle)
else:
    def append_line(path, data):
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)  # each line holds the caller's state
        try:
            os.write(fd, data)
        finally:
            os.close(fd)
