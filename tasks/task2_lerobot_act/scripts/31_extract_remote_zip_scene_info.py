#!/usr/bin/env python3
"""Extract CALVIN scene_info.npy files from a huge remote zip via HTTP ranges."""

from __future__ import annotations

import io
import json
import os
import struct
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests


EOCD_SIG = b"PK\x05\x06"
ZIP64_LOCATOR_SIG = b"PK\x06\x07"
ZIP64_EOCD_SIG = b"PK\x06\x06"
CD_SIG = b"PK\x01\x02"
LFH_SIG = b"PK\x03\x04"
ZIP64_EXTRA_ID = 0x0001


def http_range(url: str, start: int, end: int) -> bytes:
    headers = {"Range": f"bytes={start}-{end}"}
    expected = end - start + 1
    resp = requests.get(url, headers=headers, timeout=(15, 120), stream=expected > 8 * 1024 * 1024)
    resp.raise_for_status()
    if resp.status_code not in (200, 206):
        raise RuntimeError(f"unexpected HTTP status {resp.status_code} for range {start}-{end}")
    if expected <= 8 * 1024 * 1024:
        return resp.content
    chunks = []
    received = 0
    next_report = 25 * 1024 * 1024
    for chunk in resp.iter_content(chunk_size=1024 * 1024):
        if not chunk:
            continue
        chunks.append(chunk)
        received += len(chunk)
        if received >= next_report:
            print(
                f"range {start}-{end}: received {received / (1024 * 1024):.1f} MiB / {expected / (1024 * 1024):.1f} MiB",
                flush=True,
            )
            next_report += 25 * 1024 * 1024
    return b"".join(chunks)


def content_length(url: str) -> int:
    resp = requests.head(url, timeout=(15, 60), allow_redirects=True)
    resp.raise_for_status()
    return int(resp.headers["Content-Length"])


def http_range_parallel(url: str, start: int, size: int, workers: int = 8, part_size: int = 8 * 1024 * 1024) -> bytes:
    if size <= part_size or workers <= 1:
        return http_range(url, start, start + size - 1)
    parts = []
    end = start + size
    cur = start
    while cur < end:
        part_end = min(cur + part_size, end) - 1
        parts.append((cur, part_end))
        cur = part_end + 1
    print(
        f"parallel range download: {len(parts)} parts, workers={workers}, total={size / (1024 * 1024):.1f} MiB",
        flush=True,
    )
    output = bytearray(size)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(http_range, url, a, b): (a, b) for a, b in parts}
        for future in as_completed(futures):
            a, b = futures[future]
            data = future.result()
            rel = a - start
            output[rel : rel + len(data)] = data
            done += len(data)
            print(
                f"central directory received {done / (1024 * 1024):.1f} / {size / (1024 * 1024):.1f} MiB",
                flush=True,
            )
    return bytes(output)


def parse_zip64_extra(extra: bytes, needs: tuple[bool, bool, bool]) -> tuple[int | None, int | None, int | None]:
    pos = 0
    while pos + 4 <= len(extra):
        header_id, size = struct.unpack_from("<HH", extra, pos)
        pos += 4
        payload = extra[pos : pos + size]
        pos += size
        if header_id != ZIP64_EXTRA_ID:
            continue
        vals: list[int | None] = []
        off = 0
        for need in needs:
            if need:
                vals.append(struct.unpack_from("<Q", payload, off)[0])
                off += 8
            else:
                vals.append(None)
        return vals[0], vals[1], vals[2]
    return None, None, None


def find_central_directory(url: str) -> tuple[int, int]:
    total = content_length(url)
    tail_len = min(total, 262144)
    tail_start = total - tail_len
    tail = http_range(url, tail_start, total - 1)
    eocd_pos = tail.rfind(EOCD_SIG)
    if eocd_pos < 0:
        raise RuntimeError("EOCD signature not found")
    fields = struct.unpack_from("<4sHHHHIIH", tail, eocd_pos)
    cd_size_32 = fields[5]
    cd_offset_32 = fields[6]
    if cd_size_32 != 0xFFFFFFFF and cd_offset_32 != 0xFFFFFFFF:
        return cd_offset_32, cd_size_32

    locator_pos = tail.rfind(ZIP64_LOCATOR_SIG, 0, eocd_pos)
    if locator_pos < 0:
        raise RuntimeError("Zip64 locator not found")
    _, _disk, zip64_eocd_offset, _disks = struct.unpack_from("<4sIQI", tail, locator_pos)
    hdr = http_range(url, zip64_eocd_offset, zip64_eocd_offset + 56 - 1)
    if not hdr.startswith(ZIP64_EOCD_SIG):
        raise RuntimeError("Zip64 EOCD signature not found at locator offset")
    # signature, record size, versions, disk numbers, entry counts, central size, central offset
    _sig, _size, _ver_made, _ver_need, _disk_no, _cd_disk, _entries_disk, _entries, cd_size, cd_offset = struct.unpack_from(
        "<4sQHHIIQQQQ", hdr, 0
    )
    return int(cd_offset), int(cd_size)


def iter_central_directory(cd: bytes):
    pos = 0
    while pos + 46 <= len(cd):
        if cd[pos : pos + 4] != CD_SIG:
            next_pos = cd.find(CD_SIG, pos + 1)
            if next_pos < 0:
                break
            pos = next_pos
        vals = struct.unpack_from("<4sHHHHHHIIIHHHHHII", cd, pos)
        comp_method = vals[4]
        comp_size = vals[8]
        uncomp_size = vals[9]
        name_len = vals[10]
        extra_len = vals[11]
        comment_len = vals[12]
        local_offset = vals[16]
        name_start = pos + 46
        extra_start = name_start + name_len
        comment_start = extra_start + extra_len
        name = cd[name_start:extra_start].decode("utf-8")
        extra = cd[extra_start:comment_start]
        needs = (uncomp_size == 0xFFFFFFFF, comp_size == 0xFFFFFFFF, local_offset == 0xFFFFFFFF)
        z_uncomp, z_comp, z_offset = parse_zip64_extra(extra, needs)
        if z_uncomp is not None:
            uncomp_size = z_uncomp
        if z_comp is not None:
            comp_size = z_comp
        if z_offset is not None:
            local_offset = z_offset
        yield {
            "name": name,
            "compression": comp_method,
            "compressed_size": int(comp_size),
            "uncompressed_size": int(uncomp_size),
            "local_offset": int(local_offset),
        }
        pos = comment_start + comment_len


def read_zip_member(url: str, entry: dict) -> bytes:
    head = http_range(url, entry["local_offset"], entry["local_offset"] + 30 + 65535)
    if not head.startswith(LFH_SIG):
        raise RuntimeError(f"local header signature not found for {entry['name']}")
    vals = struct.unpack_from("<4sHHHHHIIIHH", head, 0)
    name_len = vals[9]
    extra_len = vals[10]
    data_start = entry["local_offset"] + 30 + name_len + extra_len
    comp = http_range(url, data_start, data_start + entry["compressed_size"] - 1)
    if entry["compression"] == 0:
        return comp
    if entry["compression"] == 8:
        return zlib.decompress(comp, -15)
    raise RuntimeError(f"unsupported compression method {entry['compression']} for {entry['name']}")


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: extract_remote_zip_scene_info.py <zip_url> <out_json>", file=sys.stderr)
        return 2
    url = sys.argv[1]
    out = Path(sys.argv[2])
    cd_offset, cd_size = find_central_directory(url)
    print(f"central_directory offset={cd_offset} size={cd_size}", flush=True)
    workers = int(os.environ.get("HW3_RANGE_WORKERS", "8"))
    cd = http_range_parallel(url, cd_offset, cd_size, workers=workers)
    matches = []
    for i, entry in enumerate(iter_central_directory(cd), start=1):
        if i % 200000 == 0:
            print(f"scanned central directory entries={i}", flush=True)
        if entry["name"].endswith("scene_info.npy"):
            matches.append(entry)
            print(f"found {entry['name']}", flush=True)
    print("matches", [m["name"] for m in matches], flush=True)
    result = {}
    for entry in matches:
        raw = read_zip_member(url, entry)
        scene_info = np.load(io.BytesIO(raw), allow_pickle=True).item()
        result[entry["name"]] = to_jsonable(scene_info)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"saved {out} bytes={out.stat().st_size}", flush=True)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
