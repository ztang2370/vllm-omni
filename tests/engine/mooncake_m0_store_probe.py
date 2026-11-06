# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native Mooncake buffer probe for RFC #9908 M0; requires a running master.

Run with the pinned Mooncake environment::

    python tests/engine/mooncake_m0_store_probe.py --output /tmp/store-probe.json

This exercises actual Store calls, independently of the vLLM scheduler probe.
It removes only the unique keys it creates; it never resets the shared store.
"""

import argparse
import ctypes
import importlib.metadata
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def probe(master: str, metadata_server: str) -> dict:
    from mooncake.store import MooncakeDistributedStore

    store = MooncakeDistributedStore()
    result = store.setup("127.0.0.1", metadata_server, 64 << 20, 16 << 20, "tcp", "", master)
    if result != 0:
        raise RuntimeError(f"Mooncake setup failed: {result}")
    buffers = []
    keys = []
    namespace = f"omni-m0-native-{uuid.uuid4().hex}"

    def key(suffix: str) -> str:
        name = f"{namespace}/{suffix}"
        keys.append(name)
        return name

    def buffer(payload: bytes):
        allocation = ctypes.create_string_buffer(payload, len(payload))
        address = ctypes.addressof(allocation)
        status = store.register_buffer(address, len(payload))
        if status != 0:
            raise RuntimeError(f"Mooncake register_buffer failed: {status}")
        buffers.append(allocation)
        return allocation, address

    try:
        size = 65536
        data_a = bytes(range(256)) * (size // 256)
        data_b = bytes(reversed(range(256))) * (size // 256)
        _, address_a = buffer(data_a)
        _, address_b = buffer(data_b)
        destination, address_dst = buffer(b"\x00" * size)

        roundtrip = key("roundtrip")
        put_result = store.put_from(roundtrip, address_a, size)
        assert put_result == 0, put_result
        read_result = store.get_into(roundtrip, address_dst, size)
        assert read_result == size, read_result
        assert destination.raw == data_a

        missing_result = store.get_into(key("missing"), address_dst, size)
        assert missing_result < 0, missing_result
        missing_object = store.get(key("missing-object"))
        assert missing_object in (None, b""), missing_object

        short_key = key("short-object")
        short_size = size // 2
        assert store.put_from(short_key, address_a, short_size) == 0
        short_read = store.get_into(short_key, address_dst, size)
        assert short_read == short_size, short_read
        assert short_read != size  # The HS adapter must reject this incomplete object.
        undersized_result = store.get_into(roundtrip, address_dst, size // 2)
        assert undersized_result < 0, undersized_result

        removed_key = key("removed-after-lookup")
        assert store.put_from(removed_key, address_a, size) == 0
        assert store.is_exist(removed_key) == 1
        assert store.remove(removed_key, force=True) == 0
        removed_result = store.get_into(removed_key, address_dst, size)
        assert removed_result < 0, removed_result

        duplicate_result = store.put_from(roundtrip, address_b, size)
        assert store.get_into(roundtrip, address_dst, size) == size
        duplicate_kept_original = destination.raw == data_a
        assert destination.raw in (data_a, data_b), "Duplicate write produced mixed bytes"

        concurrent_key = key("concurrent")
        barrier = threading.Barrier(2)

        def competing_put(address: int) -> int:
            barrier.wait(timeout=10)
            return store.put_from(concurrent_key, address, size)

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(competing_put, address) for address in (address_a, address_b)]
            concurrent_results = [future.result(timeout=30) for future in futures]
        assert 0 in concurrent_results, concurrent_results
        assert store.get_into(concurrent_key, address_dst, size) == size
        assert destination.raw in (data_a, data_b), "Concurrent write produced mixed bytes"

        packages = {
            distribution.metadata["Name"]: distribution.version
            for distribution in importlib.metadata.distributions()
            if "mooncake" in distribution.metadata["Name"].lower()
        }
        return {
            "packages": packages,
            "protocol": "tcp",
            "master": master,
            "buffer_bytes": size,
            "roundtrip_exact": True,
            "put_result": put_result,
            "read_bytes": read_result,
            "missing_result": missing_result,
            "missing_object_result": repr(missing_object),
            "short_object_read_bytes": short_read,
            "undersized_buffer_result": undersized_result,
            "removed_after_lookup_result": removed_result,
            "duplicate_put_result": duplicate_result,
            "duplicate_kept_original": duplicate_kept_original,
            "concurrent_put_results": concurrent_results,
            "concurrent_value_complete": True,
        }
    finally:
        for name in keys:
            store.remove(name, force=True)
        for allocation in buffers:
            store.unregister_buffer(ctypes.addressof(allocation))
        store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", default="127.0.0.1:50051")
    parser.add_argument("--metadata-server", default="http://127.0.0.1:8080/metadata")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = probe(args.master, args.metadata_server)
    encoded = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
