import struct
import json
import pathlib
import tempfile
import unittest

try:
    from . import coordinator
except ImportError:
    import coordinator


class ProtocolTest(unittest.TestCase):
    def test_frame_round_trip_header(self):
        frame = coordinator.encode_frame(coordinator.ACTIVATION, 7, 11, 2, 896,
                                         coordinator.F32LE, b"1234")
        self.assertEqual(coordinator.decode_header(frame[:coordinator.HEADER.size]),
                         (coordinator.ACTIVATION, 7, 11, 2, 896, coordinator.F32LE, 4))

    def test_rejects_oversized_payload_before_allocation(self):
        header = struct.pack("!IHHQIIIHHQ", coordinator.MAGIC, coordinator.VERSION,
                             coordinator.ACTIVATION, 1, 0, 1, 1, coordinator.F32LE,
                             0, coordinator.MAX_PAYLOAD + 1)
        with self.assertRaisesRegex(ValueError, "64 MiB"):
            coordinator.decode_header(header)

    def test_rejects_shape_payload_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, "activation size"):
            coordinator.require_activation(
                (coordinator.ACTIVATION, 1, 0, 2, 4, coordinator.F32LE), b"\0" * 8, 1, 0, 4)

    def test_manifest_requires_contiguous_internal_split(self):
        manifest = {"model_id": "test", "architecture": "qwen2", "layers": 24,
                    "hidden_size": 896, "split_layer": 24, "quantization": "Q4_K_M"}
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "model.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest"):
                coordinator.load_manifest(path)


if __name__ == "__main__":
    unittest.main()
