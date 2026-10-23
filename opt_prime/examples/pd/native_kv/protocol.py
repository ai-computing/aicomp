"""Experimental 2026.4 wire subset, recovered from owned native-PD experiments.

Not a vendor-supported ABI. Supports only the observed Qwen3 global BF16 layout.
No SDK, torch, or network dependency. Unknown variants fail closed.
"""
import json
import struct

MAX_POSITIONS = 2047
MAX_HEADER = 4 * 1024 * 1024


def u32(data, offset):
    if not 0 <= offset <= len(data) - 4:
        raise ValueError('truncated u32')
    return struct.unpack_from('<I', data, offset)[0]


def rel(data, offset):
    u32(data, offset)
    return offset + struct.unpack_from('<i', data, offset)[0]


def pad(data):
    data.extend(b'\0' * (-len(data) % 4))


def string_ref(data, offset, start, length):
    # Observed out-of-line archived UTF-8 string (short inline strings excluded).
    if not 8 <= length <= 65535:
        raise ValueError('out-of-line string length must be 8..65535 bytes')
    encoded = ((length & ~127) << 2) | (length & 127) | 128
    struct.pack_into('<Ii', data, offset, encoded, start - offset)


def read_string(data, offset):
    word = u32(data, offset)
    if word & 0x180 != 0x80:
        raise ValueError('unsupported string encoding')
    length = (word & 127) | ((word >> 2) & ~127)
    if not 8 <= length <= 65535:
        raise ValueError('unsupported string length')
    start = offset + struct.unpack_from('<i', data, offset + 4)[0]
    if start < 0 or start + length > offset:
        raise ValueError('invalid string relative reference')
    return data[start:start + length].decode('utf8')


def validate_layout(layout):
    axes = layout.get('dram_shape', {}).get('intra_chip_axes', {}).get('axes', [])
    if (layout.get('attention_type') != 'Global' or
        layout.get('element_type') != 'Bfloat16' or
        layout.get('block_size') != 2048 or layout.get('unit_block_size') != 2048 or
        layout.get('num_chips') != 1 or len(axes) != 3 or
        [a.get('size') for a in axes[1:]] != [8, 128] or
        [a.get('stride') for a in axes] != [1024, 128, 1] or
        layout['dram_shape'].get('inter_chip_axes') != {'inner': []} or
        layout['dram_shape']['intra_chip_axes'].get('offset') != 0 or
        axes[0].get('size') != 272321 or
        [a.get('tag') for a in axes] != [
            {'LabelStride': {'label': {'inner': label}, 'stride': 1}} for label in ('0','2','3')]):
        raise ValueError('unsupported layout; only measured Qwen3 FXB is admitted')


def handshake(endpoint, layout):
    validate_layout(layout)
    ep = endpoint.encode('ascii')
    js = json.dumps(layout, separators=(',', ':')).encode()
    data = bytearray(ep + js)
    pad(data)
    root = len(data)
    data.extend(b'\0' * 36)
    string_ref(data, root + 8, 0, len(ep))
    string_ref(data, root + 16, len(ep), len(js))
    data[root + 24:root + 32] = b'null\xff\xff\xff\xff'
    return bytes(data)


def control(request_id, kind, block_ids=()):
    encoded = request_id.encode('ascii')
    if not 8 <= len(encoded) <= 127:
        raise ValueError('request ID must be 8..127 ASCII bytes')
    ids = list(block_ids)
    if kind not in (1, 2, 3) or (kind != 2 and ids):
        raise ValueError('unsupported control message')
    if len(ids) > MAX_POSITIONS or len(set(ids)) != len(ids) or any(not 0 <= n < 272321 for n in ids):
        raise ValueError('invalid destination block IDs')
    data = bytearray(encoded)
    pad(data)
    start = len(data)
    for n in ids:
        data.extend(struct.pack('<I', n))
    root = len(data)
    data.extend(b'\0' * 36)
    struct.pack_into('<I', data, root + 4, kind)
    string_ref(data, root + 8, 0, len(encoded))
    if kind == 2:
        struct.pack_into('<iIiI', data, root + 20, start - (root + 20), len(ids), -28, 0)
    return bytes(data)


def parse_message(parts):
    if not parts or len(parts) > 2:
        raise ValueError('expected header and optional payload')
    data = parts[0]
    if not 36 <= len(data) <= MAX_HEADER or len(data) % 4:
        raise ValueError('invalid header length')
    root = len(data) - 36
    variant = u32(data, root)
    if variant == 0:
        kind = u32(data, root + 4)
        if kind != 1 or len(parts) != 1:
            raise ValueError('only PrefillReady control accepted by receiver')
        return {'kind': 'ready', 'id': read_string(data, root + 8)}
    if variant != 1 or len(parts) != 2:
        raise ValueError('unsupported message variant')
    request_id = read_string(data, root + 4)
    start = rel(data, root + 12)
    count = u32(data, root + 16)
    final = u32(data, root + 20)
    if (not 0 < count <= 4 * MAX_POSITIONS or start < 0 or
        start + count * 16 != root or final not in (0, 1) or
        data[root + 24:] != b'\0' * 12):
        raise ValueError('invalid block vector or unsupported auxiliary fields')
    blocks = [struct.unpack_from('<4I', data, start + i * 16) for i in range(count)]
    if any(domain != 0 or slot >= 72 or block >= 272321 or size != 2048
           for domain, slot, block, size in blocks):
        raise ValueError('unsupported block metadata')
    if len(parts[1]) != sum(b[3] for b in blocks):
        raise ValueError('payload length differs from metadata')
    return {'kind': 'batch', 'id': request_id, 'blocks': blocks, 'final': bool(final)}
