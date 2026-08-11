#!/usr/bin/env python3
"""字节级校验 LoRA checkpoint 的 dit_lora.safetensors —— 不依赖 torch。

判据同 M6_STEP1_RUN.md 步骤 3 / M6_STEP1_REPORT.md §2.2:
    `304 张量,空分片 0,全零 0`

- shape 含 0 ⇒ 空分片(ZeRO-3 下 state_dict() 拿空分片的典型症状)
- 数据段全 0x00 ⇒ 全零张量

用法:
    python scripts/check_lora_ckpt.py <path/to/dit_lora.safetensors>
"""
import json
import struct
import sys


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    path = sys.argv[1]
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
        names = [k for k in header if k != "__metadata__"]
        total = len(names)
        empty = 0
        zero = 0
        checked_bytes = 0
        for name in names:
            t = header[name]
            if 0 in t["shape"]:
                empty += 1
            start, end = t["data_offsets"]
            f.seek(8 + n + start)
            data = f.read(end - start)
            checked_bytes += len(data)
            # numpy.any() 是 C 级全零扫描;环境没有 numpy 时退回 bytes 检查
            try:
                import numpy as np
                if not np.frombuffer(data, dtype=np.uint8).any():
                    zero += 1
            except ImportError:
                if not any(data):
                    zero += 1
        print(f"{path}: {total} 张量 / 空分片 {empty} / 全零 {zero}"
              f" / 数据段共 {checked_bytes/2**30:.2f} GiB")
        ok = total == 304 and empty == 0 and zero == 0
        print("✅ 通过" if ok else "❌ 未通过")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
