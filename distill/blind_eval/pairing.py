#!/usr/bin/env python3
"""配对清单的**盲法逻辑与自检**。只依赖标准库 + PIL,不依赖 FastAPI。

WHY 单独成文件:这是整个工具里唯一"错了会静默污染数据"的部分——槽位分配错、
清单校验漏,产出的标注看上去一切正常,只有事后重算才可能发现(M4 已经这样栽过一次)。
把它从 Web 层剥出来,才能在**没有 FastAPI 的机器上**(比如本地)直接跑行为测试;
否则测试只能在能起服务的机器上跑,而那正是最不方便反复跑的地方。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

SLOTS = ("L", "R")
CHOICES = ("L", "R", "tie")

REQUIRED_FIELDS = ("pair_id", "kind", "stratum", "prompt", "ref_paths",
                   "key_0", "key_1", "img_0", "img_1")


def key0_on_left(pair_id: str, blind_seed: str) -> bool:
    """清单里的 0 号候选是否显示在左边。

    纯函数、跨进程稳定(不能用内置 `hash`——`PYTHONHASHSEED` 会随机化,
    重启一次全部槽位就变,已标注的数据当场作废)。

    以 `pair_id` 而非 task_id 为输入:重放对与原对的 pair_id 不同,
    因此**槽位天然重掷**,不需要额外机制。
    """
    return int(hashlib.md5(f"{pair_id}|{blind_seed}".encode()).hexdigest(), 16) % 2 == 0


def asset_tag(manifest_text: str) -> str:
    """图片 URL 的版本戳 = 清单原文摘要。

    WHY 必须有:图片 URL 形如 `/api/img?k=ref:{idx}:{i}`,**只含下标不含内容**。
    换一批清单后同一个下标指向完全不同的图,而 URL 一字未变——浏览器直接命中上一批的
    缓存。M5 首次上机就撞上了:M4 与 M5 的 `ref:0:0` URL 完全相同,于是 M5 的第 0 对
    (闹钟)显示出 M4 第 0 对的参考图(书包)。候选图因为槽位由 A/B 改名 L/R 而 URL 变了,
    反而是对的——**只有参考图错**,画面上没有任何异常信号,人却在拿错的参考图做忠实度判断。

    用清单原文的摘要而非 batch_id:batch_id 可以在内容改变时保持不变(重新生成同名批次),
    那正是缓存会再次骗人的场景。内容变则戳变,是这里唯一靠得住的条件。
    """
    return hashlib.md5(manifest_text.encode("utf-8")).hexdigest()[:12]


def slot_of(pair: dict, blind_seed: str, slot: str) -> tuple[str, str]:
    """slot "L"=左图 / "R"=右图 → (语义标签, 图路径)。**只存在于服务端。**"""
    if slot not in SLOTS:
        raise ValueError(f"slot must be one of {SLOTS}, got {slot!r}")
    first = 0 if key0_on_left(pair["pair_id"], blind_seed) else 1
    idx = first if slot == "L" else 1 - first
    return pair[f"key_{idx}"], pair[f"img_{idx}"]


def decodable(path: Path) -> bool:
    """图存在**且能完整解码**。`Image.open` 惰性只读文件头,截断图只有 `load()` 才炸。"""
    if not path.is_file():
        return False
    try:
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:
        return False


def check_manifest(manifest: dict, root: Path) -> list[str]:
    """返回问题清单(空 = 通过)。**纯谓词,不动任何文件。**"""
    meta = manifest.get("meta", {})
    pairs = manifest.get("pairs", [])
    errs: list[str] = []

    if not meta.get("blind_seed"):
        errs.append("meta.blind_seed 缺失——没有它槽位无法确定性分配")
    if not pairs:
        errs.append("清单里一对都没有")

    ids = [p.get("pair_id") for p in pairs]
    if len(set(ids)) != len(ids):
        errs.append("pair_id 有重复——标注会互相覆盖")

    missing: list[str] = []
    for p in pairs:
        lack = [k for k in REQUIRED_FIELDS if k not in p]
        if lack:
            errs.append(f"{p.get('pair_id', '?')} 缺字段 {lack}")
            continue
        if p["key_0"] == p["key_1"]:
            errs.append(f"{p['pair_id']} 的两个候选标签相同,揭盲后分不出胜者")
        if not p["ref_paths"]:
            errs.append(f"{p['pair_id']} 没有参考图")
        for rel in (p["img_0"], p["img_1"], *p["ref_paths"]):
            target = (root / rel).resolve()
            try:      # 路径逃逸检查:清单是输入文件,不能当成可信来源
                target.relative_to(root.resolve())
            except ValueError:
                errs.append(f"{p['pair_id']} 的路径逃出仓库:{rel}")
                continue
            if not decodable(target):
                missing.append(f"{p['pair_id']}: {rel}")

    if missing:
        errs.append(f"{len(missing)} 个图片缺失或不可解码,例如:\n    " +
                    "\n    ".join(missing[:5]))
    return errs


def validate(manifest: dict, root: Path) -> tuple[list[dict], str]:
    """自检不过就 `SystemExit`。

    WHY 宁可启动即炸:评到一半才发现缺图,前面的人工判断全部白费——而人工判读时间
    是整个 M5 里最贵、最不可再生的资源。启动多花几秒换掉这个风险。
    """
    errs = check_manifest(manifest, root)
    if errs:
        raise SystemExit("❌ 配对清单自检未通过:\n  - " + "\n  - ".join(errs))
    return manifest["pairs"], manifest["meta"]["blind_seed"]
