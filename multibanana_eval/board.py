"""对比拼图的共享逻辑:一行 [参考图 | 红线 | 各变体],标题按宽度自动换行。

从 infer_multibanana.py 抽出来,让 infer(生成时拼)和 rebuild_comparison(事后重拼)
共用同一份正确逻辑,不再各拷一份、各自截断。

之前截断的原因:标题把 prompt 截到 70 字符、且单行渲染,超出图宽的部分被裁掉。
MultiBanana 的 prompt 可以很长(还有 multilingual text 任务),必须多行换行。
"""
import os

from PIL import Image, ImageDraw, ImageFont

_LINE_H = 17          # 标题行高(字号 14)
_TITLE_PAD = 6        # 标题区上下留白
_TILE_BAR = 24        # 每个小图上方标签条高度

# 优先选覆盖 CJK 的系统字体(MultiBanana 有 multilingual 文字),让中文/日文不显示成方框;
# 都找不到就回退默认位图字体——英文照常,非 ASCII 才可能豆腐块,但布局(换行)始终正确。
_FONT_CANDIDATES = [
    # Linux(远程优先):noto-cjk / 文泉驿
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    # macOS 中文(不同系统版本位置不一)
    "/System/Library/Fonts/Supplemental/PingFang.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    # 英文兜底(比默认位图清晰;无 CJK 字形)
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
_font_cache: dict[int, "ImageFont.ImageFont"] = {}


def _font(size: int):
    if size in _font_cache:
        return _font_cache[size]
    font = None
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size)
                break
            except Exception:
                continue
    if font is None:
        try:
            font = ImageFont.load_default(size=size)  # Pillow >= 10.1
        except TypeError:
            font = ImageFont.load_default()
    _font_cache[size] = font
    return font


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_w: int) -> list[str]:
    """按像素宽度换行。优先在空格断;单个词本身超宽再按字符切。

    按字符切是为了兼容没有空格的文本(CJK、URL、超长 token)——MultiBanana 有
    multilingual text rendering 任务,纯按空格换行会让整段中文/日文挤成一行被裁。
    """
    text = text.replace("\n", " ")
    lines, cur = [], ""
    for word in text.split(" "):
        if word == "":
            continue
        trial = word if not cur else f"{cur} {word}"
        if draw.textlength(trial, font=font) <= max_w:
            cur = trial
            continue
        if cur:
            lines.append(cur)
            cur = ""
        # word 单独一个也可能超宽 → 逐字符塞
        piece = ""
        for ch in word:
            if draw.textlength(piece + ch, font=font) <= max_w:
                piece += ch
            else:
                if piece:
                    lines.append(piece)
                piece = ch
        cur = piece
    if cur:
        lines.append(cur)
    return lines or [""]


def labeled_tile(img: Image.Image, text: str, cell: int) -> Image.Image:
    """把图缩放到 cell 见方(保持比例、居中留白),顶部加一条短标签。"""
    canvas = Image.new("RGB", (cell, cell + _TILE_BAR), (255, 255, 255))
    w, h = img.size
    scale = min(cell / w, cell / h)
    resized = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    canvas.paste(resized, ((cell - resized.width) // 2, _TILE_BAR + (cell - resized.height) // 2))
    ImageDraw.Draw(canvas).text((4, 5), text[:38], fill=(0, 0, 0), font=_font(14))
    return canvas


def build_row(title: str, prompt: str, refs: list[Image.Image], results: dict,
              times: dict | None = None, cell: int = 256) -> Image.Image:
    """一行:[ref1..refN | 红线 | 各变体]。标题 `title | "prompt"` 完整多行显示。"""
    font = _font(14)
    tiles = [labeled_tile(r, f"ref{i + 1}", cell) for i, r in enumerate(refs)]
    for name, img in results.items():
        t = (times or {}).get(name)
        tiles.append(labeled_tile(img, f"{name}  {t:.1f}s" if t is not None else name, cell))

    gap, sep_at = 8, len(refs)
    total_w = len(tiles) * cell + gap

    # 先量算标题需要几行(用整行宽度减边距),再据此定标题区高度
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    header = f'{title}  |  "{prompt}"'
    lines = _wrap_text(probe, header, font, total_w - 8)
    title_h = len(lines) * _LINE_H + _TITLE_PAD

    row = Image.new("RGB", (total_w, cell + _TILE_BAR + title_h), (255, 255, 255))
    draw = ImageDraw.Draw(row)
    for i, ln in enumerate(lines):
        draw.text((4, 3 + i * _LINE_H), ln, fill=(0, 0, 0), font=font)

    x = 0
    for i, t in enumerate(tiles):
        if i == sep_at:
            x += gap
        row.paste(t, (x, title_h))
        x += cell
    if sep_at:  # 输入/输出分隔线
        lx = sep_at * cell + gap // 2
        draw.line([(lx, title_h), (lx, row.height)], fill=(200, 0, 0), width=2)
    return row


def stack_board(rows: list[Image.Image], bg=(255, 255, 255)) -> Image.Image:
    """把多行纵向拼成总览图(各行高度可不同,宽度取最大)。"""
    width = max(r.width for r in rows)
    board = Image.new("RGB", (width, sum(r.height for r in rows)), bg)
    y = 0
    for r in rows:
        board.paste(r, (0, y))
        y += r.height
    return board
