# UNO Distill Viewer

FastAPI 版人工质检面板，替代原来的单一静态 `inspect.html`。

## 特性

- **前后端分离**：FastAPI + JSON 文件持久化标注，前端原生 JS/CSS
- **标注持久化**：pass/fail/备注自动保存到服务器，刷新不丢失
- **图片安全服务**：`/api/image?rel=...` 仅放行 `datasets/distill_multiref/`、`datasets/dreambooth/dataset/`
- **保留原有交互**：懒加载、图片放大/缩放/拖拽、左右切换、快捷键 `p`/`f`
- **导入/导出**：与旧版 `inspect.html` 的 annotations.json 兼容
- **过滤**："只看未标注"一键隐藏已标记行
- **服务端分页**：默认 5 条,可切 5/10/20/50/100,DOM 只渲染当前页,避免 8000 条卡顿
- **自动续标**：打开自动跳到第一个未标注行,工具栏也提供"跳到未标注"按钮

## 启动

```bash
cd /kaimm-distill/wuwenxuan/UNO
source .venv-uno/bin/activate
python -m distill.viewer.server
```

默认打开 http://localhost:8000

## 常用参数

```bash
# 只看前 200 条
python -m distill.viewer.server --limit 200

# 随机抽 200 条
python -m distill.viewer.server --limit 200 --shuffle

# 只看 shard0（第 0-999 条）
python -m distill.viewer.server --shard 0

# 换端口
python -m distill.viewer.server --port 8080

# 自定义 manifest / 标注输出位置
python -m distill.viewer.server \
  --manifest datasets/distill_multiref/manifest_raw.json \
  --out-annotations datasets/distill_multiref/annotations.json
```

## API

- `GET /`：前端页面
- `GET /static/*`：静态资源
- `GET /api/health`：健康检查
- `GET /api/manifest?page=N&per_page=M`：分页获取 manifest 元数据，返回 `n_rows/page/pages/per_page/rows`
- `GET /api/annotations`：读取当前标注
- `POST /api/annotations`：保存标注（请求体 `{ "annotations": {...} }`）
- `GET /api/image?rel=...`：安全返回图片

## 文件说明

```
distill/viewer/
├── server.py       # FastAPI 服务入口
├── manifest.py     # manifest 加载/过滤
├── storage.py      # annotations.json 读写
└── static/
    ├── index.html  # 前端页面骨架
    ├── style.css   # 样式（提取自 inspect_html.py）
    ├── viewer.js   # 图片查看器（提取自 inspect_html.py）
    └── anno.js     # 标注逻辑（改为 API 驱动）
```

## 与旧版的关系

`distill/inspect_html.py` 继续保留，用于生成单一静态 HTML 备份：

```bash
python distill/inspect_html.py --limit 200
```

新版服务不再生成大 HTML，数据全走 API 按需加载。
