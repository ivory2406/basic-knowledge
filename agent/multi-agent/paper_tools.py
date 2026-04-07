"""
论文搜索、下载与解析工具集

提供给 Multi-Agent 使用的工具函数：
1. search_papers    - 在 arXiv 上搜索论文
2. download_paper   - 下载 PDF 到本地
3. extract_pdf_text - 提取 PDF 文本内容
4. read_paper_text  - 读取已下载论文的文本（供分析 Agent 使用）
"""

import os
import re
import ssl
import json
import urllib.request
from pathlib import Path

import arxiv

# 处理 SSL 证书验证问题（某些服务器环境无完整证书链）
ssl._create_default_https_context = ssl._create_unverified_context

# ====================================================================
# 配置
# ====================================================================

# 论文下载目录
PAPER_DIR = Path(__file__).parent / "papers"
PAPER_DIR.mkdir(exist_ok=True)

# 论文元数据索引文件
INDEX_FILE = PAPER_DIR / "paper_index.json"


def _load_index() -> dict:
    """加载论文索引。"""
    if INDEX_FILE.exists():
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_index(index: dict) -> None:
    """保存论文索引。"""
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def _sanitize_filename(name: str) -> str:
    """将论文标题转为安全的文件名。"""
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = name.replace(" ", "_")
    return name[:80]


# ====================================================================
# 工具函数
# ====================================================================

def search_papers(query: str, max_results: int = 10) -> str:
    """在 arXiv 上搜索论文。

    Args:
        query: 搜索关键词，如 "LLM long-term memory benchmark"
        max_results: 最大返回数量，默认 10

    Returns:
        搜索结果的格式化字符串，包含标题、作者、摘要、arxiv_id 等
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=min(max_results, 20),
        sort_by=arxiv.SortCriterion.Relevance,
    )

    results = []
    for paper in client.results(search):
        results.append({
            "arxiv_id": paper.entry_id.split("/")[-1],
            "title": paper.title,
            "authors": ", ".join(a.name for a in paper.authors[:5]),
            "published": paper.published.strftime("%Y-%m-%d"),
            "abstract": paper.summary[:500],
            "pdf_url": paper.pdf_url,
            "categories": [c for c in paper.categories],
        })

    if not results:
        return f"未找到与 '{query}' 相关的论文。请尝试调整关键词。"

    output_lines = [f"🔍 搜索 '{query}' 共找到 {len(results)} 篇论文：\n"]
    for i, r in enumerate(results, 1):
        output_lines.append(
            f"[{i}] {r['title']}\n"
            f"    arxiv_id: {r['arxiv_id']}\n"
            f"    作者: {r['authors']}\n"
            f"    发布日期: {r['published']}\n"
            f"    分类: {', '.join(r['categories'])}\n"
            f"    摘要: {r['abstract'][:300]}...\n"
        )

    # 保存到索引
    index = _load_index()
    for r in results:
        index[r["arxiv_id"]] = r
    _save_index(index)

    return "\n".join(output_lines)


def download_paper(arxiv_id: str) -> str:
    """下载指定 arXiv ID 的论文 PDF。

    Args:
        arxiv_id: arXiv 论文 ID，如 "2504.14225" 或 "2410.10813"

    Returns:
        下载结果信息
    """
    # 清理 ID
    arxiv_id = arxiv_id.strip()
    if "/" in arxiv_id:
        arxiv_id = arxiv_id.split("/")[-1]

    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])

    try:
        paper = next(client.results(search))
    except StopIteration:
        return f"❌ 未找到 arxiv_id={arxiv_id} 的论文，请检查 ID 是否正确。"

    filename = _sanitize_filename(paper.title) + ".pdf"
    filepath = PAPER_DIR / filename

    if filepath.exists():
        return (
            f"📄 论文已存在：{filename}\n"
            f"   标题: {paper.title}\n"
            f"   路径: {filepath}"
        )

    paper.download_pdf(dirpath=str(PAPER_DIR), filename=filename)

    # 更新索引
    index = _load_index()
    entry = index.get(arxiv_id, {})
    entry.update({
        "arxiv_id": arxiv_id,
        "title": paper.title,
        "authors": ", ".join(a.name for a in paper.authors[:5]),
        "published": paper.published.strftime("%Y-%m-%d"),
        "abstract": paper.summary,
        "pdf_url": paper.pdf_url,
        "local_pdf": str(filepath),
        "local_filename": filename,
    })
    index[arxiv_id] = entry
    _save_index(index)

    return (
        f"✅ 下载成功！\n"
        f"   标题: {paper.title}\n"
        f"   文件: {filename}\n"
        f"   路径: {filepath}\n"
        f"   大小: {filepath.stat().st_size / 1024:.1f} KB"
    )


def extract_pdf_text(arxiv_id: str, max_chars: int = 15000) -> str:
    """提取已下载论文 PDF 的文本内容。

    Args:
        arxiv_id: arXiv 论文 ID
        max_chars: 最大提取字符数，默认 15000（约覆盖摘要+引言+方法+结论）

    Returns:
        提取的文本内容
    """
    index = _load_index()

    if arxiv_id not in index:
        return f"❌ 论文 {arxiv_id} 未在索引中，请先使用 search_papers 搜索或 download_paper 下载。"

    entry = index[arxiv_id]
    local_pdf = entry.get("local_pdf")

    if not local_pdf or not Path(local_pdf).exists():
        return f"❌ 论文 {arxiv_id} 的 PDF 文件未下载，请先使用 download_paper 下载。"

    try:
        import pymupdf
        doc = pymupdf.open(local_pdf)
    except ImportError:
        try:
            import fitz  # PyMuPDF legacy import
            doc = fitz.open(local_pdf)
        except ImportError:
            return (
                "❌ 需要安装 pymupdf 来解析 PDF。请运行: pip install pymupdf\n"
                f"论文信息:\n  标题: {entry.get('title')}\n  摘要: {entry.get('abstract', 'N/A')[:500]}"
            )

    text_parts = []
    total_chars = 0
    for page in doc:
        page_text = page.get_text()
        text_parts.append(page_text)
        total_chars += len(page_text)
        if total_chars >= max_chars:
            break
    doc.close()

    full_text = "\n".join(text_parts)
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars] + "\n\n... [文本已截断，共提取前 {} 字符]".format(max_chars)

    return (
        f"📖 论文: {entry.get('title')}\n"
        f"   arxiv_id: {arxiv_id}\n"
        f"   作者: {entry.get('authors', 'N/A')}\n"
        f"{'=' * 60}\n"
        f"{full_text}"
    )


def list_downloaded_papers() -> str:
    """列出所有已下载的论文。

    Returns:
        已下载论文列表
    """
    index = _load_index()

    downloaded = {k: v for k, v in index.items() if v.get("local_pdf")}

    if not downloaded:
        return "📭 暂无已下载的论文。请使用 search_papers 搜索后用 download_paper 下载。"

    lines = [f"📚 已下载 {len(downloaded)} 篇论文：\n"]
    for arxiv_id, entry in downloaded.items():
        pdf_exists = Path(entry["local_pdf"]).exists() if entry.get("local_pdf") else False
        status = "✅" if pdf_exists else "❌ (文件丢失)"
        lines.append(
            f"  {status} [{arxiv_id}] {entry.get('title', 'N/A')}\n"
            f"     发布: {entry.get('published', 'N/A')} | 文件: {entry.get('local_filename', 'N/A')}\n"
        )

    return "\n".join(lines)
