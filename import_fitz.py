import fitz
import re

# 可扩展的字体“粗体”关键字集合（可根据语言/字体补充）
BOLD_KEYWORDS = ["bold", "bd", "black", "heavy", "semibold", "demibold", "hei", "黑体", "粗体"]

# 判断一个字体名是否看起来是加粗（启发式）
def font_name_is_bold(fontname: str) -> bool:
    if not fontname:
        return False
    name = fontname.lower()
    for kw in BOLD_KEYWORDS:
        if kw in name:
            return True
    return False

# 判断 token 是否为题号候选（数字或带括号的数字）
def token_is_number(token: str) -> bool:
    token = token.strip()
    # 匹配：1 1. 1） 1) （1） (1) 第1题  等常见格式（可自行扩展）
    patterns = [
        r"^\d+$",
        r"^\d+\.$",
        r"^\d+[）\)]$",
        r"^[（(]\d+[)）]$",
        r"^第\d+题$"
    ]
    for p in patterns:
        if re.fullmatch(p, token):
            return True
    return False

def find_leftmost_bold_numbers_on_page(page, left_ratio=0.3, y_margin=(0.05, 0.95), size_min=None):
    """
    在单页中寻找满足条件的题号（数字 & 字体名显示为 bold & x0 是最小/接近最小）。
    left_ratio: 只在页面左侧 left_ratio 的宽度内寻找候选（防止页眉/页脚干扰）
    y_margin: 忽略页面顶部/底部的 y 区域（比例）
    size_min: 若提供，排除字号小于该值的文本（避免页码/注释）
    返回：列表 of dict: { 'token': str, 'x0': float, 'y0': float, 'bbox': (x0,y0,x1,y1), 'font': str, 'size': float, 'bold': bool }
    """
    page_dict = page.get_text("dict")
    page_width = page.mediabox_size[0]
    page_height = page.mediabox_size[1]

    # 收集所有 word-level 信息（来自 spans -> words）
    words = []  # 每项: (x0,y0,x1,y1, text, font, size)
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:  # 0 == text
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                font = span.get("font", "")
                size = span.get("size", 0)
                text = span.get("text", "")
                # spans 的 text 可能包含多个词，需要按空格拆分并估算位置
                # 更精确的方法是使用 page.get_text("words"), 但这里我们也可以从 spans 中拆词
                # 使用 page.get_text("words") 得到更准确的 x0,x1
    # 下面使用 words 输出（更精确）
    raw_words = page.get_text("words")  # (x0, y0, x1, y1, "text", block_no, line_no, word_no)
    # 要把每个 word 映射到对应的 span 的 font/size：使用 rawdict 的 spans 字符信息更精确（含 chars）
    # 为简洁起见，我们 will try to map by spatial overlap: find span that overlaps the word bbox
    raw_dict = page.get_text("rawdict")
    spans_index = []
    for block in raw_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                spans_index.append({
                    "bbox": span.get("bbox"),  # [x0,y0,x1,y1]
                    "font": span.get("font"),
                    "size": span.get("size"),
                    "text": span.get("text")
                })

    # helper: test overlap of boxes
    def bbox_overlap(a, b):
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)

    for w in raw_words:
        x0, y0, x1, y1, txt, *_ = w
        # filter by y margin
        if not (page_height * y_margin[0] <= y0 <= page_height * y_margin[1]):
            continue
        # filter by left area
        if x0 > page_width * left_ratio:
            continue
        # optional size filter: will be filled from matched span
        matched_font = ""
        matched_size = None
        for s in spans_index:
            if bbox_overlap((x0, y0, x1, y1), tuple(s["bbox"])):
                matched_font = s.get("font", "")
                matched_size = s.get("size", None)
                break
        if size_min is not None and matched_size is not None and matched_size < size_min:
            continue
        words.append({
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "text": txt.strip(),
            "font": matched_font,
            "size": matched_size
        })

    if not words:
        return []

    # 计算页面的最小 x0（在我们筛选后的正文候选词中）
    min_x0 = min(w["x0"] for w in words)

    # 现在筛选：文本为数字/题号样式 & font 看起来是 bold & x0 接近 min_x0
    candidates = []
    for w in words:
        txt = w["text"]
        if not txt:
            continue
        # 把类似 "1." 或 "1" 或 "(1)" 等作为 token 前缀匹配
        token = txt
        # strip trailing punctuation often attached like "1." or "1)"
        token_norm = token.strip().strip(".、)）")
        # 判断是否数字
        if not re.fullmatch(r"\d+", token_norm):
            continue
        # 判断bold（启发式）
        bold = font_name_is_bold(w.get("font", ""))
        # 判断是否最左：我们允许小容差，如 2-3 个像素或 1% 页宽
        tol = max(3.0, page_width * 0.01)
        is_leftmost = abs(w["x0"] - min_x0) <= tol
        if bold and is_leftmost:
            candidates.append({**w, "token": token, "bold": bold})
    return candidates

# 用候选题号来分割页面题目（基于行索引）
def extract_questions_using_candidates(pdf_path):
    doc = fitz.open(pdf_path)
    all_qs = []
    for pi, page in enumerate(doc):
        candidates = find_leftmost_bold_numbers_on_page(page)
        # 如果没有候选，可以放宽条件（例如允许非加粗或扩大 left_ratio）
        if not candidates:
            candidates = find_leftmost_bold_numbers_on_page(page, left_ratio=0.4)
        # map candidate y0 -> sorted
        candidates_sorted = sorted(candidates, key=lambda c: c["y0"])
        # 如果还是空，跳过或记录页面为需人工复核
        if not candidates_sorted:
            continue

        # 为了分割题目，我们也需要 page 的行文本（用 words -> 行聚合）
        words = page.get_text("words")
        words.sort(key=lambda w: (w[3], w[0]))
        lines = {}
        for w in words:
            x0,y0,x1,y1,txt, *_ = w
            y_key = round(y0 / 5) * 5
            lines.setdefault(y_key, []).append((x0, txt))
        sorted_lines = sorted(lines.items(), key=lambda x: x[0])
        page_lines = ["".join(t for _, t in sorted(line, key=lambda x: x[0])) for _, line in sorted_lines]

        # 把候选的 y0 转成行索引（找到最接近的行）
        candidate_line_idxs = []
        line_ys = [y for y,_ in sorted_lines]
        for c in candidates_sorted:
            # 找最接近的 y_key index
            y = c["y0"]
            # find index
            best_idx = min(range(len(line_ys)), key=lambda i: abs(line_ys[i]-y))
            candidate_line_idxs.append((best_idx, c["token"]))

        # 去重并排序
        candidate_line_idxs = sorted(dict(candidate_line_idxs).items(), key=lambda x: x[0])

        # 分割题目
        for i, (start_idx, token) in enumerate(candidate_line_idxs):
            end_idx = candidate_line_idxs[i+1][0] if i+1 < len(candidate_line_idxs) else len(page_lines)
            q_text = "\n".join(page_lines[start_idx:end_idx])
            all_qs.append({
                "page": pi+1,
                "token": token,
                "y0": sorted(candidates_sorted, key=lambda c: c["y0"])[i]["y0"],
                "content": q_text
            })
    return all_qs

# 示例用法
if __name__ == "__main__":
    path = "example.pdf"
    qs = extract_questions_using_candidates(path)
    for q in qs:
        print(f"Page {q['page']} token={q['token']} y0={q['y0']}\n{q['content'][:200]}...\n---")
