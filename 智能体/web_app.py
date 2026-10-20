# -*- coding: utf-8 -*-
"""
================================================================================
 智能体 Web 服务（前后端 + 联网搜索）
================================================================================
在 local_agent.py 五大模块之上增加：
  * 联网能力 —— web_search（Bing 搜索，免 API Key；DuckDuckGo 兜底）+ web_fetch（网页正文抓取）
  * 后端     —— FastAPI：多会话管理、SSE 实时流式输出思考/工具/观察全过程
  * 前端     —— web/index.html：浏览器聊天界面，含"联网搜索"开关

启动：
    python web_app.py                 # 默认 http://127.0.0.1:8000 ，自动打开浏览器
    python web_app.py --port 8080
    （或双击 start_web.bat）

接口一览：
    GET  /                     网页界面
    GET  /api/health           服务与 Ollama 状态
    GET  /api/models           本地模型列表
    POST /api/config           切换模型 / 温度
    POST /api/session          创建会话
    POST /api/chat/stream      对话（SSE 流式事件）
    POST /api/clear            清空某会话短期记忆
================================================================================
"""
import argparse
import gzip
import html as html_mod
import json
import os
import queue
import re
import threading
import time
import urllib.parse
import urllib.request
import uuid
import webbrowser
import zlib
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

import local_agent as la
from local_agent import Agent, CONFIG, LLMBrain, MemorySystem, ShortTermMemory, TaskPlanner, Tool, ToolBox

# =====================================================================
# 联网工具（仅使用 Python 标准库；无需任何 API Key）
# =====================================================================
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
_HEADERS = {"User-Agent": _UA, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate"}  # 不请求 br，避免依赖 brotli
FETCH_MAX_BYTES = 300_000     # web_fetch 解压后最多保留字节
FETCH_MAX_CHARS = 3000        # web_fetch 返回给模型的正文字数
SEARCH_RESULTS = 6            # 每次搜索返回条数

# ---------- 全局限流：按主机名控制最小请求间隔，防止高频访问触发搜索引擎验证码 ----------
_throttle_lock = threading.Lock()
_last_request_at: Dict[str, float] = {}
_THROTTLE_INTERVAL = {"www.so.com": 2.5, "so.com": 2.5, "news.so.com": 2.5}  # 秒


def _throttle(host: str) -> None:
    interval = _THROTTLE_INTERVAL.get(host, 0.4)
    with _throttle_lock:
        last = _last_request_at.get(host, 0.0)
        now = time.time()
        wait = last + interval - now
        _last_request_at[host] = now + max(wait, 0.0)
    if wait > 0:
        time.sleep(wait)


def _http_open(url: str, post_query: Optional[str] = None, timeout: int = 12):
    host = urllib.parse.urlsplit(url).hostname or ""
    _throttle(host)
    headers = dict(_HEADERS)
    if post_query is None:
        req = urllib.request.Request(url, headers=headers, method="GET")
    else:  # DuckDuckGo HTML 版使用 POST 表单
        body = urllib.parse.urlencode({"q": post_query}).encode("utf-8")
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    return urllib.request.urlopen(req, timeout=timeout)


def _read_response(resp, max_bytes: int):
    """读取响应并透明解压 gzip/deflate/br；返回 (最终URL, Content-Type, 解压后字节)。"""
    raw = resp.read()  # 压缩流必须完整读取才能解压
    encoding = (resp.headers.get("Content-Encoding") or "").lower()
    if "gzip" in encoding:
        raw = gzip.decompress(raw)
    elif "deflate" in encoding:
        try:
            raw = zlib.decompress(raw)
        except zlib.error:
            raw = zlib.decompress(raw, -zlib.MAX_WBITS)
    elif "br" in encoding:  # 个别服务器无视 Accept-Encoding 强制 br
        import brotli  # 仅在需要时依赖：pip install brotli
        raw = brotli.decompress(raw)
    if len(raw) > max_bytes:
        raw = raw[:max_bytes]
    return resp.geturl(), (resp.headers.get("Content-Type") or ""), raw


def _html_to_text(fragment: str) -> str:
    """HTML 片段 → 纯文本。"""
    fragment = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", fragment)
    fragment = re.sub(r"(?s)<!--.*?-->", " ", fragment)
    fragment = re.sub(r"<[^>]+>", " ", fragment)
    return re.sub(r"\s+", " ", html_mod.unescape(fragment)).strip()


# ---------- 结果日期解析：从标题/摘要中提取发布时间，用于时效性排序与新旧标注 ----------
def _parse_item_date(text: str):
    """返回 date 对象或 None。支持 2026-10-11 / 2026年10月11日 / 10月11日 / N天前 / 昨天 等。"""
    today = date.today()
    m = re.search(r"(20\d{2})\s*[-年/.]\s*(\d{1,2})\s*[-月/.]\s*(\d{1,2})", text)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    m = re.search(r"(\d{1,2})\s*月\s*(\d{1,2})\s*日", text)
    if m:
        mon, day = int(m.group(1)), int(m.group(2))
        year = today.year
        try:
            d = date(year, mon, day)
            if d > today:  # 日期在未来 → 实际是去年的内容
                d = date(year - 1, mon, day)
            return d
        except ValueError:
            return None
    m = re.search(r"(\d+)\s*(分钟|小时|天|周)前", text)
    if m:
        num = int(m.group(1))
        unit = {"分钟": "minutes", "小时": "hours", "天": "days", "周": "weeks"}[m.group(2)]
        return today - timedelta(**{unit: num})
    if "昨天" in text:
        return today - timedelta(days=1)
    if "前天" in text:
        return today - timedelta(days=2)
    return None


def _annotate_dates(items: List[Dict[str, Any]], fresh: bool) -> List[Dict[str, Any]]:
    """为每条结果解析日期；时效查询时按日期新→旧排序（无日期的沉底）。"""
    today = date.today()
    for it in items:
        d = _parse_item_date((it.get("date_hint") or "") + " " + it.get("title", "") + " " + it.get("snippet", ""))
        it["date"] = d
        it["age_days"] = (today - d).days if d else None
    if fresh:
        items.sort(key=lambda x: (x["date"] is None, -(x["date"].toordinal() if x["date"] else 0)))
    return items


_FRESH_WORDS = ("最新", "最近", "近期", "今天", "今日", "今晚", "今早", "近日", "现在", "目前", "眼前",
                "刚才", "本周", "本月", "今年", "新闻", "快讯", "头条", "热点", "实时", "刚刚",
                "latest", "recent", "today", "now", "current", "news", "this week")


def _is_fresh_query(query: str) -> bool:
    return any(w in query.lower() for w in _FRESH_WORDS)


# ---------- 搜索引擎 1：头条搜索（时效性最佳，含精确发布时间；新闻类查询首选） ----------
def _toutiao_search(query: str, n: int, fresh: bool = False) -> List[Dict[str, str]]:
    # fresh=True 走“资讯”标签页，纯新闻结果更多、更及时
    extra = "&pd=information&source=input&action_type=search_subtab_switch" if fresh else ""
    url = "https://so.toutiao.com/search?dvpf=pc&keyword=" + urllib.parse.quote(query) + extra
    with _http_open(url, timeout=12) as resp:
        _, _, raw = _read_response(resp, 4_000_000)
    page = raw.decode("utf-8", errors="replace")
    # 卡片 <script> 的属性顺序不固定，只锚定 data-druid-card-data-id，再逐个尝试 JSON 解析
    blocks = re.findall(
        r'<script[^>]*data-druid-card-data-id[^>]*>(.*?)</script>', page, re.S)
    results: List[Dict[str, Any]] = []
    seen_urls = set()
    for b in blocks:
        try:
            d = json.loads(b).get("data", {})
        except Exception:
            continue
        if not isinstance(d, dict) or not d.get("title"):
            continue
        link = d.get("source_url") or d.get("article_url") or d.get("open_url") or d.get("item_source_url") or ""
        link = html_mod.unescape(str(link)).split("?")[0] or html_mod.unescape(str(link))
        title = _html_to_text(str(d["title"]))
        snippet = _html_to_text(str(d.get("abstract") or ""))
        date_hint = str(d.get("datetime") or d.get("publish_time") or "")
        source = str(d.get("source") or "")
        if not title or not link or link in seen_urls:
            continue
        seen_urls.add(link)
        if snippet:
            snippet = f"（{source}）{snippet}" if source else snippet
        elif source:
            snippet = f"来源：{source}"
        results.append({"title": title, "url": link, "snippet": snippet, "date_hint": date_hint})
        if len(results) >= n:
            break
    return results


# ---------- 搜索引擎 2：360 搜索（免 Key，中文相关性好；带限流防验证码） ----------
def _360_search(query: str, n: int) -> List[Dict[str, str]]:
    url = "https://www.so.com/s?q=" + urllib.parse.quote(query)
    with _http_open(url, timeout=10) as resp:
        _, _, raw = _read_response(resp, 2_000_000)
    page = raw.decode("utf-8", errors="replace")
    if "访问异常页面" in page or "请输入验证码" in page:
        raise RuntimeError("360搜索触发验证码，自动切换其它引擎")  # 立即降级，不解析垃圾页
    results: List[Dict[str, str]] = []
    for seg in re.split(r'<li class="res-list[^"]*">', page)[1:]:
        hm = re.search(
            r'<h3[^>]*class="[^"]*res-title[^"]*"[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            seg, re.S)
        if not hm:
            continue
        link = html_mod.unescape(hm.group(1))
        title = _html_to_text(hm.group(2))
        snippet = ""
        sm = re.search(r'<p[^>]*class="[^"]*(?:res-desc|res-rich)[^"]*"[^>]*>(.*?)</p>', seg, re.S)
        if sm:
            snippet = _html_to_text(sm.group(1))
        if title and link:
            results.append({"title": title, "url": link, "snippet": snippet, "date_hint": ""})
        if len(results) >= n:
            break
    return results


# ---------- 搜索引擎 3：Bing（免 Key；作为次选兜底） ----------
def _bing_search(query: str, n: int) -> List[Dict[str, str]]:
    url = ("https://cn.bing.com/search?q=" + urllib.parse.quote(query)
           + "&setlang=zh-CN&ensearch=0&count=10")
    with _http_open(url, timeout=10) as resp:
        _, _, raw = _read_response(resp, 2_000_000)
    page = raw.decode("utf-8", errors="replace")
    results: List[Dict[str, str]] = []
    # 按每条结果的 <li class="b_algo..."> 宽松分段，段内取 h2 链接 + 首个摘要段落
    for seg in re.split(r'<li class="b_algo[^"]*"', page)[1:]:
        m = re.search(r'<h2><a[^>]*href="(http[^"]+)"[^>]*>(.*?)</a></h2>', seg, re.S)
        if not m:
            continue
        link, title = m.group(1), _html_to_text(m.group(2))
        tail = seg[m.end():]
        pm = re.search(r"<p[^>]*>(.*?)</p>", tail, re.S)
        snippet = _html_to_text(pm.group(1)) if pm else ""
        if title and link:
            results.append({"title": title, "url": link, "snippet": snippet, "date_hint": ""})
        if len(results) >= n:
            break
    return results


# ---------- 搜索引擎 4：DuckDuckGo HTML 版（最后兜底，国内网络可能超时） ----------
def _ddg_search(query: str, n: int) -> List[Dict[str, str]]:
    with _http_open("https://html.duckduckgo.com/html/", post_query=query, timeout=8) as resp:
        _, _, raw = _read_response(resp, 2_000_000)
    page = raw.decode("utf-8", errors="replace")
    results: List[Dict[str, str]] = []
    for m in re.finditer(r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', page, re.S):
        raw_link, title = m.group(1), _html_to_text(m.group(2))
        link = raw_link
        pm = re.search(r"uddg=([^&]+)", raw_link)  # DDG 跳转链接解开为真实地址
        if pm:
            link = urllib.parse.unquote(pm.group(1))
        seg = page[m.end(): m.end() + 4000]
        sm = re.search(r'class="result__snippet"[^>]*>(.*?)</a>|class="result__snippet"[^>]*>(.*?)</span>',
                       seg, re.S)
        snippet = _html_to_text(sm.group(1) or sm.group(2)) if sm else ""
        if title and link:
            results.append({"title": title, "url": link, "snippet": snippet, "date_hint": ""})
        if len(results) >= n:
            break
    return results


def web_search(query: str) -> str:
    """联网搜索工具：自动识别时效意图，多引擎取结果，按日期新旧标注排序。"""
    query = query.strip().strip("`\"'，。 ")
    if not query:
        return "错误：搜索关键词为空。"
    fresh = _is_fresh_query(query)
    today_str = time.strftime("%Y年%m月%d日")
    # 新闻/时效类查询：头条（带发布时间）优先；普通查询：360 优先
    engines = (("头条搜索", _toutiao_search), ("360搜索", _360_search), ("Bing", _bing_search)) \
        if fresh else (("360搜索", _360_search), ("头条搜索", _toutiao_search), ("Bing", _bing_search))
    errors: List[str] = []
    merged: List[Dict[str, str]] = []
    used_engines: List[str] = []
    seen_urls = set()
    for engine_name, engine_fn in engines:
        if len(merged) >= SEARCH_RESULTS:
            break
        try:
            items = engine_fn(query, SEARCH_RESULTS, fresh) if engine_fn is _toutiao_search \
                else engine_fn(query, SEARCH_RESULTS)
            if items:
                used_engines.append(engine_name)
                for it in items:
                    key = it["url"].split("?")[0]
                    if key in seen_urls:
                        continue
                    seen_urls.add(key)
                    merged.append(it)
            else:
                errors.append(f"{engine_name} 未返回结果")
        except Exception as e:  # 单个引擎失败不致命，继续尝试下一个
            errors.append(f"{engine_name} {type(e).__name__}")
    if merged:
        items = _annotate_dates(merged[:SEARCH_RESULTS], fresh)
        src = used_engines[0] if len(used_engines) == 1 else "、".join(used_engines)
        lines = [f"以下是通过 {src} 搜索「{query}」得到的 {len(items)} 条网络结果"
                 f"（今天是 {today_str}，结果已按时间新旧排序，请注意每条标注的发布日期）："]
        for i, it in enumerate(items, 1):
            tag = f"[发布于 {it['date'].isoformat()}] " if it.get("date") else "[日期未知] "
            lines.append(
                f"\n{i}. {tag}标题：{it['title']}\n   链接：{it['url']}\n   摘要：{it['snippet'][:300]}"
            )
        if fresh:
            lines.append(
                "\n【时效性要求】用户问的是“最近/今天/最新”类问题："
                "1) 只能采用发布日期接近今天（原则上30天内）的结果作答，日期未知或明显久远的内容一律视为旧闻，不得当作新闻；"
                "2) 若所有结果日期都很旧，必须直接告知“未检索到近期信息”，绝不能拿旧闻冒充；"
                "3) 回答中写明每条新闻的日期与来源。")
        lines.append(
            "通用规则：只能直接复制上面列出的完整链接调用 web_fetch，严禁猜测网址；"
            "区分“预测/计划”与“已发生”的事实；回答末尾附来源链接。")
        return "\n".join(lines)
    return ("网络搜索失败（" + "；".join(errors) + "）。"
            "【重要】本次未能从互联网获取到任何结果。"
            "如果用户问的是新闻/天气/股价/赛事等时效性内容，必须直接回答“未能检索到相关信息”，"
            "绝对不要凭记忆编造新闻、日期或链接；"
            "只有在用户问题与时效性无关时，才可以基于已有知识作答，并明确标注“未经联网核实”。")


def _extract_js_redirect(raw: bytes) -> Optional[str]:
    """识别搜索引擎跳转中间页（页面极短，含 location.replace / meta refresh），提取目标 URL。"""
    if len(raw) > 5000:
        return None
    head = raw.decode("utf-8", errors="ignore")
    low = head.lower()
    m = re.search(r"""location\.(?:replace|href)\s*=?\s*['"](https?://[^'"]+)['"]""", head, re.I)
    if not m and ("http-equiv" in low and "refresh" in low):
        m = re.search(r"""url\s*=\s*['"]?(https?://[^'">\s]+)""", head, re.I)
    return html_mod.unescape(m.group(1)).strip("'\"") if m else None


def web_fetch(raw_url: str) -> str:
    """网页正文抓取工具：打开指定 URL，提取标题与正文文本（自动穿透 JS/meta 跳转中间页）。"""
    m = re.search(r"https?://[^\s\]\)）>\"']+", raw_url)
    if not m:
        return "错误：未在参数中找到有效的 http(s) 网址。"
    cur = m.group(0).rstrip(".,;，。；")
    try:
        final_url, ctype, raw, page = cur, "", b"", ""
        for _hop in range(3):  # 最多跟随 3 次 JS/meta 跳转
            with _http_open(cur, timeout=15) as resp:
                final_url, ctype, raw = _read_response(resp, FETCH_MAX_BYTES)
            if "pdf" in ctype or "image" in ctype or "octet-stream" in ctype or "video" in ctype:
                return f"错误：该链接不是文本网页（Content-Type: {ctype or '未知'}），暂不支持抓取。"
            nxt = _extract_js_redirect(raw)
            if nxt and nxt != cur:
                cur = nxt
                continue
            break
        # 编码探测：HTTP 头 → meta charset → utf-8
        encoding = "utf-8"
        cm = re.search(r"charset=([\w-]+)", ctype)
        if cm:
            encoding = cm.group(1)
        page = raw.decode(encoding, errors="replace")
        if "\ufffd" in page[:2000] and encoding.lower() != "gbk":
            page = raw.decode("gbk", errors="replace")
        tm = re.search(r"(?is)<title[^>]*>(.*?)</title>", page)
        title = _html_to_text(tm.group(1)) if tm else "(无标题)"
        body = re.sub(r"(?is)<(script|style|noscript|iframe|svg|header|footer|nav)[^>]*>.*?</\1>",
                      " ", page)
        text = _html_to_text(body)[:FETCH_MAX_CHARS]
        return (f"网页标题：{title}\n网页链接：{final_url}\n"
                f"正文（已截取前 {FETCH_MAX_CHARS} 字）：\n{text}")
    except Exception as e:
        return f"网页抓取失败：{type(e).__name__}: {e}。可换用其它链接或仅依据搜索摘要回答。"


# =====================================================================
# 会话管理：每个浏览器会话独立短期记忆；长期记忆（同名集合）全局共享
# =====================================================================
class Session:
    def __init__(self, sid: str, brain: LLMBrain):
        self.sid = sid
        self.lock = threading.Lock()       # 同一会话的请求串行，避免上下文错乱
        self.created = time.strftime("%Y-%m-%d %H:%M:%S")
        toolbox = ToolBox()
        toolbox.register(Tool(
            "web_search",
            "联网搜索互联网上的最新信息（新闻、天气、股价、赛事、实时资讯等），返回标题/链接/摘要",
            web_search,
            "搜索关键词，如 2026年诺贝尔奖 最新消息"))
        toolbox.register(Tool(
            "web_fetch",
            "打开指定网页链接并提取正文全文，用于阅读搜索结果中某条链接的详细内容",
            web_fetch,
            "完整网页 URL，如 https://example.com/article/123"))
        self.memory = MemorySystem(brain)
        self.agent = Agent(brain, TaskPlanner(), self.memory, toolbox, debug=CONFIG["debug"])


class SessionManager:
    def __init__(self, brain: LLMBrain):
        self.brain = brain
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()

    def get_or_create(self, sid: Optional[str]) -> Session:
        sid = sid or uuid.uuid4().hex
        with self._lock:
            sess = self._sessions.get(sid)
            if sess is None:
                sess = Session(sid, self.brain)
                self._sessions[sid] = sess
            return sess

    def create(self) -> Session:
        return self.get_or_create(None)


# =====================================================================
# FastAPI 应用
# =====================================================================
app = FastAPI(title="本地智能体 Web 服务", version="1.1")
_brain = LLMBrain(CONFIG["ollama_host"], CONFIG["model"], CONFIG["temperature"],
                  CONFIG["num_ctx"], CONFIG["num_predict"])
manager = SessionManager(_brain)
_web_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
# 工具白名单
_BASE_TOOLS = ["calculator", "read_file"]
_WEB_TOOLS = ["web_search", "web_fetch"]

# 算力档位（类似云端模型的“额度利用”档位）：越高生成越长、ReAct 迭代越深
_UTILIZATION_LEVELS = {
    1: {"name": "极速", "num_predict": 256, "max_steps": 3},
    2: {"name": "轻快", "num_predict": 512, "max_steps": 4},
    3: {"name": "均衡", "num_predict": 1024, "max_steps": 6},
    4: {"name": "详尽", "num_predict": 1536, "max_steps": 8},
    5: {"name": "深度", "num_predict": 2048, "max_steps": 10},
}
_current_utilization = 3  # 默认均衡


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    web_enabled: bool = True


class ConfigRequest(BaseModel):
    model: Optional[str] = None
    temperature: Optional[float] = None
    utilization: Optional[int] = None  # 算力档位 1~5


class ClearRequest(BaseModel):
    session_id: str


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/")
def index_page():
    return FileResponse(os.path.join(_web_dir, "index.html"))


@app.get("/api/health")
def health():
    model_details = _brain.list_models_detail()
    mem_count = 0
    try:  # 长期记忆条数（复用任意一个已建会话，否则临时探测）
        any_sess = next(iter(manager._sessions.values()), None)
        mem_count = any_sess.memory.long.count() if any_sess else 0
    except Exception:
        pass
    return {
        "ollama_ok": bool(model_details),
        "model": _brain.model,
        "models": [m["name"] for m in model_details],
        "model_details": model_details,
        "temperature": _brain.temperature,
        "utilization": _current_utilization,
        "utilization_name": _UTILIZATION_LEVELS[_current_utilization]["name"],
        "long_term_memories": mem_count,
    }


@app.get("/api/models")
def list_models():
    details = _brain.list_models_detail()
    return {"models": [m["name"] for m in details], "details": details}


@app.post("/api/config")
def set_config(req: ConfigRequest):
    global _current_utilization
    if req.model:
        installed = _brain.list_models()
        if req.model not in installed:
            return {"ok": False, "error": f"模型 {req.model} 未安装", "models": installed}
        _brain.model = req.model
    if req.temperature is not None:
        _brain.temperature = max(0.0, min(1.0, float(req.temperature)))
    if req.utilization is not None:
        lvl = max(1, min(5, int(req.utilization)))
        _current_utilization = lvl
        cfg = _UTILIZATION_LEVELS[lvl]
        _brain.num_predict = cfg["num_predict"]
        CONFIG["max_react_steps"] = cfg["max_steps"]
    return {"ok": True, "model": _brain.model, "temperature": _brain.temperature,
            "utilization": _current_utilization,
            "utilization_name": _UTILIZATION_LEVELS[_current_utilization]["name"]}


@app.post("/api/session")
def create_session():
    sess = manager.create()
    return {"session_id": sess.sid, "created": sess.created}


@app.post("/api/clear")
def clear_session(req: ClearRequest):
    sess = manager._sessions.get(req.session_id)
    if sess:
        sess.memory.short.clear()
        return {"ok": True}
    return {"ok": False, "error": "会话不存在"}


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    """SSE 流式对话：事件顺序 thought/action/observation ... → final → done（或 error → done）。"""
    sess = manager.get_or_create(req.session_id)
    sid = sess.sid
    message = req.message.strip()
    allowed = _BASE_TOOLS + _WEB_TOOLS if req.web_enabled else _BASE_TOOLS

    def event_generator():
        q: "queue.Queue[tuple]" = queue.Queue()

        def callback(event: str, data: Any) -> None:
            q.put((event, data))

        def worker() -> None:
            try:
                answer = sess.agent.run(message, event_callback=callback, allowed_tools=allowed)
                q.put(("final", {"text": answer}))
            except Exception as e:
                q.put(("error", {"message": str(e)}))
            finally:
                q.put(("done", {"session_id": sid}))

        yield _sse("session", {"session_id": sid})
        if not message:
            yield _sse("error", {"message": "消息为空"})
            yield _sse("done", {"session_id": sid})
            return

        t = threading.Thread(target=worker, daemon=True)
        with sess.lock:  # 同一会话串行执行
            t.start()
            while True:
                event, data = q.get()
                yield _sse(event, data)
                if event == "done":
                    break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="本地智能体 Web 服务（FastAPI + 联网搜索）")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认仅本机；0.0.0.0 可局域网访问）")
    parser.add_argument("--port", type=int, default=8000, help="端口号")
    args = parser.parse_args()

    import uvicorn
    url = f"http://{args.host if args.host != '0.0.0.0' else '127.0.0.1'}:{args.port}"
    if not _brain.list_models():
        print("⚠ 未检测到 Ollama 服务或本地模型，网页可打开但对话将报错。")
        print("  请先安装 Ollama 并执行：ollama pull qwen2.5:3b")
    print(f"智能体网页服务启动中… 访问地址：{url}")
    print("（浏览器未自动打开时，可手动复制上面的地址）")
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
