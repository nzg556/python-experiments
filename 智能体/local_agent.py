# -*- coding: utf-8 -*-
"""
================================================================================
 本地可运行完整智能体（5 大核心模块 + 4 大进阶扩展方向预留接口）
================================================================================
特点：
  * 完全本地运行 —— Ollama 本地大模型 + Chroma 本地向量库，无需任何外网 API / GPU
  * 标准 ReAct 智能体循环 —— 思考 → 规划 → 调用工具 → 观察 → 迭代 → 输出结果
  * 双记忆系统 —— 短期对话记忆 + Chroma 长期向量记忆（重启不丢失、自动召回）

【运行前准备】
 1. 安装 Ollama（https://ollama.com），并拉取一个本地模型（CPU 即可运行）：
       ollama pull qwen2.5:3b
 2. 安装 Python 依赖（Python 3.10+）：
       pip install chromadb
 3. 确认 Ollama 服务已启动（安装后一般自动后台运行；也可手动执行：ollama serve）
 4. 运行本程序：
       python local_agent.py                          # 自动选择本地可用模型
       python local_agent.py --model qwen2.5:3b --debug

【对话内命令】
 /help 查看命令     /tools 查看工具     /memory 查看记忆条数   /clear 清空短期记忆
 /model 模型名 切换模型                /temp 数值 调整温度    /debug 调试开关    /exit 退出

【代码结构（与需求规范一一对应）】
   模块一  LLM 大脑模块          LLMBrain          本地推理 / 嵌入向量 / 模型管理
   模块二  任务规划模块          TaskPlanner       ReAct 提示词 / 输出解析 / 步数控制
   模块三  工具调用模块          Tool / ToolBox    内置计算器、文件读取，支持装饰器扩展
   模块四  双记忆系统            ShortTermMemory / LongTermMemory / MemorySystem
   模块五  自主执行循环模块      Agent             记忆召回→规划→工具→观察→迭代→输出
   进阶1   单任务专精智能体      create_vertical_agent + KnowledgeBase
   进阶2   多智能体协作系统      MultiAgentSystem
   进阶3   具身智能体硬件对接    HardwareInterface
   进阶4   工程化部署（FastAPI） create_fastapi_app
================================================================================
"""
import argparse
import ast
import hashlib
import json
import math
import operator as op
import os
import re
import sys
import time
import urllib.request
import uuid
from collections import deque
from typing import Any, Callable, Dict, Iterable, List, Optional

# =====================================================================
# 全局配置（可按需修改）
# =====================================================================
CONFIG: Dict[str, Any] = {
    "ollama_host": "http://localhost:11434",  # Ollama 服务地址
    "model": "qwen2.5:3b",                    # 默认模型（若本地不存在会自动改选其它已装模型）
    "temperature": 0.7,                       # 温度：越高越有创造性，越低越严谨
    "num_ctx": 4096,                          # 上下文窗口长度（可调大以支持更长任务）
    "num_predict": 1024,                      # 单次生成最大 token 数（算力档位控制）
    "max_react_steps": 6,                     # 单个任务 ReAct 最大迭代步数（算力档位控制）
    "llm_retries": 3,                         # 大模型调用失败自动重试次数
    "short_term_turns": 10,                   # 短期记忆保留的对话轮数
    "long_term_dir": "./agent_memory",        # 长期记忆（Chroma）持久化目录
    "memory_collection": "agent_long_term_memory",
    "recall_top_k": 3,                        # 每次提问自动召回的历史记忆条数
    "recall_max_distance": 0.85,              # 召回相似度阈值（cosine 距离，越小越相关）
    "embed_model": "nomic-embed-text",        # Ollama 向量模型；不可用时自动降级为内置本地向量
    "max_obs_chars": 6000,                    # 工具观察结果注入上下文的最大字符数
    "max_file_bytes": 200_000,                # 文件读取工具单次最大读取字节数
    "debug": False,                           # 调试模式：可视化打印思考全过程
}


# =====================================================================
# 模块一：LLM 大脑模块（推理核心）
#   本地私有化大模型推理；支持切换模型、调节温度与上下文窗口；
#   具备自然语言理解、任务推理、自我反思、分步思考能力（由 ReAct 循环驱动）。
# =====================================================================
class LLMBrain:
    """Ollama 本地大模型推理核心：对话生成、向量嵌入、模型管理。"""

    def __init__(self, host: str, model: str, temperature: float, num_ctx: int,
                 num_predict: int = 1024):
        self.host = host.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self._embed_fallback_warned = False

    # ---------- 底层 HTTP（仅依赖标准库，无需外网） ----------
    def _post(self, path: str, payload: dict, timeout: int) -> dict:
        url = self.host + path
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # ---------- 对话推理（带自动重试） ----------
    def chat(self, messages: List[Dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx,
                        "num_predict": self.num_predict},
        }
        last_err: Optional[Exception] = None
        for attempt in range(1, CONFIG["llm_retries"] + 1):
            try:
                data = self._post("/api/chat", payload, timeout=600)
                return (data.get("message", {}).get("content") or "").strip()
            except Exception as e:  # 网络抖动 / 服务重启等，自动重试
                last_err = e
                if CONFIG["debug"]:
                    print(f"  [LLM] 第 {attempt} 次调用失败：{e}")
                time.sleep(1.5 * attempt)
        raise RuntimeError(
            f"调用本地大模型失败（已重试 {CONFIG['llm_retries']} 次）：{last_err}\n"
            f"请确认 Ollama 正在运行：命令行执行 `ollama serve`，或检查地址 {self.host}"
        )

    # ---------- 向量嵌入（用于长期记忆） ----------
    def embed(self, text: str) -> List[float]:
        """优先使用 Ollama 向量模型；不可用时自动降级为内置纯本地向量，保证开箱即用。"""
        try:
            data = self._post(
                "/api/embeddings",
                {"model": CONFIG["embed_model"], "prompt": text[:2000]},
                timeout=60,
            )
            emb = data.get("embedding")
            if emb:
                return emb
        except Exception:
            pass
        if not self._embed_fallback_warned:
            self._embed_fallback_warned = True
            if CONFIG["debug"]:
                print(f"  [记忆] 未检测到向量模型 {CONFIG['embed_model']}，已切换为内置本地向量。")
        return self._hash_embed(text)

    @staticmethod
    def _hash_embed(text: str, dim: int = 512) -> List[float]:
        """纯 Python 轻量本地向量：英文按词、中文按字 bigram，符号哈希投影，完全离线。"""
        text = re.sub(r"\s+", " ", text.lower())
        tokens = re.findall(r"[a-z0-9]+", text)                      # 英文/数字词
        cjk = re.sub(r"[^\u4e00-\u9fff]", "", text)                  # 中文字符
        if len(cjk) >= 2:
            tokens += [cjk[i: i + 2] for i in range(len(cjk) - 1)]   # 中文 bigram
        elif cjk:
            tokens.append(cjk)
        vec = [0.0] * dim
        for tok in tokens:                                           # 符号哈希：随机投影近似正交
            digest = hashlib.md5(tok.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % dim
            vec[idx] += 1.0 if digest[4] % 2 == 0 else -1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    # ---------- 模型管理 ----------
    def list_models(self) -> List[str]:
        """列出本地已安装的模型（通过 Ollama /api/tags）。"""
        return [m["name"] for m in self.list_models_detail()]

    def list_models_detail(self) -> List[Dict[str, Any]]:
        """列出本地模型详情：名称、参数量、量化等级、文件大小(GB)。"""
        try:
            with urllib.request.urlopen(self.host + "/api/tags", timeout=5) as resp:
                models = json.loads(resp.read().decode("utf-8")).get("models", [])
            details: List[Dict[str, Any]] = []
            for m in models:
                d = m.get("details", {})
                details.append({
                    "name": m.get("name", ""),
                    "param_size": d.get("parameter_size", ""),
                    "quantization": d.get("quantization_level", ""),
                    "family": d.get("family", ""),
                    "size_gb": round((m.get("size") or 0) / 1024 ** 3, 2),
                })
            return details
        except Exception:
            return []


# =====================================================================
# 模块二：任务规划模块（Agent 核心能力）
#   自动拆解任务：由 ReAct 循环实现“边执行、边思考、边修正方案”；
#   自动判断是否调用工具、是否需要继续迭代。
# =====================================================================
class TaskPlanner:
    """负责系统提示词构建、ReAct 输出解析与格式控制。"""

    SYSTEM_TEMPLATE = """{persona}
你是一个运行在本地、可以调用工具解决任务的智能体。请严格遵循 ReAct 格式分步工作：

【当前日期】{today}（涉及“今天/最近/最新/今年”等时间问题一律以此为准）

【可用工具】
{tools}

【相关历史记忆】（自动召回，仅供参考，可能为空）
{memory}

【输出格式】每次回复只能包含以下两种方式之一：
方式一（需要调用工具时）：
Thought: <你的思考：分析任务、判断是否需要工具、下一步计划>
Action: <工具名，必须是上面可用工具之一>
Action Input: <工具参数>

方式二（信息足够、可以直接回答时）：
Thought: <你的结论性思考>
Final Answer: <面向用户的最终中文回答>

【规则】
1. 每次只输出一组 Thought/Action 或 Thought/Final Answer，不要跳步、不要虚构工具执行结果。
2. 工具的真实结果会以 Observation 提供给你，请基于它继续思考或修正方案。
3. 简单问候、闲聊或无需工具的问题，直接输出 Final Answer。
4. Action Input 若为 JSON 必须是合法 JSON；最终回答使用中文、条理清晰。
5. 时效性问题（新闻、天气、股价、赛事、版本发布、“最近/最新/今天”等）必须先调用联网搜索工具，
   并以搜索结果中【接近当前日期】的信息为准；搜索结果日期明显过旧时，必须如实说明“未找到近期信息”，
   绝不能用旧闻或你训练数据里的过时内容冒充最新情况。
6. 【严禁捏造】如果工具返回“搜索失败/无结果”或观察结果为空：
   - 对时效性问题，必须直接告知“未能检索到相关信息”，绝不允许凭记忆编造新闻、数据、链接或来源；
   - 对非时效性问题，可基于已有知识回答，但必须明确说明“以下内容未经联网核实，仅供参考”。
   任何情况下都不得伪造 URL、日期、数据来源或引用不存在的网页。"""

    def __init__(self, persona: str = "你是一个乐于助人的通用智能体。"):
        self.persona = persona  # 专属提示词入口（供进阶 1 垂直智能体覆写）

    def build_system_prompt(self, tools_desc: str, memory_block: str) -> str:
        return self.SYSTEM_TEMPLATE.format(
            persona=self.persona, tools=tools_desc or "（无可用工具）",
            memory=memory_block or "（暂无）", today=time.strftime("%Y年%m月%d日"),
        )

    # ---------- ReAct 输出解析 ----------
    def parse(self, text: str) -> Dict[str, Any]:
        """解析模型输出 → {'type': 'final'|'action', ...}；小模型不守格式时兜底为最终答案。"""
        clean = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()  # 去除思考标签
        clean = re.sub(r"```[a-zA-Z]*\n?|```", "", clean).strip()                    # 去除代码围栏

        m_final = re.search(r"Final Answer\s*[:：]\s*(.+)\s*$", clean, re.S | re.I)
        m_act = re.search(r"Action\s*[:：]\s*([^\n]+)", clean, re.I)
        m_in = re.search(r"Action Input\s*[:：]\s*(.+)", clean, re.S | re.I)
        m_th = re.search(r"Thought\s*[:：]\s*(.+?)(?=\nAction|\nFinal Answer|\Z)", clean, re.S | re.I)
        thought = m_th.group(1).strip() if m_th else ""

        if m_act:
            tool_name = m_act.group(1).strip().strip("`\"' ")
            # 容错：个别小模型会误写成 Action: Final Answer
            if tool_name.lower().replace("_", " ").replace("：", ":") in ("final answer", "final", "最终答案"):
                answer_text = (m_in.group(1).strip() if m_in else "") or thought or clean
                return {"type": "final", "thought": thought, "answer": answer_text}
            return {
                "type": "action",
                "thought": thought,
                "tool": tool_name,
                "input": (m_in.group(1).strip() if m_in else ""),
            }
        if m_final:
            return {"type": "final", "thought": thought, "answer": m_final.group(1).strip()}
        # 未遵循任何格式：整体视作最终答案（兜底，避免死循环）
        return {"type": "final", "thought": thought, "answer": clean}


# =====================================================================
# 模块三：工具调用模块（行动能力）
#   内置基础工具：计算器、本地文件读取；模型自动判断何时调用、选择哪个工具、解析参数；
#   支持通过 @tool 装饰器无限扩展自定义工具（联网搜索、代码执行、API 调用、硬件控制等）。
# =====================================================================
class Tool:
    def __init__(self, name: str, description: str, func: Callable[[str], str], param_hint: str):
        self.name = name
        self.description = description
        self.func = func
        self.param_hint = param_hint


TOOL_REGISTRY: Dict[str, Tool] = {}


def tool(name: str, description: str, param_hint: str = "字符串参数"):
    """自定义工具注册装饰器（无限扩展入口）。示例：

        @tool("web_search", "联网搜索最新信息", "搜索关键词")
        def web_search(keyword: str) -> str:
            ...
    """
    def decorator(func: Callable[[str], str]) -> Callable[[str], str]:
        TOOL_REGISTRY[name] = Tool(name, description, func, param_hint)
        return func
    return decorator


# ---------- 内置工具 1：安全计算器 ----------
_AST_BIN = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Mod: op.mod, ast.Pow: op.pow, ast.FloorDiv: op.floordiv,
}
_AST_FN = {
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "log": math.log, "log2": math.log2, "log10": math.log10, "exp": math.exp,
    "abs": abs, "round": round, "floor": math.floor, "ceil": math.ceil, "pow": pow,
}
_AST_CONST = {"pi": math.pi, "e": math.e}


def _safe_eval(node) -> float:
    """基于 AST 白名单的安全表达式求值（拒绝任意代码执行）。"""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _AST_BIN:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("幂运算过大")
        return _AST_BIN[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        val = _safe_eval(node.operand)
        return val if isinstance(node.op, ast.UAdd) else -val
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id in _AST_FN and not node.keywords:
        return _AST_FN[node.func.id](*[_safe_eval(a) for a in node.args])
    if isinstance(node, ast.Name) and node.id in _AST_CONST:
        return _AST_CONST[node.id]
    raise ValueError("不支持的表达式")


def _balance_parens(expr: str) -> str:
    """容错：全角括号转半角；自动补齐缺失的右括号、丢弃多余的右括号（小模型常漏写括号）。"""
    expr = expr.replace("（", "(").replace("）", ")")
    out, depth = [], 0
    for ch in expr:
        if ch == "(":
            depth += 1
            out.append(ch)
        elif ch == ")":
            if depth > 0:  # 仅在能匹配时保留，多余的右括号丢弃
                depth -= 1
                out.append(ch)
        else:
            out.append(ch)
    out.append(")" * depth)  # 补齐未闭合的左括号
    return "".join(out)


@tool("calculator", "精确的数学计算器，支持四则运算与 sqrt/sin/cos/log/pi 等函数常量", "数学表达式，如 (3+5)*12/2")
def _calculator(expression: str) -> str:
    expr = expression.strip().strip("`\"' ")
    expr = expr.replace("×", "*").replace("÷", "/").replace("^", "**")
    expr = _balance_parens(expr)
    result = _safe_eval(ast.parse(expr, mode="eval"))
    return f"{expr} = {result}"


# ---------- 内置工具 2：本地文件读取 ----------
@tool("read_file", "读取本地文件内容（文本类文件）", "本地文件的完整路径，如 D:/data/notes.txt")
def _read_file(path: str) -> str:
    p = os.path.abspath(path.strip().strip("`\"' "))
    if not os.path.isfile(p):
        return f"错误：文件不存在：{p}"
    size = os.path.getsize(p)
    raw = open(p, "rb").read(CONFIG["max_file_bytes"])
    text = None
    for enc in ("utf-8", "gbk"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw.decode("utf-8", errors="replace")
    note = "（内容过长已截断）" if size > len(raw) else ""
    return f"文件 {p}（{size} 字节）{note} 内容：\n{text}"


class ToolBox:
    """工具箱：注册 / 描述 / 执行工具。执行永远返回字符串（错误也作为 Observation 返还模型自纠）。

    allowed 为 None 时所有工具可用；传入集合（如 {"calculator"}）则只有白名单内工具可用，
    供网页端按“联网开关”动态启停联网类工具。
    """

    def __init__(self):
        self.tools: Dict[str, Tool] = dict(TOOL_REGISTRY)  # 拷贝，供垂直智能体追加私有工具
        self.allowed: Optional[set] = None

    def register(self, t: Tool) -> None:
        self.tools[t.name] = t

    def _active(self) -> List[Tool]:
        return [t for t in self.tools.values() if self.allowed is None or t.name in self.allowed]

    def descriptions(self) -> str:
        return "\n".join(f"- {t.name}：{t.description}（参数：{t.param_hint}）" for t in self._active())

    def call(self, name: str, action_input: str) -> str:
        t = self.tools.get(name)
        if t is None:
            return f"错误：未知工具 '{name}'。可用工具：{', '.join(x.name for x in self._active()) or '无'}"
        if self.allowed is not None and name not in self.allowed:
            return f"错误：工具 '{name}' 当前未被启用。可用工具：{', '.join(x.name for x in self._active())}"
        try:
            result = str(t.func(action_input))
        except Exception as e:
            return f"工具执行出错：{e}。请修正 Action Input 后重试。"
        if len(result) > CONFIG["max_obs_chars"]:  # 防止超长观察撑爆上下文
            result = result[: CONFIG["max_obs_chars"]] + "\n…（观察结果过长已截断）"
        return result


# =====================================================================
# 模块四：双记忆系统（短期记忆 + 长期向量记忆）
#   短期记忆：当前对话上下文，多轮连续对话不掉上下文；
#   长期记忆：Chroma 本地向量数据库持久化，重启不丢失，提问时自动召回。
# =====================================================================
class ShortTermMemory:
    """短期记忆：环形缓冲保存最近 N 条对话消息。"""

    def __init__(self):
        self._buf: deque = deque(maxlen=CONFIG["short_term_turns"] * 2)

    def add(self, role: str, content: str) -> None:
        self._buf.append({"role": role, "content": content})

    def snapshot(self) -> List[Dict[str, str]]:
        return list(self._buf)

    def clear(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)


def _get_chroma():
    try:
        import chromadb
        return chromadb
    except ImportError:
        sys.exit("缺少依赖 chromadb，请先执行：pip install chromadb")


class LongTermMemory:
    """长期记忆：基于 Chroma 的本地持久化向量库（cosine 相似度）。"""

    def __init__(self, brain: LLMBrain):
        self.brain = brain
        chromadb = _get_chroma()
        os.makedirs(CONFIG["long_term_dir"], exist_ok=True)
        self.client = chromadb.PersistentClient(path=CONFIG["long_term_dir"])
        self.col = self.client.get_or_create_collection(
            CONFIG["memory_collection"], metadata={"hnsw:space": "cosine"}
        )
        self._dim_ready = True

    def add(self, text: str, meta: Optional[dict] = None) -> None:
        if not text.strip():
            return
        try:
            self.col.add(
                ids=[uuid.uuid4().hex],
                documents=[text],
                embeddings=[self.brain.embed(text)],
                metadatas=[meta or {}],
            )
        except Exception as e:
            self._handle_dim_error(e)

    def search(self, query: str, k: int, max_distance: float) -> List[str]:
        """按语义相似度召回历史记忆。"""
        if self.col.count() == 0:
            return []
        try:
            res = self.col.query(
                query_embeddings=[self.brain.embed(query)],
                n_results=min(k, self.col.count()),
                include=["documents", "distances"],
            )
        except Exception as e:
            self._handle_dim_error(e)
            return []
        docs, dists = res["documents"][0], res["distances"][0]
        return [d for d, dist in zip(docs, dists) if dist <= max_distance]

    def count(self) -> int:
        return self.col.count()

    def _handle_dim_error(self, e: Exception) -> None:
        """切换了嵌入模型导致向量维度变化时，自动重建记忆库（旧记忆与新向量不兼容）。"""
        if "dimension" in str(e).lower() and self._dim_ready:
            self._dim_ready = False
            print("提示：检测到向量维度变化（嵌入模型被切换），已重建长期记忆库。")
            self.client.delete_collection(CONFIG["memory_collection"])
            self.col = self.client.get_or_create_collection(
                CONFIG["memory_collection"], metadata={"hnsw:space": "cosine"}
            )
        elif CONFIG["debug"]:
            print(f"  [记忆] 读写异常：{e}")


class MemorySystem:
    """双记忆系统门面：统一管理短期/长期记忆的写入与召回。"""

    def __init__(self, brain: LLMBrain):
        self.brain = brain
        self.short = ShortTermMemory()
        self.long = LongTermMemory(brain)

    def recall(self, query: str) -> List[str]:
        """根据用户输入自动召回相关历史记忆。"""
        return self.long.search(query, CONFIG["recall_top_k"], CONFIG["recall_max_distance"])

    def add_turn(self, user: str, answer: str) -> None:
        self.short.add("user", user)
        self.short.add("assistant", answer)

    def store_long_term(self, user: str, answer: str) -> None:
        """自动记忆：历史问答 + 任务记录持久化，重启程序不丢失。"""
        self.long.add(
            f"用户问：{user}\n助手答：{answer}",
            {"time": time.strftime("%Y-%m-%d %H:%M:%S")},
        )


# =====================================================================
# 模块五：自主执行循环模块（Agent 运行骨架）
#   标准闭环：接收输入 → 召回记忆 → 任务规划 → 工具调用 → 结果观察 →
#             继续推理 → 输出最终答案；自动处理错误、自动重试、判断任务完成。
# =====================================================================
class Agent:
    def __init__(self, brain: LLMBrain, planner: TaskPlanner, memory: MemorySystem,
                 toolbox: Optional[ToolBox] = None, debug: bool = False):
        self.brain = brain
        self.planner = planner
        self.memory = memory
        self.toolbox = toolbox or ToolBox()
        self.debug = debug
        self._first_run_hint_shown = False

    def run(self, user_input: str, event_callback: Optional[Callable[[str, Any], None]] = None,
            allowed_tools: Optional[List[str]] = None) -> str:
        """执行一次完整 ReAct 闭环。

        :param event_callback: 可选回调 fn(event:str, data)，事件类型：
               recall（长期记忆召回）/ thought（思考）/ action（调用工具）/ observation（观察结果），
               供网页端实时可视化思考全过程。
        :param allowed_tools: 本轮允许使用的工具白名单；None 表示工具箱内全部可用。
        """
        self.toolbox.allowed = set(allowed_tools) if allowed_tools is not None else None

        def emit(event: str, data: Any) -> None:
            if event_callback is None:
                return
            try:
                event_callback(event, data)
            except Exception:
                pass  # 回调异常不影响主流程

        # ① 接收输入 + ② 自动召回长期记忆
        recalled = self.memory.recall(user_input)
        if self.debug and recalled:
            print(f"  [记忆召回] 命中 {len(recalled)} 条：")
            for r in recalled:
                print(f"    · {r[:80]}{'…' if len(r) > 80 else ''}")
        emit("recall", recalled)

        # ③ 构建上下文：系统提示（含工具与记忆）+ 短期历史 + 当前任务
        system_prompt = self.planner.build_system_prompt(
            self.toolbox.descriptions(), "\n".join(f"- {r}" for r in recalled)
        )
        messages: List[Dict[str, str]] = (
            [{"role": "system", "content": system_prompt}]
            + self.memory.short.snapshot()
            + [{"role": "user", "content": user_input}]
        )

        # ④⑤⑥ ReAct 自主循环：思考 → 工具 → 观察 → 迭代
        if not self._first_run_hint_shown:
            self._first_run_hint_shown = True
            print("（首次调用需把模型加载进内存，可能较慢，请稍候…）", flush=True)
            emit("hint", "首次调用需把模型加载进内存，可能较慢，请稍候…")
        answer: Optional[str] = None
        for step in range(1, CONFIG["max_react_steps"] + 1):
            reply = self.brain.chat(messages)          # 思考（含自动重试）
            parsed = self.planner.parse(reply)         # 规划/解析
            if parsed["type"] == "final":
                if self.debug:
                    print(f"  [思考·第{step}步] {parsed['thought']}")
                emit("thought", {"step": step, "text": parsed["thought"], "final": True})
                answer = parsed["answer"]
                break
            if self.debug:
                print(f"  [思考·第{step}步] {parsed['thought']}")
                print(f"  [调用工具] {parsed['tool']}（参数：{parsed['input']}）")
            emit("thought", {"step": step, "text": parsed["thought"], "final": False})
            emit("action", {"step": step, "tool": parsed["tool"], "input": parsed["input"]})
            observation = self.toolbox.call(parsed["tool"], parsed["input"])
            if self.debug:
                preview = observation[:300].replace("\n", " ")
                print(f"  [观察] {preview}{'…' if len(observation) > 300 else ''}")
            emit("observation", {"step": step, "tool": parsed["tool"], "text": observation})
            messages.append({"role": "assistant", "content": reply})
            messages.append({
                "role": "user",
                "content": f"Observation: {observation}\n\n请继续：若信息足够请输出 Final Answer，否则继续 Thought/Action。",
            })

        # 达到最大步数仍未完成：强制收敛，避免无限循环
        if answer is None:
            messages.append({"role": "user",
                             "content": "已达最大迭代步数，请立即基于以上全部观察直接输出 Final Answer。"})
            parsed = self.planner.parse(self.brain.chat(messages))
            if parsed["type"] == "final":
                answer = parsed["answer"]
            else:  # 模型仍不配合：汇总已有观察作为答案，保证流程不中断
                observations = [m["content"] for m in messages if m["content"].startswith("Observation:")]
                answer = "\n".join(observations)[-1500:] or "抱歉，本次任务未能得到明确结果，请重试或换一种问法。"

        # ⑦ 输出前更新双记忆
        self.memory.add_turn(user_input, answer)       # 短期：多轮对话上下文
        self.memory.store_long_term(user_input, answer)  # 长期：持久化向量记忆
        return answer


# =====================================================================
# 进阶 1：单任务专精智能体（垂直场景扩展入口）
#   快速改造为：PDF/文档问答、数据分析、笔记整理、论文辅助、代码工程助手等。
#   能力特点：限定领域提示词 + 专用工具链 + 专属知识库。
# =====================================================================
class KnowledgeBase:
    """专属知识库：目录下 .txt/.md 自动分块向量化入库（独立 Chroma 集合）。"""

    def __init__(self, domain: str, kb_dir: str, brain: LLMBrain):
        chromadb = _get_chroma()
        self.brain = brain
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", domain)[:40]
        name = f"kb_{safe}_{hashlib.md5(domain.encode('utf-8')).hexdigest()[:6]}"
        self.client = chromadb.PersistentClient(path=CONFIG["long_term_dir"])
        self.col = self.client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
        self._ingest(kb_dir)

    def _ingest(self, kb_dir: str) -> None:
        if self.col.count() > 0:  # 已建过库则跳过（重建请删除 ./agent_memory 下对应集合）
            return
        files = [os.path.join(kb_dir, f) for f in os.listdir(kb_dir)
                 if f.lower().endswith((".txt", ".md"))] if os.path.isdir(kb_dir) else []
        if not files:
            print(f"提示：知识库目录 {kb_dir} 中没有 .txt/.md 文件。")
            return
        for fp in files:
            text = open(fp, "rb").read().decode("utf-8", errors="replace")
            chunks = [text[i:i + 500] for i in range(0, len(text), 450)]  # 500 字块、50 字重叠
            if chunks:
                self.col.add(
                    ids=[uuid.uuid4().hex for _ in chunks],
                    documents=chunks,
                    embeddings=[self.brain.embed(c) for c in chunks],
                    metadatas=[{"source": os.path.basename(fp)}] * len(chunks),
                )
        print(f"知识库已建立：{kb_dir}（{len(files)} 个文件，共 {self.col.count()} 个知识块）")

    def search(self, query: str) -> str:
        if self.col.count() == 0:
            return "知识库为空。"
        res = self.col.query(query_embeddings=[self.brain.embed(query)],
                             n_results=min(3, self.col.count()),
                             include=["documents", "metadatas"])
        docs = res["documents"][0]
        srcs = [m.get("source", "?") for m in res["metadatas"][0]]
        return "\n---\n".join(f"[来源:{s}] {d}" for s, d in zip(srcs, docs)) or "知识库中未找到相关内容。"


def create_vertical_agent(domain: str, extra_tools: Optional[Iterable[Tool]] = None,
                          system_prompt: Optional[str] = None,
                          kb_dir: Optional[str] = None,
                          brain: Optional[LLMBrain] = None) -> Agent:
    """进阶 1 入口：创建领域专精智能体。

    示例：
        doc_agent = create_vertical_agent("文档问答", kb_dir="./docs")
        print(doc_agent.run("合同里关于违约金的条款是怎么约定的？"))
    """
    brain = brain or LLMBrain(CONFIG["ollama_host"], CONFIG["model"],
                              CONFIG["temperature"], CONFIG["num_ctx"])
    toolbox = ToolBox()
    for t in (extra_tools or []):
        toolbox.register(t)
    if kb_dir:
        kb = KnowledgeBase(domain, kb_dir, brain)
        toolbox.register(Tool("search_knowledge", f"在{domain}专属知识库中检索资料", kb.search, "检索问题"))
    planner = TaskPlanner(persona=system_prompt or f"你是{domain}领域的专业智能体，回答务必专业、准确、聚焦该领域。")
    return Agent(brain, planner, MemorySystem(brain), toolbox, CONFIG["debug"])


# =====================================================================
# 进阶 2：多智能体协作系统（复杂任务扩展入口）
#   支持 规划 Agent / 执行 Agent / 搜索 Agent / 评审 Agent 分工协作；
#   二次开发建议：在 run() 中加入循环校验、动态路由、并行执行等。
# =====================================================================
class MultiAgentSystem:
    """多智能体流水线骨架：按注册顺序接力完成任务，上一棒的输出作为下一棒的输入。"""

    def __init__(self):
        self.agents: Dict[str, Agent] = {}

    def register(self, name: str, agent: Agent) -> None:
        self.agents[name] = agent

    def run(self, task: str) -> Dict[str, str]:
        trace: Dict[str, str] = {}
        context = task
        prev = "用户"
        for name, agent in self.agents.items():
            prompt = context if not trace else (
                f"【总体任务】{task}\n【上一环节（{prev}）结果】\n{context}\n请基于以上内容完成你负责的部分。"
            )
            trace[name] = agent.run(prompt)
            prev, context = name, trace[name]
        return trace

    # TODO 扩展示例：评审 Agent 对结果打分，不达标则回传执行 Agent 重做（互相校验、多轮协作）。


# =====================================================================
# 进阶 3：具身智能体（机器人硬件对接扩展入口）
#   可对接 ROS/ROS2、小车底盘、机械臂、智能家居设备，实现：
#   语言指令 → Agent 思考 → 硬件动作执行。
# =====================================================================
class HardwareInterface:
    """硬件控制骨架。二次开发示例：

        class MyRobot(HardwareInterface):
            def connect(self):
                # TODO: 初始化 ROS2 节点 / 串口 / MQTT 等
                return True

        robot = MyRobot()
        robot.register_action("前进", lambda meters: chassis.move(meters))
        robot.register_action("抓取", lambda obj: arm.grip(obj))
        # 注册为 Agent 工具后即可语音/文字操控：
        toolbox.register(Tool("robot_control", "控制机器人执行动作", 'JSON，如 {"action": "前进", "arg": 1.5}'))
        toolbox.tools["robot_control"].func = robot.execute
    """

    def __init__(self):
        self._actions: Dict[str, Callable] = {}

    def register_action(self, name: str, func: Callable) -> None:
        self._actions[name] = func

    def connect(self) -> bool:
        # TODO: 建立真实硬件连接（ROS2 节点、串口、蓝牙、智能家居 API 等）
        return True

    def execute(self, command: str) -> str:
        """执行硬件动作命令（可作为 Agent 工具函数直接使用）。"""
        try:
            cmd = json.loads(command) if command.strip().startswith("{") else {"action": command.strip()}
        except json.JSONDecodeError:
            return f"错误：无法解析硬件指令：{command}"
        name, arg = cmd.get("action"), cmd.get("arg")
        fn = self._actions.get(name)
        if fn is None:
            return f"错误：未注册动作 '{name}'。已注册：{', '.join(self._actions) or '无'}"
        try:
            return f"已执行动作 [{name}]，结果：{fn(arg)}"
        except Exception as e:
            return f"硬件执行出错：{e}"


# =====================================================================
# 进阶 4：工程化部署能力（上线生产扩展入口）
#   FastAPI 接口封装，供网页端 / 小程序调用；可进一步扩展日志、任务队列、
#   权限管理、批量任务等。
# =====================================================================
def create_fastapi_app(agent: Agent):
    """进阶 4 入口：把 Agent 封装为 HTTP 服务。

    用法：
        pip install fastapi uvicorn
        python -c "from local_agent import *; app = build_default_agent(); \
                   import uvicorn; uvicorn.run(create_fastapi_app(app), host='0.0.0.0', port=8000)"
    之后即可 POST /chat {"message": "..."} 调用。
    """
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
    except ImportError:
        raise RuntimeError("请先安装：pip install fastapi uvicorn")

    app = FastAPI(title="Local Agent API", description="本地智能体 HTTP 服务")

    class ChatRequest(BaseModel):
        message: str

    @app.get("/health")
    def health():
        return {"status": "ok", "model": agent.brain.model}

    @app.post("/chat")
    def chat(req: ChatRequest):
        try:
            return {"answer": agent.run(req.message)}
        except Exception as e:
            return {"error": str(e)}

    # TODO 扩展示例：/tools 工具列表、/memory 记忆管理、API-Key 鉴权、Celery 任务队列等。
    return app


# =====================================================================
# 命令行主程序：多轮对话 REPL
# =====================================================================
def _pick_model(brain: LLMBrain, requested: str, installed: List[str]) -> str:
    """优先使用请求的模型；不存在则自动挑选（偏好 qwen / llama）。"""
    for m in installed:
        if m == requested or m.split(":")[0] == requested.split(":")[0]:
            return m
    for key in ("qwen", "llama"):
        for m in installed:
            if key in m.lower():
                return m
    return installed[0]


def _print_banner() -> None:
    print("=" * 64)
    print("  本地可运行完整智能体（ReAct + 双记忆 + 工具调用）")
    print("  输入 /help 查看命令，/exit 退出")
    print("=" * 64)


def main() -> None:
    parser = argparse.ArgumentParser(description="本地可运行完整智能体（Ollama + Chroma）")
    parser.add_argument("--model", default=CONFIG["model"], help="本地模型名，如 qwen2.5:3b")
    parser.add_argument("--host", default=CONFIG["ollama_host"], help="Ollama 服务地址")
    parser.add_argument("--temperature", type=float, default=CONFIG["temperature"], help="温度 0~1")
    parser.add_argument("--debug", action="store_true", help="调试模式：打印思考全过程")
    args = parser.parse_args()
    CONFIG["debug"] = args.debug or CONFIG["debug"]

    brain = LLMBrain(args.host, args.model, args.temperature, CONFIG["num_ctx"],
                     CONFIG["num_predict"])

    # 启动自检：Ollama 与本地模型
    installed = brain.list_models()
    if not installed:
        print("未检测到 Ollama 服务或本地模型。请按以下步骤准备环境后重试：")
        print("  1. 安装 Ollama：https://ollama.com/download")
        print("  2. 拉取模型：  ollama pull qwen2.5:3b   （CPU 可跑，无需 GPU）")
        print("  3. 启动服务：  ollama serve            （安装版一般已自动运行）")
        sys.exit(1)
    brain.model = _pick_model(brain, args.model, installed)
    if brain.model != args.model:
        print(f"提示：模型 {args.model} 未安装，已自动改用本地模型 {brain.model}。")

    # 组装五大模块
    print("正在初始化记忆系统…")
    memory = MemorySystem(brain)
    agent = Agent(brain, TaskPlanner(), memory, ToolBox(), CONFIG["debug"])

    _print_banner()
    print(f"  模型：{brain.model}   温度：{brain.temperature}   上下文：{brain.num_ctx}")
    print(f"  长期记忆：{memory.long.count()} 条（目录 {CONFIG['long_term_dir']}）")
    print(f"  可用工具：{', '.join(agent.toolbox.tools) or '无'}")

    # 多轮对话 REPL
    while True:
        try:
            query = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not query:
            continue
        low = query.lower()
        if low in ("/exit", "/quit", "exit", "quit"):
            print("再见！")
            break
        if low in ("/help", "help"):
            print("""命令：
  /tools            查看可用工具
  /memory           查看记忆条数（短期/长期）
  /clear            清空短期记忆（长期记忆保留）
  /model 模型名      切换本地模型
  /temp 数值         调整温度（0~1）
  /debug            开关调试模式（打印思考全过程）
  /exit             退出""")
            continue
        if low == "/tools":
            print(agent.toolbox.descriptions() or "（无工具）")
            continue
        if low == "/memory":
            print(f"短期记忆 {len(memory.short)} 条消息；长期记忆 {memory.long.count()} 条。")
            continue
        if low == "/clear":
            memory.short.clear()
            print("短期记忆已清空。")
            continue
        if low.startswith("/model"):
            parts = query.split(maxsplit=1)
            if len(parts) == 2 and parts[1].strip() in brain.list_models():
                brain.model = parts[1].strip()
                print(f"模型已切换为 {brain.model}。")
            else:
                print(f"本地可用模型：{', '.join(brain.list_models())}")
            continue
        if low.startswith("/temp"):
            parts = query.split(maxsplit=1)
            try:
                brain.temperature = max(0.0, min(1.0, float(parts[1])))
                print(f"温度已调整为 {brain.temperature}。")
            except (IndexError, ValueError):
                print("用法：/temp 0.7")
            continue
        if low == "/debug":
            agent.debug = CONFIG["debug"] = not CONFIG["debug"]
            print(f"调试模式：{'开' if agent.debug else '关'}")
            continue

        try:
            print("\n智能体：", flush=True)
            print(agent.run(query))
        except RuntimeError as e:  # LLM 调用失败等已包装的错误信息
            print(f"[错误] {e}")


if __name__ == "__main__":
    main()
