import json
import os
from uuid import uuid4

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

API_KEY = os.environ.get("SUPER_MIND_API_KEY")
BASE_URL = "https://space.ai-builders.com/backend/v1"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


class ChatRequest(BaseModel):
    user_message: str
    conversation_id: str | None = None


# ---------------------------------------------------------------------------
# Agentic tools
#
# A "tool" has two halves:
#   1. web_search()  -> the real Python function that does the work.
#   2. WEB_SEARCH_TOOL -> a JSON schema describing the tool to the LLM, so the
#      LLM knows the tool exists, what it does, and what arguments it takes.
# The LLM never runs the function itself; it only decides to *call* it and
# emits a structured tool call. Our code is what actually executes it.
# ---------------------------------------------------------------------------


def web_search(query: str):
    """Call the internal search API and return its JSON results."""
    response = httpx.post(
        f"{BASE_URL}/search/",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={"keywords": [query], "max_results": 3},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


# JSON schema the LLM reads to understand the tool (OpenAI function-calling format).
WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current, real-time, or factual information that "
            "may not be in the model's training data — recent events, news, "
            "weather, sports results, prices, etc."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query, e.g. 'who won the Super Bowl 2024'",
                }
            },
            "required": ["query"],
        },
    },
}

def read_page(url: str):
    """Fetch a web page and return its main text with tags/scripts/styles stripped."""
    response = httpx.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; AgentBot/1.0)"},
        timeout=30,
        follow_redirects=True,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Collapse whitespace and drop blank lines so the LLM gets clean text.
    lines = [line.strip() for line in soup.get_text(separator="\n").splitlines()]
    text = "\n".join(line for line in lines if line)

    # Cap the length so one big page can't blow up the context window.
    max_chars = 6000
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]"

    return {"url": url, "text": text}


READ_PAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_page",
        "description": (
            "Fetch a web page by its URL and return its main text content. Use "
            "this AFTER web_search to read the full content of a specific result "
            "— an article, a changelog, or a documentation page."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL to read, e.g. 'https://docs.python.org/3/whatsnew/3.13.html'",
                }
            },
            "required": ["url"],
        },
    },
}

# The list we hand the LLM, and a registry mapping each tool name to the real
# Python function so the loop can look it up and actually run it. The loop is
# generic, so adding a tool is just: add its schema here + its function below.
TOOLS = [WEB_SEARCH_TOOL, READ_PAGE_TOOL]
AVAILABLE_TOOLS = {"web_search": web_search, "read_page": read_page}
# Higher now: a "search -> read that page -> answer" flow spends several turns.
MAX_TURNS = 5

# Guardrails for how the agent should treat search results:
#  1. cite the sources it actually relied on
#  2. be honest about conflicts / weak evidence instead of fabricating
SYSTEM_PROMPT = (
    "You are a helpful assistant with two tools: web_search (to find pages) and "
    "read_page (to read the full text of a specific URL). A common pattern is to "
    "web_search first, then read_page the most relevant result for details.\n"
    "\n"
    "STYLE — structured but concise:\n"
    "- Lead with a direct 2-3 sentence conclusion before any detail.\n"
    "- Short headings or bullet lists are fine, but keep them tight (about 3-5 "
    "items max). Never dump an exhaustive 'laundry list'.\n"
    "- If the request is broad or ambiguous (e.g. 'the latest LLMs'), give a brief "
    "answer first, then ask ONE clarifying question about what the user actually "
    "cares about, instead of listing everything.\n"
    "\n"
    "SEARCH — do not over-search: one or two focused searches are usually enough; "
    "search again only if the first results clearly miss the question.\n"
    "\n"
    "SOURCING — if you used search results, end with a short '来源 / Sources' "
    "section listing only the URLs you actually relied on. If you did not use the "
    "tool, do not invent sources.\n"
    "\n"
    "HONESTY — if sources disagree or the evidence is weak, say so briefly instead "
    "of guessing. Never fabricate facts or a source.\n"
    "\n"
    "Always answer in the same language the user used."
)

# The agent's "memory": conversation_id -> full message history.
# NOTE: this lives in the server's RAM, so it resets when Render restarts or
# spins down, and it grows over time. A real app would use a database and trim
# or summarize old turns — that trimming is itself context engineering.
CONVERSATIONS: dict[str, list] = {}


@app.get("/", response_class=HTMLResponse)
def home():
    return CHAT_PAGE


@app.get("/hello/{name}")
def hello(name: str):
    return {"message": f"Hello, World {name}"}


def log(line: str):
    """Print to the console (Render Logs) and flush so it shows up immediately."""
    print(line, flush=True)


def dump_context_trace(messages: list):
    """Print the ENTIRE message history the model saw — the agent's 'mind'.

    Reading this context trace is how you debug an agent: you see exactly what
    the model looked at (every role, the raw tool_calls object, the tool results
    with their tool_call_id, and the final answer) before it decided. Debugging
    an agent = reading its context trace, not guessing.
    """
    log("[Context Trace] full message history the model saw:")
    log(json.dumps(messages, ensure_ascii=False, indent=2))


@app.post("/chat")
def chat(request: ChatRequest):
    """The full agentic loop: reason -> (maybe) call a tool -> feed the result
    back -> reason again, up to MAX_TURNS times, then answer.
    """
    convo_id = request.conversation_id or uuid4().hex
    # Load this conversation's past messages (the "memory"), or start fresh.
    # list(...) copies it so a mid-loop error can't corrupt the stored history.
    messages = list(CONVERSATIONS.get(convo_id) or [{"role": "system", "content": SYSTEM_PROMPT}])
    messages.append({"role": "user", "content": request.user_message})
    log(
        f"[User] Question: '{request.user_message}' "
        f"(convo {convo_id[:8]}, {len(messages)} msgs already in context)"
    )

    try:
        for turn in range(MAX_TURNS):
            completion = client.chat.completions.create(
                model="gpt-5",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            message = completion.choices[0].message

            # No tool call -> the LLM is done reasoning and gave a final answer.
            if not message.tool_calls:
                messages.append({"role": "assistant", "content": message.content})
                dump_context_trace(messages)  # the full chain of thought
                log(f"[Agent] Final Answer: '{message.content}'")
                CONVERSATIONS[convo_id] = messages  # remember for next turn
                return {"response": message.content, "conversation_id": convo_id}

            # Record the assistant's decision (its tool calls) in the transcript.
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments,
                            },
                        }
                        for call in message.tool_calls
                    ],
                }
            )

            # Execute every tool the LLM asked for, and feed each result back.
            for call in message.tool_calls:
                name = call.function.name
                args = json.loads(call.function.arguments or "{}")
                log(f"[Agent] Decided to call tool: '{name}' (turn {turn + 1}) args={args}")

                func = AVAILABLE_TOOLS.get(name)
                if func is None:
                    result = {"error": f"Unknown tool: {name}"}
                else:
                    try:
                        result = func(**args)
                    except Exception as tool_error:
                        result = {"error": str(tool_error)}

                result_str = json.dumps(result, ensure_ascii=False)
                log(f"[System] Tool Output: '{result_str[:800]}'")

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result_str,
                    }
                )

        # Ran out of turns while still calling tools: force a final text answer.
        final = client.chat.completions.create(model="gpt-5", messages=messages)
        answer = final.choices[0].message.content
        messages.append({"role": "assistant", "content": answer})
        dump_context_trace(messages)  # the full chain of thought
        log(f"[Agent] Final Answer (after {MAX_TURNS} turns): '{answer}'")
        CONVERSATIONS[convo_id] = messages  # remember for next turn
        return {"response": answer, "conversation_id": convo_id}

    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Upstream API error: {exc}")


@app.post("/agent")
def agent(request: ChatRequest):
    """Step 1 of the agentic loop: let the LLM *decide* whether to use a tool.

    We hand the model the tool schema and let it choose (tool_choice="auto").
    We do NOT run the tool yet — we just surface the decision so we can verify
    the LLM emits a valid tool call for questions that need fresh information.
    """
    try:
        completion = client.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": request.user_message}],
            tools=TOOLS,
            tool_choice="auto",
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Upstream API error: {exc}")

    message = completion.choices[0].message

    # The LLM decided it needs a tool: it returns structured tool_calls.
    if message.tool_calls:
        return {
            "decision": "tool_call",
            "tool_calls": [
                {
                    "id": call.id,
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                }
                for call in message.tool_calls
            ],
        }

    # The LLM answered directly from its own knowledge.
    return {"decision": "answer", "response": message.content}


CHAT_PAGE = """<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1" />
  <title>My AI Chat</title>
  <style>
    :root {
      --bg: #f4f5f7;
      --panel: #ffffff;
      --text: #1a1a1a;
      --muted: #8a8f98;
      --user: #2563eb;
      --user-text: #ffffff;
      --bot: #eceef1;
      --bot-text: #1a1a1a;
      --border: #e2e4e8;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #0f1115;
        --panel: #171a21;
        --text: #e8eaed;
        --muted: #9aa0aa;
        --user: #3b82f6;
        --user-text: #ffffff;
        --bot: #242832;
        --bot-text: #e8eaed;
        --border: #2a2f3a;
      }
    }
    * { box-sizing: border-box; }
    html, body { height: 100%; margin: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "PingFang SC", "Microsoft YaHei", sans-serif;
      display: flex;
      justify-content: center;
    }
    .shell {
      width: 100%;
      max-width: 960px;
      height: 100dvh;
      display: flex;
      position: relative;
      overflow: hidden;
      background: var(--panel);
      border-left: 1px solid var(--border);
      border-right: 1px solid var(--border);
    }
    /* ---- sidebar: conversation history (lives in this browser) ---- */
    .sidebar {
      width: 250px;
      flex-shrink: 0;
      display: flex;
      flex-direction: column;
      background: var(--bg);
      border-right: 1px solid var(--border);
    }
    .side-head { padding: 14px 14px 10px; display: flex; flex-direction: column; gap: 10px; }
    .side-head h2 { margin: 0; font-size: 12px; letter-spacing: 0.05em; text-transform: uppercase; color: var(--muted); }
    #new-chat {
      border: none; background: var(--user); color: var(--user-text);
      font-size: 14px; font-weight: 600; padding: 9px 12px; border-radius: 10px; cursor: pointer;
    }
    #convo-list { flex: 1; overflow-y: auto; padding: 6px 8px 12px; display: flex; flex-direction: column; gap: 4px; }
    .convo {
      display: flex; align-items: center; gap: 6px; padding: 9px 10px; border-radius: 9px;
      cursor: pointer; font-size: 14px; color: var(--text); border: 1px solid transparent;
    }
    .convo:hover { background: var(--bot); }
    .convo.active { background: var(--bot); border-color: var(--border); }
    .convo .name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .convo .del { border: none; background: transparent; color: var(--muted); cursor: pointer; font-size: 14px; padding: 0 2px; opacity: 0; }
    .convo:hover .del, .convo.active .del { opacity: 1; }
    .side-empty { color: var(--muted); font-size: 13px; padding: 10px; text-align: center; }
    /* ---- main chat column ---- */
    .main { flex: 1; min-width: 0; display: flex; flex-direction: column; }
    header {
      padding: 12px 14px;
      border-bottom: 1px solid var(--border);
      font-weight: 600;
      font-size: 17px;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    header .dot { width: 9px; height: 9px; border-radius: 50%; background: #22c55e; flex-shrink: 0; }
    header .title { flex: 1; }
    #menu { display: none; border: none; background: transparent; color: var(--text); font-size: 20px; cursor: pointer; padding: 0 4px; }
    #messages {
      flex: 1;
      overflow-y: auto;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .row { display: flex; }
    .row.user { justify-content: flex-end; }
    .bubble {
      max-width: 82%;
      padding: 10px 14px;
      border-radius: 16px;
      line-height: 1.5;
      white-space: pre-wrap;
      word-wrap: break-word;
      font-size: 15px;
    }
    .user .bubble { background: var(--user); color: var(--user-text); border-bottom-right-radius: 4px; }
    .bot .bubble { background: var(--bot); color: var(--bot-text); border-bottom-left-radius: 4px; }
    .hint { color: var(--muted); text-align: center; font-size: 14px; margin: auto 0; }
    /* animated thinking dots */
    .dots { display: inline-flex; gap: 4px; align-items: center; }
    .dots span { width: 7px; height: 7px; border-radius: 50%; background: var(--muted); animation: blink 1.2s infinite ease-in-out both; }
    .dots span:nth-child(2) { animation-delay: 0.2s; }
    .dots span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes blink { 0%, 80%, 100% { opacity: 0.25; transform: translateY(0); } 40% { opacity: 1; transform: translateY(-3px); } }
    @media (prefers-reduced-motion: reduce) { .dots span { animation: none; opacity: 0.6; } }
    form {
      display: flex;
      gap: 8px;
      padding: 12px;
      border-top: 1px solid var(--border);
    }
    #input {
      flex: 1;
      resize: none;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 11px 13px;
      font-size: 15px;
      background: var(--bg);
      color: var(--text);
      max-height: 140px;
      font-family: inherit;
    }
    #input:focus { outline: 2px solid var(--user); border-color: transparent; }
    #send {
      border: none;
      background: var(--user);
      color: var(--user-text);
      font-size: 15px;
      font-weight: 600;
      padding: 0 18px;
      border-radius: 12px;
      cursor: pointer;
    }
    #send:disabled { opacity: 0.5; cursor: default; }
    .backdrop { display: none; position: absolute; inset: 0; background: rgba(0,0,0,0.4); z-index: 5; }
    /* mobile: sidebar slides over the chat */
    @media (max-width: 640px) {
      .sidebar {
        position: absolute; top: 0; bottom: 0; left: 0; z-index: 10;
        transform: translateX(-100%); transition: transform 0.22s ease;
        box-shadow: 0 0 40px rgba(0,0,0,0.25);
      }
      .shell.open .sidebar { transform: translateX(0); }
      .shell.open .backdrop { display: block; }
      #menu { display: block; }
    }
  </style>
</head>
<body>
  <div class="shell" id="shell">
    <aside class="sidebar">
      <div class="side-head">
        <h2>对话历史</h2>
        <button id="new-chat" type="button">＋ 新对话</button>
      </div>
      <div id="convo-list"></div>
    </aside>
    <div class="backdrop" id="backdrop"></div>
    <div class="main">
      <header>
        <button id="menu" type="button" aria-label="打开历史">☰</button>
        <span class="dot"></span>
        <span class="title">My AI Chat</span>
      </header>
      <div id="messages">
        <div class="hint">给 AI 发条消息开始聊天吧 👋</div>
      </div>
      <form id="form">
        <textarea id="input" rows="1" placeholder="输入消息…（Enter 发送，Shift+Enter 换行）"></textarea>
        <button id="send" type="submit">发送</button>
      </form>
    </div>
  </div>

  <script>
    const shell = document.getElementById("shell");
    const messages = document.getElementById("messages");
    const form = document.getElementById("form");
    const input = document.getElementById("input");
    const sendBtn = document.getElementById("send");
    const newChatBtn = document.getElementById("new-chat");
    const convoList = document.getElementById("convo-list");
    const menuBtn = document.getElementById("menu");
    const backdrop = document.getElementById("backdrop");

    const STORE_KEY = "chats_v1";
    const HINT = '<div class="hint">给 AI 发条消息开始聊天吧 👋</div>';

    function newId() {
      return (crypto.randomUUID && crypto.randomUUID()) || String(Math.random()).slice(2);
    }

    // ---- The conversation list lives ONLY in this browser (localStorage).
    //      Switch device/browser and it's gone — no login, no server storage. ----
    function loadChats() {
      try { return JSON.parse(localStorage.getItem(STORE_KEY)) || []; }
      catch (e) { return []; }
    }
    function saveChats() { localStorage.setItem(STORE_KEY, JSON.stringify(chats)); }

    let chats = loadChats();
    let currentId = null;
    const chatById = (id) => chats.find((c) => c.id === id);

    function renderSidebar() {
      convoList.innerHTML = "";
      if (!chats.length) { convoList.innerHTML = '<div class="side-empty">还没有对话</div>'; return; }
      chats.forEach((c) => {
        const item = document.createElement("div");
        item.className = "convo" + (c.id === currentId ? " active" : "");
        const name = document.createElement("span");
        name.className = "name";
        name.textContent = c.title || "新对话";
        const del = document.createElement("button");
        del.className = "del"; del.type = "button"; del.textContent = "🗑"; del.title = "删除";
        del.addEventListener("click", (e) => { e.stopPropagation(); deleteChat(c.id); });
        item.appendChild(name); item.appendChild(del);
        item.addEventListener("click", () => openChat(c.id));
        convoList.appendChild(item);
      });
    }

    function paintBubble(text, who, isDots) {
      const hint = messages.querySelector(".hint");
      if (hint) hint.remove();
      const row = document.createElement("div");
      row.className = "row " + who;
      const bubble = document.createElement("div");
      bubble.className = "bubble";
      if (isDots) bubble.innerHTML = '<span class="dots"><span></span><span></span><span></span></span>';
      else bubble.textContent = text;
      row.appendChild(bubble);
      messages.appendChild(row);
      messages.scrollTop = messages.scrollHeight;
      return bubble;
    }

    function renderMessages() {
      const chat = chatById(currentId);
      messages.innerHTML = "";
      if (!chat || !chat.messages.length) { messages.innerHTML = HINT; return; }
      chat.messages.forEach((m) => paintBubble(m.text, m.who));
      messages.scrollTop = messages.scrollHeight;
    }

    function startNewChat() {
      currentId = newId();
      chats.unshift({ id: currentId, title: "", messages: [] });
      saveChats(); renderSidebar(); renderMessages(); closeSidebar(); input.focus();
    }
    function openChat(id) { currentId = id; renderSidebar(); renderMessages(); closeSidebar(); }
    function deleteChat(id) {
      chats = chats.filter((c) => c.id !== id);
      saveChats();
      if (currentId === id) {
        if (chats.length) currentId = chats[0].id;
        else { startNewChat(); return; }
      }
      renderSidebar(); renderMessages();
    }

    // sidebar toggle (mobile)
    function openSidebar() { shell.classList.add("open"); }
    function closeSidebar() { shell.classList.remove("open"); }
    menuBtn.addEventListener("click", openSidebar);
    backdrop.addEventListener("click", closeSidebar);
    newChatBtn.addEventListener("click", startNewChat);

    function autogrow() {
      input.style.height = "auto";
      input.style.height = Math.min(input.scrollHeight, 140) + "px";
    }
    input.addEventListener("input", autogrow);
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
    });

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const text = input.value.trim();
      if (!text) return;
      const chat = chatById(currentId);
      if (!chat) return;

      paintBubble(text, "user");
      chat.messages.push({ who: "user", text: text });
      if (!chat.title) { chat.title = text.slice(0, 30); renderSidebar(); }
      saveChats();

      input.value = ""; autogrow(); sendBtn.disabled = true;
      const thinking = paintBubble("", "bot", true);  // animated dots

      try {
        const res = await fetch("/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          // same id server-side = the agent keeps this conversation's memory
          body: JSON.stringify({ user_message: text, conversation_id: currentId }),
        });
        const data = await res.json();
        const reply = res.ok ? data.response : ("出错了：" + (data.detail || res.status));
        thinking.textContent = reply;
        if (res.ok) { chat.messages.push({ who: "bot", text: reply }); saveChats(); }
      } catch (err) {
        thinking.textContent = "网络错误：" + err.message;
      } finally {
        sendBtn.disabled = false; input.focus();
        messages.scrollTop = messages.scrollHeight;
      }
    });

    // boot: reopen the most recent chat, or start a fresh one
    if (chats.length) { currentId = chats[0].id; renderSidebar(); renderMessages(); }
    else { startNewChat(); }
  </script>
</body>
</html>
"""
