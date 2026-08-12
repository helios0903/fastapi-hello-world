import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

client = OpenAI(
    api_key=os.environ.get("SUPER_MIND_API_KEY"),
    base_url="https://space.ai-builders.com/backend/v1",
)


class ChatRequest(BaseModel):
    user_message: str


@app.get("/", response_class=HTMLResponse)
def home():
    return CHAT_PAGE


@app.get("/hello/{name}")
def hello(name: str):
    return {"message": f"Hello, World {name}"}


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        completion = client.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": request.user_message}],
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Upstream API error: {exc}")

    return {"response": completion.choices[0].message.content}


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
    .app {
      width: 100%;
      max-width: 720px;
      height: 100dvh;
      display: flex;
      flex-direction: column;
      background: var(--panel);
      border-left: 1px solid var(--border);
      border-right: 1px solid var(--border);
    }
    header {
      padding: 14px 16px;
      border-bottom: 1px solid var(--border);
      font-weight: 600;
      font-size: 17px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    header .dot { width: 9px; height: 9px; border-radius: 50%; background: #22c55e; }
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
    .typing { color: var(--muted); font-style: italic; }
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
    button {
      border: none;
      background: var(--user);
      color: var(--user-text);
      font-size: 15px;
      font-weight: 600;
      padding: 0 18px;
      border-radius: 12px;
      cursor: pointer;
    }
    button:disabled { opacity: 0.5; cursor: default; }
  </style>
</head>
<body>
  <div class="app">
    <header><span class="dot"></span> My AI Chat</header>
    <div id="messages">
      <div class="hint">给 AI 发条消息开始聊天吧 👋</div>
    </div>
    <form id="form">
      <textarea id="input" rows="1" placeholder="输入消息…（Enter 发送，Shift+Enter 换行）"></textarea>
      <button id="send" type="submit">发送</button>
    </form>
  </div>

  <script>
    const messages = document.getElementById("messages");
    const form = document.getElementById("form");
    const input = document.getElementById("input");
    const sendBtn = document.getElementById("send");
    let firstMessage = true;

    function addBubble(text, who, extraClass) {
      if (firstMessage) { messages.innerHTML = ""; firstMessage = false; }
      const row = document.createElement("div");
      row.className = "row " + who;
      const bubble = document.createElement("div");
      bubble.className = "bubble" + (extraClass ? " " + extraClass : "");
      bubble.textContent = text;
      row.appendChild(bubble);
      messages.appendChild(row);
      messages.scrollTop = messages.scrollHeight;
      return bubble;
    }

    function autogrow() {
      input.style.height = "auto";
      input.style.height = Math.min(input.scrollHeight, 140) + "px";
    }
    input.addEventListener("input", autogrow);

    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        form.requestSubmit();
      }
    });

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const text = input.value.trim();
      if (!text) return;

      addBubble(text, "user");
      input.value = "";
      autogrow();
      sendBtn.disabled = true;
      const thinking = addBubble("思考中…", "bot", "typing");

      try {
        const res = await fetch("/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ user_message: text }),
        });
        const data = await res.json();
        if (!res.ok) {
          thinking.textContent = "出错了：" + (data.detail || res.status);
          thinking.classList.remove("typing");
        } else {
          thinking.textContent = data.response;
          thinking.classList.remove("typing");
        }
      } catch (err) {
        thinking.textContent = "网络错误：" + err.message;
        thinking.classList.remove("typing");
      } finally {
        sendBtn.disabled = false;
        input.focus();
        messages.scrollTop = messages.scrollHeight;
      }
    });
  </script>
</body>
</html>
"""
