import os
import json
from datetime import datetime
from openai import OpenAI

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
STATE_FILE = os.path.join(BASE_DIR, "state.json")

API_KEY = os.environ.get("LLM_API_KEY", "").strip()
BASE_URL = os.environ.get("LLM_BASE_URL", "").strip() or "https://open.bigmodel.cn/api/paas/v4"
MODEL = os.environ.get("LLM_MODEL", "").strip() or "glm-5.3-flash"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def calc(expression):
    # 计算表达式的值
    return eval(expression)

def read_file(path):
    if not os.path.exists(path):
        return f"文件 {path} 不存在"
    with open(path, "r", encoding="utf-8") as f:
        return f.read(3000)

def write_file(path, content):
    full =os.path.abspath(os.path.join(OUTPUT_DIR, path))
    if not full.startswith(OUTPUT_DIR + os.sep):
        return f"不允许写到 {path}，只能写在output目录里"
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)
    return f"已写入 {len(content)} 个字符到 {full}"

def move_file(src, dst):
    if not os.path.exists(src):
        return f"源文件{src}不存在"
    dst_dir = os.path.dirname(dst)
    if dst_dir:
        os.makedirs(dst_dir, exist_ok=True)
    if os.path.exists(dst):
        return f"目标{dst}已存在，没有移动，避免覆盖"
    os.rename(src, dst)
    return f"已移动{os.path.basename(src)}到{dst}"

def list_files(dir_path, start=0):
    if not os.path.exists(dir_path):
        return f"目录 {dir_path} 不存在"
    if not os.path.isdir(dir_path):
        return f"{dir_path} 不是一个目录"
    files = os.listdir(dir_path)
    files.sort()
    lines = []
    for n in files[start:start + 10]:
        p = os.path.join(dir_path, n)
        tag = "[目录]" if os.path.isdir(p) else "[文件]"
        lines.append(f"{tag} {n}")
    if len(files) > start + 10:
        extra = len(files) - start - 10
        lines.append(f"... 还有 {extra} 个未显示，可以用start={start + 10}继续")
    return "\n".join(lines)

IMPL = {
    "calc": calc,
    "read_file": read_file,
    "list_files": list_files,
    "write_file":write_file,
    "move_file":move_file,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calc",
            "description": "计算加减乘除和括号的表达式",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的表达式，例如 '2 + 3 * (4 - 1)'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取文件的前五百个字符",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "要读取的文件路径"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "列出一个目录下的文件名，最多返回 10 个；超过时会告诉你还有多少个未显示。",
            "parameters": {
                "type": "object",
                "properties": {
                    "dir_path": {
                        "type": "string",
                        "description": "要列出文件的目录路径"
                    },
                    "start": {
                        "type": "integer",
                        "description": "从第几个开始列，默认 0。看到「可以用 start=N 继续」时，就把 N 填在这里"
                    }
                },
                "required": ["dir_path"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "把一段文字写入文件，只能写在 output 目录下。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "文件名，例如 note.txt"
                    },
                    "content": {
                        "type": "string",
                        "description": "要写入的文字"
                    }
                },
                "required": ["path", "content"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "move_file",
            "description": "把文件从一个位置移动到另一个位置，用于整理文件。",
            "parameters": {
                "type": "object",
                "properties": {
                    "src": {
                        "type": "string",
                        "description": "源文件的完整路径"
                    },
                    "dst": {
                        "type": "string",
                        "description": "目标完整路径，要包含文件名"
                    }
                },
                "required": ["src", "dst"],
            }
        }
    },
]

def save_state(messages, step, trace):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "messages": messages,
                "step": step,
                "trace": trace
            },
            f,
            ensure_ascii=False,
            default=lambda o: o.model_dump() if hasattr(o, "model_dump") else str(o),
            )

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def clear_state():
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)

def write_trace(trace):
    d = os.path.join(BASE_DIR, "traces")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "trace.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(trace, ensure_ascii=False) + "\n")

def run(question, need_files=0):
    system = (
        "你是一个助手。不确定的事就说不知道，不要编。\n"
        f"用户的问题是：{question}\n"
        "不管做到第几步，最终都要回答问题本身。"
    )
    messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]

    saved = load_state()
    if saved:
        print(f"  [恢复] 发现存档，从第 {saved['step']} 步接着跑")
        messages = saved["messages"]
        step = saved["step"]
        trace = saved["trace"]
        max_steps = max(5, step + 5)
    else:
        step = 0
        max_steps = 5
        trace = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "question": question,
            "steps": [],
            "answer": "",
        }
        resp =client.chat.completions.create(
            model=MODEL,
            messages=messages
            + [{"role": "user", "content": "先用三行以内说你的执行计划：第一步做什么、第二步做什么、做到什么程度算完成。只列计划，不要调用任何工具。"}],
            tools=TOOLS
        )

        plan = resp.choices[0].message.content or ""
        print(f"\n [计划]{plan}\n")
        messages.append({"role": "assistant", "content": plan})
        messages.append({"role": "user", "content": "就按上面的计划执行。"})
        trace["plan"] = plan

    while step < max_steps:
        step += 1
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
        )
        msg = resp.choices[0].message

        # 不调工具了，交卷
        if not msg.tool_calls:
            done = len({
                s["args"].get("path")
                for s in trace["steps"]
                if s["tool"] == "read_file" and s["args"].get("path")
            })
            if done < need_files:
                print(f" [完成校验] 只做了{done}/{need_files}次文件操作，打回")
                messages.append({"role": "user", "content": f"你还没做完：至少还需要{need_files - done}次文件操作。继续。"})
                continue
            trace["answer"] = msg.content
            write_trace(trace)
            clear_state()
            return msg.content

        messages.append(msg)

        #一个个真去调
        for c in msg.tool_calls:
            args = json.loads(c.function.arguments)
            print(f" [第{step}步] {c.function.name}({args})", end="... ")
            result = IMPL[c.function.name](**args)
            print(result)
            trace["steps"].append({
                "seq": len(trace["steps"]) + 1,
                "step": step,
                "tool": c.function.name,
                "args": args,
                "result": str(result)[:200],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": c.id,
                "content": str(result)
            })

        save_state(messages, step, trace)

        if step >= max_steps and max_steps < 20:
            max_steps += 5
            print(f"  [续期] 已用 {step}步，预算延到{max_steps}步")
    
    trace["answer"] = "步数用完了"
    write_trace(trace)
    clear_state()
    return "步数用完了"

if __name__ == "__main__":
    print(run("把 C:\\Users\\ZQJ\\AppData\\Local\\Temp\\messy 目录里的文件按类型整理：文档类（txt、md、docx）移到「文档」子目录，代码类（py）移到「代码」，数据类（json、csv、xlsx）移到「数据」，图片类（png、jpg）移到「图片」，其余移到「其他」。移完告诉我每个目录里有什么。"))
