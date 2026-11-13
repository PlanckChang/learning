"""最小 agent loop：用同一次任务看清 message 列表、chat template、模型输入和模型输出。

任务（写死，所以每次跑出来的文本都一样）:
    用户: 上海现在多少度？把这个温度乘 2 再加 3。
    模型第 1 次: 调用 get_weather("上海")
    工具: {"city": "上海", "temp_c": 18}
    模型第 2 次: 调用 calc("18*2+3")
    工具: {"value": 39}
    模型第 3 次: 上海现在 18°C。18 × 2 + 3 = 39。

三层数据，不要混在一起:

1. messages
   程序里的 list[dict]。循环只往末尾追加，不改旧条目。
2. chat template
   apply_chat_template(messages) 把整份列表渲染成一个字符串。
   这是模型训练时见过的版式。调用模型之前渲染一次；
   模型原文写回历史时，还是用同一套规则包上角色标记。
3. 模型的输入和输出
   输入是这个字符串 tokenize 之后的 token id。
   输出是从最后一个 id 往后续写的 token，解码成字符串，
   碰到停止符 <|im_end|> 就停。程序再把这段字符串解析成
   一条 assistant 消息，append 回 messages。

下面的 SCRIPTED 代替真实的 model.generate，方便把每一层原文印出来。
真实调用长这样（本地模型，例如 Qwen2.5）:

    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    )
    output_ids = model.generate(
        input_ids,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
    )
    raw = tokenizer.decode(
        output_ids[0][len(input_ids[0]):],
        skip_special_tokens=True,
    )

Qwen2.5 里 <|im_start|> 的 id 是 151644，<|im_end|> 的 id 是 151645。
这份模板是教学用的精简 ChatML，标记和某家官方 Jinja 不会逐字相同。
循环本身不变: 整表渲染、续写、解析、追加。
"""

from __future__ import annotations

import ast
import json

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

# Qwen2.5 词表里这两个特殊符的真实 id。正文还会再被 BPE 切开，这里不伪造那些 id。
SPECIAL_IDS = {IM_START: 151644, IM_END: 151645}

SYSTEM = (
    "你是助手。不知道的数字必须调用工具，禁止编造。\n"
    "工具:\n"
    "- get_weather(city: string) 返回 {city, temp_c}\n"
    "- calc(expression: string) 返回 {value}，expression 只能含数字和 + - * /"
)

USER = "上海现在多少度？把这个温度乘 2 再加 3。"


def apply_chat_template(messages: list[dict], add_generation_prompt: bool = True) -> str:
    """把 message 列表渲染成模型要读的那一个字符串。

    规则:
    - system / user / 纯文本 assistant:
        <|im_start|>{role}\\n{content}\\n<|im_end|>\\n
    - 带 tool_calls 的 assistant:
        正文是一条或多条 <tool_call>{"name","arguments"}</tool_call>
        arguments 在列表里是字符串，这里 json.loads 之后按对象写回，
        这样第二次渲染才能和模型当初写下的原文一致。
    - tool:
        <|im_start|>tool\\n<tool_response ...>\\n{content}\\n</tool_response><|im_end|>\\n
    - add_generation_prompt:
        末尾再加 <|im_start|>assistant\\n，不加 <|im_end|>。
        模型从这里继续写。这个结尾还没有关闭，所以它不是一条完整消息。
    """
    chunks = [render_message(message) for message in messages]
    if add_generation_prompt:
        chunks.append(f"{IM_START}assistant\n")
    return "".join(chunks)


def render_message(message: dict) -> str:
    role = message["role"]
    if role == "assistant" and message.get("tool_calls"):
        parts: list[str] = []
        if message.get("content"):
            parts.append(message["content"])
        for call in message["tool_calls"]:
            payload = {
                "name": call["function"]["name"],
                "arguments": json.loads(call["function"]["arguments"]),
            }
            parts.append(
                "<tool_call>\n" + json.dumps(payload, ensure_ascii=False) + "\n</tool_call>"
            )
        body = "\n".join(parts)
    elif role == "tool":
        body = (
            f'<tool_response id="{message["tool_call_id"]}" name="{message["name"]}">\n'
            f'{message["content"]}\n'
            "</tool_response>"
        )
    else:
        body = message["content"]
    # <|im_end|> 独占一行，只为了读起来清楚。模型原文本身不带这个换行。
    return f"{IM_START}{role}\n{body}\n{IM_END}\n"


def parse_model_output(raw: str) -> dict:
    """模型续写的字符串 -> 一条 assistant 消息。

    采样停在 <|im_end|> 上，这个停止符不放进 raw。
    没有 <tool_call> 时，整段就是给用户看的 content，循环结束。
    有 <tool_call> 时，content 为空，arguments 以 JSON 字符串保存。
    字符串这个形状是接口层的选择: 模型是一个 token 一个 token 吐出来的，
    吐出来的本来就是文本。
    """
    text = raw.strip()
    if text.endswith(IM_END):
        text = text[: -len(IM_END)].strip()

    if "<tool_call>" not in text:
        return {"role": "assistant", "content": text}

    calls = []
    rest = text
    while "<tool_call>" in rest:
        start = rest.index("<tool_call>") + len("<tool_call>")
        end = rest.index("</tool_call>")
        payload = json.loads(rest[start:end])
        calls.append(
            {
                "id": "",
                "type": "function",
                "function": {
                    "name": payload["name"],
                    "arguments": json.dumps(payload["arguments"], ensure_ascii=False),
                },
            }
        )
        rest = rest[end + len("</tool_call>") :]
    prefix = text.split("<tool_call>", 1)[0].strip()
    return {
        "role": "assistant",
        "content": prefix or None,
        "tool_calls": calls,
    }


def get_weather(city: str) -> dict:
    table = {"上海": 18}
    if city not in table:
        return {"city": city, "error": "unknown city"}
    return {"city": city, "temp_c": table[city]}


def calc(expression: str) -> dict:
    tree = ast.parse(expression, mode="eval")
    allowed = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.UAdd,
        ast.USub,
        ast.Load,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError(f"不允许的表达式: {expression}")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError(f"不允许的字面量: {expression}")
    value = eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, {})
    return {"value": value}


def run_tool(call: dict) -> dict:
    name = call["function"]["name"]
    args = json.loads(call["function"]["arguments"])
    if name == "get_weather":
        result = get_weather(**args)
    elif name == "calc":
        result = calc(**args)
    else:
        result = {"error": f"unknown tool {name}"}
    return {
        "role": "tool",
        "tool_call_id": call["id"],
        "name": name,
        "content": json.dumps(result, ensure_ascii=False),
    }


# 真实模型会根据 prompt 自己生成这三段。这里按调用顺序写死，用来展示数据长什么样。
SCRIPTED_OUTPUTS = [
    '<tool_call>\n{"name": "get_weather", "arguments": {"city": "上海"}}\n</tool_call>',
    '<tool_call>\n{"name": "calc", "arguments": {"expression": "18*2+3"}}\n</tool_call>',
    "上海现在 18°C。18 × 2 + 3 = 39。",
]


def scripted_llm(prompt: str, call_index: int) -> str:
    """占位的模型。prompt 就是本来要 tokenize 的字符串，这里接收它但不生成。"""
    del prompt
    return SCRIPTED_OUTPUTS[call_index]


def special_token_note(prompt: str) -> str:
    starts = prompt.count(IM_START)
    ends = prompt.count(IM_END)
    return (
        f"tokenize 之后，{IM_START} = {SPECIAL_IDS[IM_START]}，"
        f"{IM_END} = {SPECIAL_IDS[IM_END]}（Qwen2.5 的真实 id）。"
        f"这次字符串里有 {starts} 个 im_start、{ends} 个 im_end，各算 1 个 token；"
        "夹在中间的正文再被 BPE 切成很多 id。"
        "模型的输入就是这条 id 序列，它从最后一个 id 的下一个位置开始预测。"
    )


def dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def agent_loop(messages: list[dict]) -> list[dict]:
    call_index = 0
    next_call_id = 1
    while True:
        prompt = apply_chat_template(messages, add_generation_prompt=True)
        print(f"\n======== 第 {call_index + 1} 次调用模型 ========")
        print(f"--- 此时的 messages（{len(messages)} 条，旧条目原样保留）---")
        print(dump(messages))
        print("--- chat template 的产物，也就是模型要读的字符串 ---")
        print(prompt, end="" if prompt.endswith("\n") else "\n")
        print("--- " + special_token_note(prompt))

        raw = scripted_llm(prompt, call_index)
        call_index += 1
        print("--- 模型续写（解码后的字符串；停止符 <|im_end|> / id 151645 已去掉）---")
        print(raw)

        assistant = parse_model_output(raw)
        for call in assistant.get("tool_calls", []):
            call["id"] = f"call_{next_call_id}"
            next_call_id += 1
        print("--- 解析后追加的 assistant 消息 ---")
        print(dump(assistant))
        print("--- 同一条消息写回历史时，模板把它包成 ---")
        print(render_message(assistant), end="")

        # 写回历史的正文必须等于模型原文，否则下一轮模型会看到另一段话。
        rendered_body = render_message(assistant)
        body = rendered_body[len(f"{IM_START}assistant\n") : -len(f"\n{IM_END}\n")]
        if body != raw:
            raise AssertionError(f"模板回写和模型原文不一致:\n{body}\n---\n{raw}")

        messages.append(assistant)
        if not assistant.get("tool_calls"):
            print("--- 没有 tool_calls，循环停止 ---")
            break

        for call in assistant["tool_calls"]:
            tool_message = run_tool(call)
            messages.append(tool_message)
            print("--- 工具执行后追加的 tool 消息 ---")
            print(dump(tool_message))

        if call_index >= len(SCRIPTED_OUTPUTS):
            raise RuntimeError("脚本里的模型回复用完了，循环还没结束")

    return messages


def main() -> None:
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER},
    ]
    agent_loop(messages)

    print("\n======== 循环结束后的 messages ========")
    print(dump(messages))
    print(f"共 {len(messages)} 条。顺序就是对话发生的顺序。")

    archive = apply_chat_template(messages, add_generation_prompt=False)
    print("--- 整段历史再渲染一次（add_generation_prompt=False，用于存档）---")
    print(archive, end="")
    print("--- 如果还要再叫一次模型，同一个列表会在末尾多出这一截，且没有 <|im_end|> ---")
    print(f"{IM_START}assistant\n")

    assert len(messages) == 7
    assert [m["role"] for m in messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    assert messages[2]["tool_calls"][0]["function"]["arguments"] == '{"city": "上海"}'
    assert messages[3]["content"] == '{"city": "上海", "temp_c": 18}'
    assert messages[4]["tool_calls"][0]["function"]["arguments"] == '{"expression": "18*2+3"}'
    assert messages[5]["content"] == '{"value": 39}'
    assert messages[6]["content"] == "上海现在 18°C。18 × 2 + 3 = 39。"
    assert "tool_calls" not in messages[6]

    for count in (2, 4, 6):
        prefix = messages[:count]
        prompt = apply_chat_template(prefix, add_generation_prompt=True)
        assert prompt.startswith(f"{IM_START}system\n")
        assert USER in prompt
        assert prompt.endswith(f"{IM_START}assistant\n")
    second = apply_chat_template(messages[:4], add_generation_prompt=True)
    assert '"temp_c": 18' in second
    third = apply_chat_template(messages[:6], add_generation_prompt=True)
    assert '"value": 39' in third
    assert archive.endswith("上海现在 18°C。18 × 2 + 3 = 39。\n" + IM_END + "\n")
    print("--- 断言通过：三次 prompt 都从 system 开始、以未关闭的 assistant 结尾；最终 7 条消息 ---")


if __name__ == "__main__":
    main()
