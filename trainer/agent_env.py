"""Simulated tool environment for agent RL: tool schemas, mock execution, multi-turn rollout and reward."""
import json
import math
import random
import re
import signal
from dataclasses import dataclass, field
from typing import List

import torch

from trainer.rl_common import parse_chat_prompt, rep_penalty
from trainer.trainer_utils import Logger

# ======== 工具定义 ========
TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "unit_converter", "description": "单位换算", "parameters": {"type": "object", "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}}, "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "get_current_weather", "description": "获取天气", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "获取时间", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "default": "Asia/Shanghai"}}, "required": []}}},
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "查询汇率", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string"}, "to_currency": {"type": "string"}}, "required": ["from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "翻译文本", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target_language": {"type": "string"}}, "required": ["text", "target_language"]}}},
]

# ======== 模拟数据 ========
WEATHER_DATA = {"北京": ("28°C", "晴"), "上海": ("15°C", "多云"), "广州": ("32°C", "闷热"), "深圳": ("30°C", "晴"), "杭州": ("22°C", "阴"), "成都": ("18°C", "小雨"), "武汉": ("25°C", "多云"), "南京": ("20°C", "晴"), "西安": ("16°C", "大风"), "重庆": ("26°C", "阴"), "Tokyo": ("12°C", "晴"), "New York": ("8°C", "多云"), "London": ("5°C", "小雨"), "Paris": ("10°C", "阴"), "Sydney": ("25°C", "晴朗")}
TIME_DATA = {"Asia/Shanghai": "2025-03-07 14:30:00", "America/New_York": "2025-03-07 01:30:00", "Europe/London": "2025-03-07 06:30:00", "Asia/Tokyo": "2025-03-07 15:30:00", "Europe/Paris": "2025-03-07 07:30:00", "Australia/Sydney": "2025-03-07 17:30:00"}
EXCHANGE_DATA = {("USD", "CNY"): 7.21, ("EUR", "CNY"): 7.85, ("GBP", "CNY"): 9.12, ("JPY", "CNY"): 0.048, ("USD", "EUR"): 0.92, ("USD", "GBP"): 0.79, ("CNY", "JPY"): 20.83, ("AUD", "CNY"): 4.72}
TRANSLATE_DATA = {("你好世界", "english"): "Hello World", ("Good morning", "chinese"): "早上好", ("今天天气真好", "english"): "The weather is nice today", ("I love programming", "chinese"): "我喜欢编程", ("机器学习很有趣", "english"): "Machine learning is interesting", ("Happy birthday", "chinese"): "生日快乐"}
UNIT_DATA = {"km_miles": 0.621371, "miles_km": 1.60934, "kg_pounds": 2.20462, "pounds_kg": 0.453592, "meters_feet": 3.28084, "feet_meters": 0.3048, "celsius_fahrenheit": 1.8, "fahrenheit_celsius": 0.5556}


def _calculate(expression):
    expr = str(expression).replace("^", "**").replace("×", "*").replace("÷", "/").replace("−", "-").replace("（", "(").replace("）", ")")
    return str(eval(expr, {"__builtins__": {}, "math": math}))  # no builtins: arithmetic and math.* only


# ======== 模拟执行 ========
MOCK_RESULTS = {
    "calculate_math": lambda args: {"result": _calculate(args.get("expression", "0"))},
    "unit_converter": lambda args: {"result": round(float(args.get("value", 0)) * UNIT_DATA.get(f"{args.get('from_unit', '').lower()}_{args.get('to_unit', '').lower()}", 1), 4)},
    "get_current_weather": lambda args: (lambda w: {"city": args.get("location"), "temperature": w[0], "humidity": "65%", "condition": w[1]})(WEATHER_DATA.get(args.get("location"), ("22°C", "晴"))),
    "get_current_time": lambda args: {"datetime": TIME_DATA.get(args.get("timezone", "Asia/Shanghai"), "2025-03-07 14:30:00"), "timezone": args.get("timezone", "Asia/Shanghai")},
    "get_exchange_rate": lambda args: {"from": args.get("from_currency"), "to": args.get("to_currency"), "rate": EXCHANGE_DATA.get((args.get("from_currency"), args.get("to_currency")), 1.0)},
    "translate_text": lambda args: {"translated_text": TRANSLATE_DATA.get((args.get("text"), args.get("target_language")), args.get("text", ""))},
}

# ======== 参数校验 ========
CHECK_ARGS = {
    "calculate_math": lambda a: bool(a.get("expression")),
    "unit_converter": lambda a: a.get("value") is not None and a.get("from_unit") and a.get("to_unit"),
    "get_current_weather": lambda a: bool(a.get("location")),
    "get_current_time": lambda a: True,
    "get_exchange_rate": lambda a: bool(a.get("from_currency")) and bool(a.get("to_currency")),
    "translate_text": lambda a: bool(a.get("text")) and bool(a.get("target_language")),
}


# ======== 工具调用解析与执行 ========
def parse_tool_calls(text):
    """All well-formed JSON objects inside <tool_call>...</tool_call>; malformed ones are skipped."""
    calls = []
    for m in re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):
        try: calls.append(json.loads(m.strip()))
        except Exception: pass
    return calls


def parse_arguments(raw):
    """Tool-call arguments may arrive as a dict or as a JSON string; anything unparsable becomes {}."""
    if isinstance(raw, str):
        try: return json.loads(raw)
        except Exception: return {}
    return raw


def execute_tool(name, args):
    """Run a mock tool with a 1 s alarm; None for unknown tools or any failure."""
    fn = MOCK_RESULTS.get(name)
    if not fn: return None
    try:
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
        signal.alarm(1)
        return fn(args)
    except Exception:
        return None
    finally:
        try: signal.alarm(0)
        except Exception: pass


# ======== 多轮 Rollout ========
@dataclass
class Trajectory:
    """One multi-turn episode. response_ids interleaves model tokens (mask 1, with old log-probs) and tool
    observations / chat-template glue (mask 0, log-prob 0)."""
    completion: str = ""
    context: str = ""
    prompt_ids: List[int] = field(default_factory=list)
    response_ids: List[int] = field(default_factory=list)
    response_mask: List[int] = field(default_factory=list)
    old_logps: List[float] = field(default_factory=list)
    turn_outputs: List[str] = field(default_factory=list)
    unfinished: bool = False


def rollout_single(rollout_engine, tokenizer, messages, tools, max_turns=3, max_new_tokens=256, thinking_ratio=0.5, device="cuda"):
    """Generate, execute any tool calls, append the observations and continue, for up to max_turns model turns."""
    traj = Trajectory()
    prompt_ids = None
    open_thinking = random.random() < thinking_ratio
    for turn in range(max_turns):
        context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools, open_thinking=open_thinking)
        inputs = tokenizer(context, return_tensors="pt", add_special_tokens=False).to(device)
        if prompt_ids is None:
            prompt_ids = inputs["input_ids"][0].tolist()
        result = rollout_engine.rollout(prompt_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                                        num_generations=1, max_new_tokens=max_new_tokens, temperature=0.8)
        new_ids, new_logps = result.completion_ids[0].tolist(), result.per_token_logps[0].tolist()
        if len(new_ids) != len(new_logps): Logger(f"rollout token/logprob length mismatch: {len(new_ids)} vs {len(new_logps)}")
        pairs = [(t, lp) for t, lp in zip(new_ids, new_logps) if t != tokenizer.pad_token_id and t != tokenizer.eos_token_id]
        new_text = result.completions[0]
        traj.turn_outputs.append(new_text)
        traj.response_ids.extend(t for t, _ in pairs)
        traj.response_mask.extend([1] * len(pairs))
        traj.old_logps.extend(lp for _, lp in pairs)
        traj.context = context + new_text
        calls = parse_tool_calls(new_text)
        if not calls:
            break
        traj.unfinished = turn == max_turns - 1
        messages.append({"role": "assistant", "content": new_text})
        for call in calls:
            result = execute_tool(call.get("name", ""), parse_arguments(call.get("arguments", {})))
            result_str = (json.dumps(result, ensure_ascii=False) if result else '{"error": "tool not found"}')[:2048]  # 防止天文数字撑爆tokenizer
            messages.append({"role": "tool", "content": result_str})

        observe_context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=not traj.unfinished, tools=tools, open_thinking=open_thinking)
        observe_ids = tokenizer(observe_context, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        obs_delta = observe_ids[len(prompt_ids) + len(traj.response_ids):]
        traj.response_ids.extend(obs_delta)
        traj.response_mask.extend([0] * len(obs_delta))
        traj.old_logps.extend([0.0] * len(obs_delta))
        traj.context = observe_context

    traj.completion = traj.turn_outputs[-1] if traj.turn_outputs else ""
    traj.prompt_ids = prompt_ids or []
    return traj


def rollout_batch(rollout_engine, tokenizer, messages_batch, tools_batch, num_gen, max_turns=3, max_new_tokens=256, thinking_ratio=0.5, device="cuda"):
    """num_gen independent trajectories per sample, grouped by sample (sample-major order)."""
    return [rollout_single(rollout_engine, tokenizer, [dict(m) for m in messages], tools, max_turns, max_new_tokens, thinking_ratio, device)
            for messages, tools in zip(messages_batch, tools_batch) for _ in range(num_gen)]


# ======== Reward 计算 ========
def validate_gt_in_text(text, gt_list):
    """Ground-truth items found in text: case-insensitive substring match, or numeric match within 1e-6."""
    text, text_num = str(text), str(text).replace(',', '')
    nums = [float(x) for x in re.findall(r'(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w.])', text_num)]
    return {g for g in gt_list if ((s := str(g).strip()) and s.lower() in text.lower()) or (re.fullmatch(r'[-+]?\d+(?:\.\d+)?', str(g).strip().replace(',', '')) and any(abs(float(str(g).strip().replace(',', '')) - n) < 1e-6 for n in nums))}


def _strip_think(turn):
    return turn.split('</think>', 1)[-1].strip() if '</think>' in turn else turn.strip()


def chat_reward(response, answer, prompt, reward_model):
    """No tool call: length / think-format shaping + reward-model score - repetition. `answer` is the stripped final
    turn; when the response has a think block, the text after </think> is scored instead."""
    reward = 0.0
    reward += 0.5 if 5 <= len(response.strip()) <= 800 else -0.5  # 长度分
    if '</think>' in response:
        think, answer = response.split('</think>', 1)
        reward += 1.0 if 20 <= len(think.strip()) <= 300 else -0.5  # 思考长度分
        reward += 0.25 if response.count('</think>') == 1 else -0.25  # 思考闭合分
        answer = answer.strip()
    if reward_model is not None:
        reward += reward_model.get_score(parse_chat_prompt(prompt), answer)  # RM分
    return reward - rep_penalty(answer), answer


def tool_reward(tool_calls, answer, tools, gt, unfinished):
    """Tool calls made: reward matching the expected call count with valid arguments, and ground truth in the final answer."""
    valid_names = {t['function']['name'] for t in tools} if tools else set()
    valid_call_count = 0
    for tool_call in tool_calls:
        name, args = tool_call.get("name", ""), parse_arguments(tool_call.get("arguments", {}))
        check = CHECK_ARGS.get(name)
        valid_call_count += int(bool(name in valid_names and check and check(args)))
    tool_gap = abs(valid_call_count - len(gt)) + max(0, len(tool_calls) - valid_call_count)  # tool数差值
    reward = 0.5 if tool_gap == 0 else -0.5 * tool_gap  # tool对齐分
    final_text = "" if unfinished else (answer.split('</tool_call>')[-1] if '</tool_call>' in answer else answer)
    verified = validate_gt_in_text(final_text, gt) if gt else set()
    if gt: reward += 2.5 * len(verified) / len(gt)  # GT分
    if unfinished: reward -= 0.5  # 未完成扣分
    return reward - rep_penalty(final_text if final_text else answer)


def calculate_rewards(prompts, completions, gt_batch, tools_batch, num_gen, reward_model=None, device="cuda", turn_outputs_batch=None, unfinished_batch=None):
    """Reward per trajectory, clipped to [-3, 3]; unbalanced <tool_call> tags cost 0.5 each."""
    rewards = torch.zeros(len(completions), device=device)
    for idx, response in enumerate(completions):
        sample_idx = idx // num_gen
        turn_outputs = turn_outputs_batch[idx] if turn_outputs_batch is not None else [response]
        unfinished = unfinished_batch[idx] if unfinished_batch is not None else False
        turn_answers = [_strip_think(turn) for turn in turn_outputs]
        answer = turn_answers[-1] if turn_answers else response.strip()
        tool_calls = [call for turn_answer in turn_answers for call in parse_tool_calls(turn_answer)]
        tag_penalty = 0.5 * sum(abs(turn.count('<tool_call>') - turn.count('</tool_call>')) for turn in turn_answers)  # 标签扣分
        if not tool_calls:
            reward, _ = chat_reward(response, answer, prompts[sample_idx], reward_model)
        else:
            reward = tool_reward(tool_calls, answer, tools_batch[sample_idx], gt_batch[sample_idx], unfinished)
        rewards[idx] = max(min(reward - tag_penalty, 3.0), -3.0)  # 总分Clip
    return rewards
