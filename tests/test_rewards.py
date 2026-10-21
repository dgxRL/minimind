"""Reward shaping (GRPO / PPO) and the simulated tool environment (agent RL)."""
import pytest
import torch
from conftest import StubRewardModel
from trainer import agent_env
from trainer.agent_env import (Trajectory, calculate_rewards, execute_tool, parse_arguments, parse_tool_calls, rollout_single,
                               validate_gt_in_text)
from trainer.rl_common import format_reward_terms, parse_chat_prompt, rep_penalty, score_responses


def test_rep_penalty():
    assert rep_penalty("the quick brown fox jumps over the lazy dog") == 0
    assert rep_penalty("a b") == 0
    assert rep_penalty("go go go go go go go go go go") == 0.5
    assert 0 < rep_penalty("one two three one two three four five six seven eight nine ten") < 0.5


def test_parse_chat_prompt():
    prompt = "<|im_start|>system\nbe nice<|im_end|>\n<|im_start|>user\n hi <|im_end|>\n<|im_start|>assistant\n"
    assert parse_chat_prompt(prompt) == [{'role': 'system', 'content': 'be nice'}, {'role': 'user', 'content': 'hi'}]


@pytest.mark.parametrize('response,expected_terms,answer', [
    ('x' * 19, [-0.5], 'x' * 19),
    ('x' * 20, [0.5], 'x' * 20),
    ('x' * 801, [-0.5], 'x' * 801),
    ('t' * 25 + '</think>' + ' answer is here ok', [0.5, 1.0, 0.25], 'answer is here ok'),
    ('t' * 5 + '</think>a</think>' + 'y' * 20, [0.5, -0.5, -0.25], 'a</think>' + 'y' * 20),
])
def test_format_reward_terms(response, expected_terms, answer):
    terms, got = format_reward_terms(response)
    assert terms[:-1] == expected_terms and got == answer
    assert terms[-1] == -rep_penalty(answer)


def test_score_responses_maps_each_response_to_its_prompt():
    rm = StubRewardModel()
    prompts = ['<|im_start|>user\nP0<|im_end|>\n', '<|im_start|>user\nP1<|im_end|>\n']
    responses = ['r' * 30, 'think' * 5 + '</think>final', 'r' * 31, 'r' * 32]
    rewards = score_responses(prompts, responses, rm, num_generations=2)
    assert [c[0][0]['content'] for c in rm.calls] == ['P0', 'P0', 'P1', 'P1']
    assert rm.calls[1][1] == 'final'                                   # the reward model scores the answer without the think block
    terms, ans = format_reward_terms(responses[0])
    assert rewards[0].item() == pytest.approx(sum(terms) + rm.get_score([], ans), abs=1e-6)


# ── agent tools ──
def test_parse_tool_calls_skips_malformed_json():
    text = '<tool_call>{"name": "a", "arguments": {}}</tool_call> junk <tool_call>{bad</tool_call><tool_call>\n{"name": "b"}\n</tool_call>'
    assert [c['name'] for c in parse_tool_calls(text)] == ['a', 'b']


def test_parse_arguments():
    assert parse_arguments('{"x": 1}') == {'x': 1} and parse_arguments('nope') == {} and parse_arguments({'y': 2}) == {'y': 2}


def test_execute_tool():
    assert execute_tool('calculate_math', {'expression': '2^10 × 3'}) == {'result': '3072'}
    assert execute_tool('get_exchange_rate', {'from_currency': 'USD', 'to_currency': 'CNY'})['rate'] == 7.21
    assert execute_tool('no_such_tool', {}) is None
    assert execute_tool('calculate_math', {'expression': '1/0'}) is None
    assert execute_tool('calculate_math', {'expression': "__import__('os').getcwd()"}) is None  # no builtins in the sandbox


def test_validate_gt_in_text():
    assert validate_gt_in_text('The total is 1,234.0 dollars, in Tokyo', ['1234', 'tokyo', 'paris']) == {'1234', 'tokyo'}
    assert validate_gt_in_text('version 12.5', ['2.5']) == {'2.5'}      # plain substring match comes first and is lenient
    assert validate_gt_in_text('x=3.0', ['3']) == {'3'} and validate_gt_in_text('x=30', ['3.5']) == set()


def test_chat_reward_scores_the_stripped_final_turn():
    """Regression: without tool calls and without a think block, the reward model must see the stripped answer."""
    rm = StubRewardModel()
    calculate_rewards(['<|im_start|>user\nq<|im_end|>\n'], ['  hello there, this is a long enough reply  '], [[]], [None], 1, rm, device='cpu')
    assert rm.calls[0][1] == 'hello there, this is a long enough reply'


def test_tool_rewards():
    tools = [t for t in agent_env.TOOLS if t['function']['name'] == 'calculate_math']
    good = '<tool_call>{"name": "calculate_math", "arguments": {"expression": "2+2"}}</tool_call>'
    final = 'The answer is 4'
    r = calculate_rewards(['p'], [final], [['4']], [tools], 1, None, device='cpu', turn_outputs_batch=[[good, final]], unfinished_batch=[False])
    assert r.item() == pytest.approx(0.5 + 2.5 - rep_penalty(final))      # aligned call count + full ground truth
    wrong = calculate_rewards(['p'], [final], [['5']], [tools], 1, None, device='cpu', turn_outputs_batch=[[good, final]], unfinished_batch=[True])
    assert wrong.item() == pytest.approx(0.5 - 0.5 - rep_penalty(final))  # unfinished: no ground-truth credit, -0.5
    unbalanced = calculate_rewards(['p'], ['<tool_call>oops'], [[]], [None], 1, None, device='cpu')
    plain = calculate_rewards(['p'], ['xxxxxxxxxxxoops'], [[]], [None], 1, None, device='cpu')
    assert unbalanced.item() == pytest.approx(plain.item() - 0.5)       # 0.5 per unmatched <tool_call> tag


def test_rewards_are_clipped():
    rm = StubRewardModel(); rm.get_score = lambda m, a: 100.0
    assert calculate_rewards(['p'], ['x' * 30], [[]], [None], 1, rm, device='cpu').item() == 3.0


class ScriptedEngine:
    """Rollout engine that replays fixed completions — lets rollout_single run without a model."""
    def __init__(self, tokenizer, texts):
        self.tokenizer, self.texts = tokenizer, list(texts)

    def rollout(self, prompt_ids, attention_mask, num_generations, max_new_tokens, temperature=0.8):
        from trainer.rollout_engine import RolloutResult
        text = self.texts.pop(0)
        ids = torch.tensor([self.tokenizer(text, add_special_tokens=False).input_ids + [self.tokenizer.eos_token_id]])
        return RolloutResult(torch.cat([prompt_ids, ids], 1), ids, torch.full(ids.shape, -0.5), [text],
                             torch.tensor([prompt_ids.size(1)]), torch.ones_like(ids))


def test_rollout_single_interleaves_model_tokens_and_observations(tokenizer):
    call = '<tool_call>{"name": "calculate_math", "arguments": {"expression": "6*7"}}</tool_call>'
    engine = ScriptedEngine(tokenizer, [call, 'It is 42.'])
    messages = [{'role': 'user', 'content': 'what is 6*7?'}]
    traj = rollout_single(engine, tokenizer, messages, agent_env.TOOLS, max_turns=3, thinking_ratio=0.0, device='cpu')
    assert isinstance(traj, Trajectory) and traj.turn_outputs == [call, 'It is 42.'] and traj.completion == 'It is 42.' and not traj.unfinished
    assert messages[-1] == {'role': 'tool', 'content': '{"result": "42"}'}
    assert len(traj.response_ids) == len(traj.response_mask) == len(traj.old_logps)
    model_tokens = [t for t, m in zip(traj.response_ids, traj.response_mask) if m]
    assert tokenizer.decode(model_tokens) == call + 'It is 42.'           # EOS dropped, observations masked out
    assert all(lp == 0.0 for lp, m in zip(traj.old_logps, traj.response_mask) if not m)
    assert '{"result": "42"}' in tokenizer.decode([t for t, m in zip(traj.response_ids, traj.response_mask) if not m])


def test_rollout_single_flags_unfinished_when_turns_run_out(tokenizer):
    call = '<tool_call>{"name": "get_current_time", "arguments": {}}</tool_call>'
    traj = rollout_single(ScriptedEngine(tokenizer, [call, call]), tokenizer, [{'role': 'user', 'content': 't?'}], agent_env.TOOLS,
                          max_turns=2, thinking_ratio=0.0, device='cpu')
    assert traj.unfinished and len(traj.turn_outputs) == 2
