"""Dataset tokenisation and loss masking, using the real tokenizer and small jsonl files."""
import json
import os
import random
import pytest
import torch
from conftest import DATA
from dataset.lm_dataset import (AgentRLDataset, DPODataset, PretrainDataset, RLAIFDataset, SFTDataset, assistant_spans,
                                post_processing_chat, pre_processing_chat, split_tools)
from trainer.trainer_utils import SkipBatchSampler


def test_assistant_spans():
    bos, eos = [7, 8], [9]
    ids = [1, 7, 8, 4, 5, 9, 2, 7, 8, 6, 9, 0, 0]
    assert [list(s) for s in assistant_spans(ids, bos, eos, max_length=13)] == [[3, 4, 5], [9, 10]]
    assert [list(s) for s in assistant_spans(ids, bos, eos, max_length=10)] == [[3, 4, 5], [9]]          # clipped
    assert [list(s) for s in assistant_spans([7, 8, 4, 4], bos, eos, max_length=4)] == [[2, 3]]        # unterminated


def test_split_tools_decodes_json_strings():
    conv = [{'role': 'system', 'content': '', 'tools': '[{"name": "f"}]'},
            {'role': 'assistant', 'content': '', 'tool_calls': '[{"name": "f", "arguments": {}}]'}]
    messages, tools = split_tools(conv)
    assert tools == [{'name': 'f'}] and messages[1]['tool_calls'][0]['name'] == 'f'
    assert isinstance(conv[1]['tool_calls'], str)  # input not mutated


def test_pre_and_post_processing():
    tool_conv = [{'role': 'user', 'content': 'x', 'tools': '[]'}]
    assert pre_processing_chat(tool_conv, add_system_ratio=1.0) is tool_conv
    assert pre_processing_chat([{'role': 'user', 'content': 'x'}], add_system_ratio=1.0)[0]['role'] == 'system'
    assert pre_processing_chat([{'role': 'user', 'content': 'x'}], add_system_ratio=0.0)[0]['role'] == 'user'
    assert post_processing_chat('a<think>\n\n</think>\n\nb', empty_think_ratio=0.0) == 'ab'
    assert post_processing_chat('a<think>\n\n</think>\n\nb', empty_think_ratio=1.0) == 'a<think>\n\n</think>\n\nb'


def test_pretrain_dataset(tokenizer):
    ds = PretrainDataset(os.path.join(DATA, 'pretrain.jsonl'), tokenizer, max_length=64)
    ids, labels = ds[0]
    assert ids.shape == labels.shape == (64,) and ids[0] == tokenizer.bos_token_id
    real = ids != tokenizer.pad_token_id
    assert ids[real][-1] == tokenizer.eos_token_id
    assert (labels[~real] == -100).all() and torch.equal(labels[real], ids[real])


def test_sft_labels_cover_exactly_the_assistant_replies(tokenizer):
    random.seed(0)
    ds = SFTDataset(os.path.join(DATA, 'sft.jsonl'), tokenizer, max_length=4096)
    for i in range(4):
        ids, labels = ds[i]
        labelled = tokenizer.decode(labels[labels != -100])
        for turn in ds.samples[i]['conversations']:
            snippet = (turn['content'] or '').strip()[:20]
            if not snippet: continue
            if turn['role'] == 'assistant': assert snippet in labelled
            elif turn['role'] in ('user', 'system'): assert snippet not in labelled
        assert labelled.endswith('<|im_end|>\n')  # each reply is trained through its end marker


def test_dpo_dataset_shapes_and_masks(tokenizer):
    ds = DPODataset(os.path.join(DATA, 'dpo.jsonl'), tokenizer, max_length=128)
    item = ds[0]
    assert all(v.shape == (127,) for v in item.values())
    for side in ('chosen', 'rejected'):
        x, y, m = item[f'x_{side}'], item[f'y_{side}'], item[f'mask_{side}']
        torch.testing.assert_close(x[1:], y[:-1])          # y is x shifted by one
        assert m.sum() > 0 and set(m.tolist()) <= {0, 1}
        assert '<|im_start|>user' not in tokenizer.decode(y[m.bool()])


def test_rlaif_prompt_drops_the_answer(tokenizer):
    off = RLAIFDataset(os.path.join(DATA, 'rlaif.jsonl'), tokenizer, max_length=512, thinking_ratio=0.0)
    on = RLAIFDataset(os.path.join(DATA, 'rlaif.jsonl'), tokenizer, max_length=512, thinking_ratio=1.0)
    assert off[0]['prompt'].endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n')  # thinking off: empty think block
    assert on[0]['prompt'].endswith('<|im_start|>assistant\n<think>\n')                  # thinking on: model writes the think block
    turns = off.samples[0]['conversations']
    assert off[0]['prompt'].count('<|im_start|>') == len(turns) - 1 + 1  # every turn but the answer, plus the generation prompt


def test_agent_dataset(tokenizer):
    ds = AgentRLDataset(os.path.join(DATA, 'agent.jsonl'), tokenizer)
    raw = json.loads(open(os.path.join(DATA, 'agent.jsonl')).readline())
    item = ds[0]
    assert len(item['messages']) == len(raw['conversations']) - 1 and item['gt'] == raw['gt']
    assert item['tools'] is None or isinstance(item['tools'], list)


@pytest.mark.parametrize('n,batch,skip,expected', [(10, 3, 0, [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]), (10, 3, 2, [[6, 7, 8], [9]]),
                                                    (6, 3, 2, []), (6, 3, 5, [])])
def test_skip_batch_sampler(n, batch, skip, expected):
    sampler = SkipBatchSampler(list(range(n)), batch, skip)
    assert list(sampler) == expected and len(sampler) == len(expected)
