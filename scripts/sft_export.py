"""Export resolved agent traces as SFT data, rendered exactly as the scorer's vLLM renders each model call.

    uv run python scripts/sft_export.py build --dataset sa_think_v1 --mode think
    uv run python scripts/sft_export.py build --dataset sa_nothink_v1 --mode nothink
    uv run python scripts/sft_export.py build --dataset ma_v1 --per-agent --val-tasks-from sa_think_v1 \\
        --arms v008_editfirst_cap14,v008b_editfirst_think1k_cap14,v008c_editfirst_nothink_cap14
    uv run python scripts/sft_export.py build --dataset sa_think_tail_v1 --mode think --tail \\
        --val-tasks-from sa_think_v1 --trajectories-from sa_think_v1 --trajectories-cut compaction
    uv run python scripts/sft_export.py verify --dataset sa_think_v1          # decode samples, check masks
    uv run python scripts/sft_export.py check-log --log <vllm_server.log> --traces <dir> --submission <dir>

--per-agent: multi-agent arms, one sample per sub-agent call sequence with that agent's own context (ADK
include_contents and 'For context:' presentation of other agents), checked call by call against the trace.
--tail: the turns after each teacher compaction, in the context ADK compaction leaves (system turn, the summary as a
'For context:[model] said: ...' user message, the retained events) with a synthetic summary built from the trace.
Evidence and counts: docs/experiments/sft-data-20261029.md.

Output: data/sft/<dataset>/{train,val}.jsonl, one sample per line:
    {"id", "instance_id", "repo", "arm", "agent", "n_tokens", "input_ids", "loss_mask"}
plus meta.json (tokenizer and template paths and sha256, rendering rules, counts, split, task lists),
index.jsonl (per-sample provenance: run dir, segment, turns, loss tokens, cut reason) and READY when complete.

How a prompt is rebuilt (each step is the real code where it can be, a copy of the code where it cannot):
1. The trace (adk_eval_core ATIF JSON) is regrouped into model calls by event timestamp: thought text, visible
   text, tool calls with arguments, tool responses (json.loads of the recorded JSON), usage.
2. The calls become ADK session events. The submission is compiled with adk_submission.compile_submission and
   swegemma's tool functions, and ADK's own LLM flow preprocessing (instructions, identity, contents, tool
   declarations) builds the LlmRequest; the adk_submission LiteLLM patch and LiteLlm._get_completion_inputs turn it
   into OpenAI messages and tools. So the system prompt (instruction + ADK identity line), tool schemas, reasoning
   fields and tool-message format are what the harness sends.
3. vLLM 0.19.1's parse_chat_messages is mirrored (content format 'openai', detected from the template the way vLLM
   detects it; 'reasoning' copied to both reasoning fields; tool-call arguments json.loads'ed), and the model's
   chat_template.jinja is applied with tools, add_generation_prompt=True and enable_thinking from the config's
   thinking_budget. Tokens: tokenizer(text, add_special_tokens=False), as vLLM does for chat requests.
4. Every rebuilt prompt is checked against the trace's prompt_tokens for that call (vLLM usage). A trajectory is cut
   at the first call that does not match.

Sample rules (see docs/experiments/sft-data-20261029.md for the evidence):
- The Gemma 4 template keeps the thought of every assistant tool-call message after the last user message and drops
  thoughts of text-only messages and of everything before the last user message. A harness nudge is a user
  message, so a trajectory splits into segments at user messages; one sample per segment, rendered as the model
  saw it at the segment's last kept call, loss on that segment's assistant outputs only (earlier segments are
  context with their thoughts dropped, exactly as in the real prompt).
- Event compaction (ADK token threshold 14,336, keeps 5 events, summary not recorded in traces) replaces history
  by an unrecorded summary, so in think mode a sample ends at the last call before the first compaction. Compaction
  happens before call j when the previous call's prompt had >= 14,336 tokens.
- nothink mode renders the same calls with all thoughts removed and enable_thinking=False under the target config
  (default submissions/e001n_baseline_nothink_e4b). The no-thought prompts are shorter, so compaction is
  re-derived from the rebuilt token counts, not from the teacher's.
- Loss covers the model's own output tokens: <|channel>thought ...<channel|> (think mode), tool calls, the stop
  token (<|tool_response> after tool calls, <turn|> after text). Tool responses, prompts and the template's
  '\\n' after <turn|> get 0. A text-only answer is kept only as the final turn of the trajectory (after
  submit_patch); malformed outputs (tool-call markup left in text) and text answers that drew a nudge end the
  segment before them. In think mode the final text answer keeps its thought in the target (the template would drop
  it only from later prompts, and there are none).
- At most --max-tokens tokens per sample (default 16,384): calls are dropped from the end until it fits.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import collections
import copy
import csv
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TOKENIZER = REPO / "data/models/gemma-4-e4b-it-qat-q4_0-unquantized"
DEFAULT_TEMPLATE = REPO / "outputs/small_model/w4a16_chat_template.jinja"
TRAIN_REPOS = {"fastapi/fastapi", "Textualize/rich", "psf/requests", "encode/httpx"}
SA_ARMS = ["v001_gemma_sampling", "v001_gemma_sampling_cap14", "v002_guards", "v003_editdeadline"]
COMPACTION_THRESHOLD = 14336
TEACHER_EMPTY_THOUGHT_TOKENS = 4  # '<|channel>', 'thought', '\n', '<channel|>'
SPECIAL_MARKERS = ("<|tool_call>", "<tool_call|>", "<|channel>", "<channel|>", "<|tool_response>",
                   "<tool_response|>", "<|turn>", "<turn|>", "<|tool>", "<tool|>")

log = logging.getLogger("sft_export")


# ----------------------------------------------------------------------------------------------------------------
# Trace parsing
# ----------------------------------------------------------------------------------------------------------------

@dataclass
class UserMsg:
    text: str
    kind: str  # task_prompt | continuation_nudge


@dataclass
class Call:
    author: str
    ts: float
    thought: str | None = None
    texts: list[str] = field(default_factory=list)
    tool_calls: list[tuple[str, dict]] = field(default_factory=list)
    responses: list[tuple[str, Any]] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    step_ids: list[int] = field(default_factory=list)
    count_delta: int | None = None  # completion_tokens minus the token count of the rebuilt raw output

    def garbled_thought(self) -> bool:
        """A degenerate generation: the channel label leaked into the thought ('<junk>thought\\n...') or the
        thought opens with a junk token in another script (the 31B W4A16 does this, e.g. Tamil or Korean words)."""
        if not self.thought:
            return False
        if re.match(r"^.{0,24}?thoughts?\s*\n", self.thought):
            return True
        return ord(self.thought[0]) > 0x2FF

    @property
    def text(self) -> str:
        return "\n".join(t for t in self.texts if t)

    def malformed(self) -> bool:
        return any(m in t for t in self.texts for m in SPECIAL_MARKERS)

    def count_ok(self) -> bool:
        """The rebuilt raw output has the call's completion_tokens (within 2). A call without a thought may also
        be 4 over: the model emitted an empty '<|channel>thought\\n<channel|>' itself (thinking off, continuation)."""
        d = self.count_delta
        return d is None or abs(d) <= 2 or (not self.thought and d == TEACHER_EMPTY_THOUGHT_TOKENS)


def _obs_content(obs: Any) -> str | None:
    if obs is None:
        return None
    if isinstance(obs, str):
        obs = ast.literal_eval(obs)  # the trace stores str(dict); literal_eval only builds literals
    if isinstance(obs, dict):
        return obs.get("content")
    return getattr(obs, "content", None)


def parse_trace(path: Path) -> tuple[str | None, list[Any]]:
    """Return (raw system instruction, items) where items are UserMsg and Call in session order."""
    d = json.loads(Path(path).read_text())
    system = None
    items: list[Any] = []
    cur: Call | None = None
    for s in d["steps"]:
        ex = s.get("extra") or {}
        et = ex.get("event_type")
        if et == "system_instruction":
            system = s.get("message")
            continue
        if et in ("task_prompt", "continuation_nudge"):
            cur = None
            items.append(UserMsg(s.get("message") or "", et))
            continue
        if s["source"] == "agent" and et in ("thinking", "final", "text", None):
            ts = ex.get("timestamp")
            if cur is None or ts != cur.ts or cur.responses:
                cur = Call(author=ex.get("author") or "", ts=ts)
                items.append(cur)
            cur.step_ids.append(s.get("step_id"))
            if et == "thinking":
                cur.thought = (cur.thought + "\n" if cur.thought else "") + (s.get("message") or "")
            elif et in ("final", "text"):
                cur.texts.append(s.get("message") or "")
            for tc in s.get("tool_calls") or []:
                args = tc.get("arguments") or {}
                if isinstance(args, str):
                    args = ast.literal_eval(args)
                cur.tool_calls.append((tc["function_name"], args))
                cur.author = cur.author or (tc.get("extra") or {}).get("author") or ""
            m = s.get("metrics") or {}
            if m.get("prompt_tokens") is not None and cur.prompt_tokens is None:
                cur.prompt_tokens = m.get("prompt_tokens")
                cur.completion_tokens = m.get("completion_tokens")
            if s.get("observation") is not None:
                cur.responses.append((cur.tool_calls[len(cur.responses)][0], _obs_content(s["observation"])))
            continue
        if s["source"] == "system" and s.get("observation") is not None and et is None:
            # extra responses of parallel tool calls
            if cur is not None and len(cur.responses) < len(cur.tool_calls):
                cur.responses.append((cur.tool_calls[len(cur.responses)][0], _obs_content(s["observation"])))
                continue
            raise ValueError(f"{path}: orphan observation at step {s.get('step_id')}")
        if et == "compaction":
            raise ValueError(f"{path}: compaction entry in trace (unexpected)")
        raise ValueError(f"{path}: unhandled step {s.get('step_id')} source={s['source']} type={et}")
    return system, items


def _fmt_arg(v: Any) -> str:
    """format_argument(value, escape_keys=False) of the Gemma 4 chat template."""
    if isinstance(v, str):
        return '<|"|>' + v + '<|"|>'
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, dict):
        return "{" + ",".join(f"{k}:{_fmt_arg(v[k])}" for k in sorted(v)) + "}"
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(_fmt_arg(x) for x in v) + "]"
    return str(v)


def raw_output(call: Call, thought: str | None) -> str:
    """The text the model generated for a call, as vLLM's parsers split it (checked against completion_tokens on
    the E4B capture): <|channel>thought\n{reasoning}<channel|>{text}{tool calls}{stop token}."""
    out = ""
    if thought is not None:
        out += "<|channel>thought\n" + thought + "<channel|>"
    out += call.text
    for name, args in call.tool_calls:
        out += "<|tool_call>call:" + name + "{" + ",".join(f"{k}:{_fmt_arg(args[k])}" for k in sorted(args)) + "}<tool_call|>"
    out += "<|tool_response>" if call.tool_calls else "<turn|>"
    return out


def recover_thought_whitespace(items: list[Any], rnd: "Renderer", stats: collections.Counter) -> None:
    """The trace strips thoughts; vLLM's reasoning field keeps trailing newlines, which the template re-renders.
    Pick the trailing whitespace whose raw-output token count equals the call's completion_tokens. '\\n' and
    '\\n\\n' cost one token each, so a two-newline ending is recorded as one newline (one token's identity differs,
    the count does not); unresolved thoughts get one newline, the common case. Also records count_delta, which
    flags outputs the parsers changed (e.g. tool-call arguments split wrongly)."""
    for it in items:
        if not isinstance(it, Call):
            continue
        if it.completion_tokens is None:
            stats["completion_count_unknown"] += 1
            continue
        if not it.thought:
            it.count_delta = it.completion_tokens - len(rnd.encode(raw_output(it, None)))
            continue
        best = None
        for trail in ("", "\n"):
            d = it.completion_tokens - len(rnd.encode(raw_output(it, it.thought + trail)))
            if d == 0:
                best = (trail, 0)
                break
            if best is None or abs(d) < abs(best[1]):
                best = (trail, d)
        trail, d = best
        if d != 0:
            trail = "\n"
            stats["thought_trail_unresolved"] += 1
        else:
            stats["thought_trail_" + ("none" if not trail else "newline")] += 1
        it.thought = it.thought + trail
        it.count_delta = d


# ----------------------------------------------------------------------------------------------------------------
# ADK request building (real harness code)
# ----------------------------------------------------------------------------------------------------------------

class AgentEnv:
    """A compiled submission plus the ADK machinery to build the exact LiteLLM request for a session state."""

    def __init__(self, submission_dir: Path):
        logging.getLogger("google_adk").setLevel(logging.ERROR)
        from adk_submission import compile_submission
        from adk_submission.discovery import discover_declared_models
        from adk_submission.registry import ModelRegistry
        from adk_submission.resolvers.generation import install_litellm_reasoning_patch
        from google.adk.models.lite_llm import LiteLlm
        from swegemma.config import build_submission_limits
        from swegemma.tools import create_tools

        install_litellm_reasoning_patch()
        self.submission_dir = Path(submission_dir)
        models = ModelRegistry()
        for alias in discover_declared_models(self.submission_dir):
            # Same model string as VllmServer.create_model_registry (openai/<served model path>); the string only
            # matters for LiteLLM's 'gemma4' tool-role check, which these paths never trigger.
            models.register(alias, LiteLlm(model=f"openai//content/models/{alias}",
                                           api_base="http://127.0.0.1:9/v1", api_key="EMPTY"))
        limits, gc = build_submission_limits()
        self.root = compile_submission(submission_dir=self.submission_dir, tool_registry=create_tools(None),
                                       model_registry=models, limits=limits, generation_constraints=gc,
                                       script_timeout=240)
        self.agents = {}
        stack = [self.root]
        while stack:
            a = stack.pop()
            self.agents[a.name] = a
            stack.extend(getattr(a, "sub_agents", None) or [])

    def llm_agent(self, name: str):
        return self.agents[name]

    def enable_thinking(self, agent_name: str) -> bool | None:
        args = getattr(self.llm_agent(agent_name).canonical_model, "_additional_args", {}) or {}
        return ((args.get("extra_body") or {}).get("chat_template_kwargs") or {}).get("enable_thinking")

    async def _request(self, agent_name: str, events: list, state: dict, branch: str | None):
        from google.adk.agents.invocation_context import InvocationContext
        from google.adk.agents.run_config import RunConfig
        from google.adk.models import lite_llm
        from google.adk.models.llm_request import LlmRequest
        from google.adk.sessions import InMemorySessionService

        agent = self.llm_agent(agent_name)
        ss = InMemorySessionService()
        sess = await ss.create_session(app_name="swegemma_eval", user_id="eval_user", state=dict(state))
        for e in events:
            await ss.append_event(sess, copy.deepcopy(e))
        ctx = InvocationContext(session_service=ss, invocation_id=events[-1].invocation_id if events else "e-0",
                                agent=agent, session=sess, run_config=RunConfig(), branch=branch)
        req = LlmRequest()
        async for _ in agent._llm_flow._preprocess_async(ctx, req):
            pass
        model = agent.canonical_model
        model._maybe_append_user_content(req)
        lite_llm._append_fallback_user_content_if_missing(req)
        msgs, tools, _rf, _gp = await lite_llm._get_completion_inputs(req, model.model)
        return msgs, tools

    def request(self, agent_name: str, events: list, state: dict, branch: str | None = None):
        return asyncio.run(self._request(agent_name, events, state, branch))


def calls_to_events(items: list[Any], agent_name_default: str, *, with_thoughts: bool):
    """ADK session events for the items; returns (events, index) where index[i] = number of events before item i."""
    from google.adk.events.event import Event
    from google.genai import types

    events = []
    index = []
    ts = 1_000_000.0
    for n, it in enumerate(items):
        index.append(len(events))
        inv = "e-trace"
        if isinstance(it, UserMsg):
            events.append(Event(author="user", invocation_id=inv, timestamp=ts,
                                content=types.Content(role="user", parts=[types.Part(text=it.text)])))
            ts += 1
            continue
        author = it.author or agent_name_default
        parts = []
        if with_thoughts and it.thought:
            parts.append(types.Part(text=it.thought, thought=True))
        for t in it.texts:
            if t:
                parts.append(types.Part(text=t))
        ids = []
        for j, (name, args) in enumerate(it.tool_calls):
            cid = f"chatcmpl-tool-{n:03d}{j:02d}"
            ids.append(cid)
            parts.append(types.Part(function_call=types.FunctionCall(id=cid, name=name, args=args)))
        if not parts:
            parts.append(types.Part(text=""))
        events.append(Event(author=author, invocation_id=inv, timestamp=ts,
                            content=types.Content(role="model", parts=parts)))
        ts += 1
        if it.responses:
            rparts = []
            for j, (name, content) in enumerate(it.responses):
                resp = json.loads(content) if isinstance(content, str) else content
                if not isinstance(resp, dict):
                    resp = {"result": resp}
                rparts.append(types.Part(function_response=types.FunctionResponse(id=ids[j], name=name,
                                                                                  response=resp)))
            events.append(Event(author=author, invocation_id=inv, timestamp=ts,
                                content=types.Content(role="user", parts=rparts)))
            ts += 1
    index.append(len(events))
    return events, index


def states_per_item(env: "AgentEnv", items: list[Any], base_state: dict) -> list[dict]:
    """Session state before each item: the harness state plus every output_key an agent's final response set.
    ADK (LlmAgent.__maybe_save_output_to_state) stores the joined non-thought text parts of the final response."""
    state = dict(base_state)
    out = []
    for it in items:
        out.append(dict(state))
        if isinstance(it, Call) and not it.tool_calls and it.author in env.agents:
            key = getattr(env.agents[it.author], "output_key", None)
            if key:
                state[key] = "".join(t for t in it.texts if t)
    return out


# ----------------------------------------------------------------------------------------------------------------
# vLLM 0.19.1 emulation and rendering
# ----------------------------------------------------------------------------------------------------------------

def vllm_conversation(messages: list[dict], content_format: str = "openai") -> list[dict]:
    """Mirror vllm.entrypoints.chat_utils.parse_chat_messages for text-only messages."""
    conv = []
    for m in messages:
        m = dict(m)
        role = m["role"]
        content = m.get("content")
        reasoning = m.get("reasoning")
        if content is None:
            parts = []
        elif isinstance(content, str):
            parts = [content]
        else:
            parts = [p.get("text") for p in content if p.get("type") in ("text", "input_text", "output_text",
                                                                          "refusal", "thinking")]
            parts = [p for p in parts if p is not None]
        if content_format == "openai":
            r = {"role": role, "content": [{"type": "text", "text": p} for p in parts]}
        else:
            r = {"role": role, "content": "\n".join(parts)}
        if role == "assistant":
            if m.get("tool_calls") is not None:
                r["tool_calls"] = [json.loads(json.dumps(dict(tc), default=str)) for tc in m["tool_calls"]]
            if reasoning is not None:
                r["reasoning"] = reasoning
                r["reasoning_content"] = reasoning
        elif role == "tool":
            if "tool_call_id" in m:
                r["tool_call_id"] = m["tool_call_id"]
        if isinstance(m.get("name"), str):
            r["name"] = m["name"]
        conv.append(r)
    for r in conv:  # _postprocess_messages
        if r["role"] == "assistant" and "tool_calls" in r:
            if not isinstance(r["tool_calls"], list):
                continue
            if len(r["tool_calls"]) == 0:
                r.pop("tool_calls")
                continue
            for it in r["tool_calls"]:
                a = it["function"].get("arguments")
                if a:
                    if not isinstance(a, (dict, list)):
                        it["function"]["arguments"] = json.loads(a)
                else:
                    it["function"]["arguments"] = {}
    return conv


def vllm_tools(tools: list[dict] | None) -> list[dict] | None:
    if tools is None:
        return None
    return [json.loads(json.dumps(t)) for t in tools]


class Renderer:
    def __init__(self, tokenizer_dir: Path, template_path: Path):
        from transformers import AutoTokenizer
        self.tokenizer_dir = Path(tokenizer_dir)
        self.template_path = Path(template_path)
        self.tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        self.template = Path(template_path).read_text()

    def render(self, conv, tools, *, add_generation_prompt: bool, enable_thinking: bool | None) -> str:
        kw = {}
        if enable_thinking is not None:
            kw["enable_thinking"] = enable_thinking
        return self.tok.apply_chat_template(conv, tools=tools, chat_template=self.template, tokenize=False,
                                            add_generation_prompt=add_generation_prompt,
                                            continue_final_message=False, **kw)

    def encode(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def encode_offsets(self, text: str):
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return enc["input_ids"], enc["offset_mapping"]


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------------------------------------------
# Sample construction
# ----------------------------------------------------------------------------------------------------------------

@dataclass
class Turn:
    item: int          # index into items
    prompt: str        # rendered prompt at this call
    prompt_ids: list[int]
    trace_tokens: int | None
    conv: list[dict]   # vLLM conversation of the prompt
    tools: list[dict] | None


def build_trajectory(env: AgentEnv, rnd: Renderer, items: list[Any], state: dict, *, mode: str,
                     agent_name: str, max_tokens: int, stats: collections.Counter,
                     ignore_compaction: bool = False, min_loss_turns: int = 1):
    """Return a list of samples (dicts with text, spans, meta) for one single-agent trajectory.

    ignore_compaction: render the full history as if no compaction had happened (exact up to the teacher's first
    compaction, an approximation after it); samples are then limited by max_tokens only."""
    think = mode == "think"
    enable_thinking = env.enable_thinking(agent_name)
    events, index = calls_to_events(items, agent_name, with_thoughts=think)
    call_ix = [i for i, it in enumerate(items) if isinstance(it, Call)]
    turns: dict[int, Turn] = {}
    cut_reason = "end"
    prev_tokens = None
    teacher_prev = None   # the teacher's previous prompt count: decides whether the teacher's context was compacted
    teacher_compacted = False
    for i in call_ix:
        it = items[i]
        if teacher_prev is not None and teacher_prev >= COMPACTION_THRESHOLD:
            teacher_compacted = True
        if prev_tokens is not None and prev_tokens >= COMPACTION_THRESHOLD and not ignore_compaction:
            cut_reason = "compaction"
            break
        msgs, tools = env.request(agent_name, events[: index[i]], state)
        conv = vllm_conversation(msgs)
        prompt = rnd.render(conv, vllm_tools(tools), add_generation_prompt=True, enable_thinking=enable_thinking)
        ids = rnd.encode(prompt)
        if len(ids) >= max_tokens:
            cut_reason = "max_tokens"
            break
        if think and it.prompt_tokens is not None and not teacher_compacted:
            if len(ids) != it.prompt_tokens:
                stats["prompt_mismatch_calls"] += 1
                cut_reason = f"mismatch({len(ids)}!={it.prompt_tokens})"
                break
            stats["prompt_match_calls"] += 1
        elif think and teacher_compacted:
            stats["unverifiable_calls_after_teacher_compaction"] += 1
        turns[i] = Turn(i, prompt, ids, it.prompt_tokens, conv, vllm_tools(tools))
        # the teacher's own count decides compaction in think mode; our count in nothink mode
        prev_tokens = it.prompt_tokens if (think and it.prompt_tokens is not None and not teacher_compacted) else len(ids)
        if it.prompt_tokens is not None:
            teacher_prev = it.prompt_tokens
    kept = sorted(turns)
    if not kept:
        return [], cut_reason

    # segments: maximal runs of calls with no user message between them (think mode: a user message makes the
    # template drop all earlier thoughts). Without thoughts nothing is dropped, so nothink keeps one segment.
    segments: list[list[int]] = []
    if not think:
        segments = [kept]
    else:
        last_user = None
        for i in kept:
            lu = max((j for j in range(i) if isinstance(items[j], UserMsg)), default=None)
            if not segments or lu != last_user:
                segments.append([])
                last_user = lu
            segments[-1].append(i)
    # first call the teacher saw after its own first compaction (its context differed from what we render)
    teacher_compacted_from = None
    prev = None
    for i in call_ix:
        if prev is not None and prev >= COMPACTION_THRESHOLD:
            teacher_compacted_from = i
            break
        if items[i].prompt_tokens is not None:
            prev = items[i].prompt_tokens

    is_last_call_of_traj = call_ix[-1]

    def valid(g: int) -> bool:
        """Turns that get loss: well-formed output the parsers did not change, ending in a tool call (or the
        trajectory's final answer)."""
        it = items[g]
        if it.malformed() or (think and it.garbled_thought()):
            return False
        if it.count_delta is not None and abs(it.count_delta) > 2:
            return False
        return bool(it.tool_calls) or g == is_last_call_of_traj

    samples = []
    for sidx, seg in enumerate(segments):
        # trim invalid tail turns: malformed output, or a text answer that is not the end of the trajectory
        good = list(seg)
        while good and not valid(good[-1]):
            it = items[good[-1]]
            reason = ("malformed_text" if it.malformed() else "garbled_thought" if it.garbled_thought()
                      else "count_mismatch" if it.count_delta is not None and abs(it.count_delta) > 2
                      else "text_before_nudge")
            stats["dropped_last_turn_" + reason] += 1
            good.pop()
        for g in good[:-1]:
            if not valid(g):
                it = items[g]
                stats["context_only_turn_" + ("malformed_text" if it.malformed() else "garbled_thought"
                                             if think and it.garbled_thought() else "count_mismatch"
                                             if it.tool_calls else "text_before_nudge")] += 1
        while good:
            loss_turns = [g for g in good if valid(g)]
            sample = render_sample(env, rnd, items, events, turns, good, loss_turns, mode=mode,
                                   enable_thinking=enable_thinking, stats=stats)
            if sample is None:
                stats["render_failures"] += 1
                good.pop()
                continue
            if len(sample["input_ids"]) > max_tokens:
                stats["trimmed_for_length_turns"] += 1
                good.pop()
                continue
            if sample["n_turns"] < min_loss_turns:
                stats["dropped_samples_below_min_loss_turns"] += 1
                break
            sample["segment"] = sidx
            sample["cut_reason"] = cut_reason if sidx == len(segments) - 1 else "nudge"
            sample["turns_after_teacher_compaction"] = (
                0 if teacher_compacted_from is None
                else sum(1 for g in good if g >= teacher_compacted_from and valid(g)))
            samples.append(sample)
            break
    return samples, cut_reason


def build_agent_sequences(env: AgentEnv, rnd: Renderer, items: list[Any], base: dict, *, max_tokens: int,
                          stats: collections.Counter, min_loss_turns: int = 1):
    """Per-agent samples for a (multi-agent) trajectory, each agent rendered with its own config and context.

    Every call of every agent is rebuilt through ADK (include_contents, other-agent 'For context' messages,
    output_key state) and checked against the trace's prompt_tokens; an agent's sequence stops at its first
    mismatch (compaction or anything else). A segment is a run of one agent's calls whose prompts extend each
    other (prefix property); it breaks where the template drops thoughts or the agent's context restarts."""
    states = states_per_item(env, items, base)
    events, index = calls_to_events(items, env.root.name, with_thoughts=True)
    by_agent: dict[str, list[int]] = collections.defaultdict(list)
    for i, it in enumerate(items):
        if isinstance(it, Call):
            by_agent[it.author].append(i)
    out = []
    for agent_name, calls in by_agent.items():
        if agent_name not in env.agents:
            stats["unknown_agent_calls"] += len(calls)
            continue
        et = env.enable_thinking(agent_name)
        mode = "think" if et else "nothink"
        turns: dict[int, Turn] = {}
        cut_reason = "end"
        for i in calls:
            it = items[i]
            msgs, tools = env.request(agent_name, events[: index[i]], states[i])
            conv = vllm_conversation(msgs)
            prompt = rnd.render(conv, vllm_tools(tools), add_generation_prompt=True, enable_thinking=et)
            ids = rnd.encode(prompt)
            # The 31B's served template (same as data/models/gemma-4-31b-it-4bit-mlx/chat_template.jinja) appends an
            # empty '<|channel>thought\n<channel|>' (4 tokens) to a fresh generation prompt when thinking is off;
            # the E4B template, which these samples are rendered with, does not.
            expect = len(ids) + (TEACHER_EMPTY_THOUGHT_TOKENS if (not et and prompt.endswith("<|turn>model\n")) else 0)
            if it.prompt_tokens is not None and expect != it.prompt_tokens:
                stats["prompt_mismatch_calls"] += 1
                cut_reason = f"mismatch({expect}!={it.prompt_tokens})"
                break
            if it.prompt_tokens is not None:
                stats["prompt_match_calls"] += 1
            if len(ids) >= max_tokens:
                cut_reason = "max_tokens"
                break
            turns[i] = Turn(i, prompt, ids, it.prompt_tokens, conv, vllm_tools(tools))
        kept = [i for i in calls if i in turns]
        segments: list[list[int]] = []
        for i in kept:
            if segments and turns[i].prompt.startswith(turns[segments[-1][-1]].prompt):
                segments[-1].append(i)
            else:
                segments.append([i])

        def next_item_is_nudge(g: int) -> bool:
            return g + 1 < len(items) and isinstance(items[g + 1], UserMsg) and items[g + 1].kind == "continuation_nudge"

        def valid(g: int, seg_last: int) -> bool:
            it = items[g]
            if it.malformed() or (et and it.garbled_thought()):
                return False
            if not it.count_ok():
                return False
            if it.tool_calls:
                return True
            if next_item_is_nudge(g):
                return False  # the answer that ended the turn without finishing
            # a hand-off answer; inside a segment only without thinking (later prompts drop a text turn's thought)
            return g == seg_last or not et

        for sidx, seg in enumerate(segments):
            good = list(seg)
            while good and not valid(good[-1], good[-1]):
                it = items[good[-1]]
                reason = ("malformed_text" if it.malformed() else "garbled_thought" if et and it.garbled_thought()
                          else "count_mismatch" if not it.count_ok() else "text_before_nudge")
                stats[f"{agent_name}_dropped_last_turn_{reason}"] += 1
                good.pop()
            for g in good[:-1]:
                if not valid(g, good[-1]):
                    it = items[g]
                    stats[f"{agent_name}_context_only_turn_" + (
                        "malformed_text" if it.malformed() else "garbled_thought" if et and it.garbled_thought()
                        else "count_mismatch" if not it.count_ok() else "text")] += 1
            while good:
                loss_turns = [g for g in good if valid(g, good[-1])]
                sample = render_sample(env, rnd, items, events, turns, good, loss_turns, mode=mode,
                                       enable_thinking=et, stats=stats)
                if sample is None:
                    stats["render_failures"] += 1
                    good.pop()
                    continue
                if len(sample["input_ids"]) > max_tokens:
                    stats["trimmed_for_length_turns"] += 1
                    good.pop()
                    continue
                if sample["n_turns"] < min_loss_turns:
                    stats["dropped_samples_below_min_loss_turns"] += 1
                    break
                sample["agent"] = agent_name
                sample["segment"] = sidx
                sample["cut_reason"] = cut_reason if sidx == len(segments) - 1 else "segment"
                sample["turns_after_teacher_compaction"] = 0
                out.append(sample)
                break
    return out


def synthetic_summary(items: list[Any], upto: int, repo: str) -> str:
    """Deterministic stand-in for the ADK compaction summary (LlmEventSummarizer), built from the trace: the issue's
    first 300 characters, files read, files edited, commands run (last 8) and calls used. Not LLM-written. Sized to
    the real 31B summaries, whose length the prompt-token deltas give (about 270-410 tokens)."""
    task = next((it.text for it in items[:upto] if isinstance(it, UserMsg) and it.kind == "task_prompt"), "")
    issue = task.split("Problem Statement:\n", 1)[-1].split("\n## ", 1)[0].strip()
    issue = " ".join(issue.split())
    if len(issue) > 300:
        issue = issue[:300].rsplit(" ", 1)[0] + " ..."
    reads, edits, cmds = [], [], []
    n_calls = 0
    submitted = False
    for it in items[:upto]:
        if not isinstance(it, Call):
            continue
        for j, (name, args) in enumerate(it.tool_calls):
            n_calls += 1
            resp = it.responses[j][1] if j < len(it.responses) else None
            try:
                r = json.loads(resp) if isinstance(resp, str) else (resp or {})
            except ValueError:
                r = {}
            status = r.get("status") if isinstance(r, dict) else None
            if name == "read_file":
                rng = ""
                if args.get("start_line") or args.get("end_line"):
                    rng = f" lines {args.get('start_line', 1)}-{args.get('end_line', 'end')}"
                entry = f"{args.get('filepath', '?')}{rng}"
                if entry not in reads:
                    reads.append(entry)
            elif name in ("edit_file", "write_file"):
                entry = f"{args.get('filepath', '?')} ({name}, {status or 'unknown'})"
                edits.append(entry)
            elif name == "run_command":
                c = " ".join(str(args.get("command", "")).split())
                if len(c) > 80:
                    c = c[:80] + " ..."
                code = r.get("exit_code", (r.get("details") or {}).get("exit_code") if isinstance(r.get("details"), dict) else None) if isinstance(r, dict) else None
                cmds.append(f"`{c}` (exit {code})" if code is not None else f"`{c}` ({status})")
            elif name == "submit_patch":
                submitted = True
    lines = [f"Summary of the work so far on the {repo} issue.", f"Issue: {issue}"]
    lines.append("Files read: " + ("; ".join(reads) if reads else "none") + ".")
    lines.append("Files edited: " + ("; ".join(edits) if edits else "none yet") + ".")
    if cmds:
        lines.append(f"Commands run ({len(cmds)}, last {min(len(cmds), 8)} shown): " + "; ".join(cmds[-8:]) + ".")
    lines.append(f"Tool calls used: {n_calls}." + (" A patch has been submitted." if submitted else " No patch submitted yet."))
    return "\n".join(lines)


def build_tail_samples(env: AgentEnv, rnd: Renderer, items: list[Any], base: dict, repo: str, *, max_tokens: int,
                       stats: collections.Counter, min_loss_turns: int = 1):
    """Samples for the turns after each teacher compaction, in the context ADK compaction leaves.

    At each compaction point (the call after one whose prompt had >= 14,336 tokens) a compaction event is added to
    the session exactly as LlmEventSummarizer builds it, over the events ADK's own
    _events_to_compact_for_token_threshold selects (everything but the last 5 events, moved earlier so no tool
    response loses its call; a later compaction rolls the previous summary in). Its compacted_content is a
    synthetic summary (synthetic_summary). ADK's contents processor then does the rest: the summary is materialised
    as a model-authored event at the end of the compacted range and, its author differing from the agent's name,
    presented as a user message 'For context:[model] said: <summary>'; the task prompt is compacted away.
    Check: within one compaction segment the trace's prompt_tokens minus ours must be constant (the real
    summary's length minus ours); the sample is dropped otherwise."""
    from google.adk.apps.compaction import _events_to_compact_for_token_threshold
    from google.adk.events.event import Event
    from google.adk.events.event_actions import EventActions, EventCompaction
    from google.genai import types

    agent_name = env.root.name
    et = env.enable_thinking(agent_name)
    events, index = calls_to_events(items, agent_name, with_thoughts=True)
    call_ix = [i for i, it in enumerate(items) if isinstance(it, Call)]
    points = []
    prev = None
    for i in call_ix:
        if prev is not None and prev >= COMPACTION_THRESHOLD:
            points.append(i)
        if items[i].prompt_tokens is not None:
            prev = items[i].prompt_tokens
    if not points:
        return []
    comp_events: list[tuple[int, Any]] = []  # (insert before events[pos], event)

    def session(upto: int) -> list:
        out, ci = [], 0
        for n, e in enumerate(events[:upto]):
            while ci < len(comp_events) and comp_events[ci][0] == n:
                out.append(comp_events[ci][1])
                ci += 1
            out.append(e)
        while ci < len(comp_events) and comp_events[ci][0] <= upto:
            out.append(comp_events[ci][1])
            ci += 1
        return out

    is_last_call = call_ix[-1]
    samples = []
    for pi, j in enumerate(points):
        sess = session(index[j])
        to_compact = _events_to_compact_for_token_threshold(events=sess, event_retention_size=5)
        if not to_compact:
            stats["tail_nothing_to_compact"] += 1
            continue
        text = synthetic_summary(items, j, repo)
        ev = Event(author="user", invocation_id=f"e-compaction-{pi}", timestamp=events[index[j] - 1].timestamp + 0.5,
                   actions=EventActions(compaction=EventCompaction(
                       start_timestamp=to_compact[0].timestamp, end_timestamp=to_compact[-1].timestamp,
                       compacted_content=types.Content(role="model", parts=[types.Part(text=text)]))))
        comp_events.append((index[j], ev))
        nxt = points[pi + 1] if pi + 1 < len(points) else None
        seg_calls = []
        for i in call_ix:
            if i < j or (nxt is not None and i >= nxt):
                continue
            if i > j and any(isinstance(items[u], UserMsg) for u in range(j, i)):
                break  # a nudge drops the thoughts: end the tail segment there
            seg_calls.append(i)
        turns: dict[int, Turn] = {}
        deltas = []
        cut = "end" if nxt is None else "next_compaction"
        for i in seg_calls:
            msgs, tools = env.request(agent_name, session(index[i]), base)
            conv = vllm_conversation(msgs)
            prompt = rnd.render(conv, vllm_tools(tools), add_generation_prompt=True, enable_thinking=et)
            ids = rnd.encode(prompt)
            if len(ids) >= max_tokens:
                cut = "max_tokens"
                break
            if items[i].prompt_tokens is not None:
                deltas.append(items[i].prompt_tokens - len(ids))
            turns[i] = Turn(i, prompt, ids, items[i].prompt_tokens, conv, vllm_tools(tools))
        if not turns:
            continue
        if len(set(deltas)) > 1:
            stats["tail_inconsistent_delta"] += 1
            # keep the calls up to the first change of delta (structure verified up to there)
            first = deltas[0]
            keep = []
            for i in sorted(turns):
                if turns[i].trace_tokens is not None and turns[i].trace_tokens - len(turns[i].prompt_ids) != first:
                    break
                keep.append(i)
            turns = {i: turns[i] for i in keep}
            cut = "delta_change"
            if not turns:
                continue
        if deltas:
            stats["tail_real_summary_minus_synthetic_tokens_sum"] += deltas[0]
            stats["tail_segments_with_delta"] += 1
        good = sorted(turns)

        def valid(g: int) -> bool:
            it = items[g]
            if it.malformed() or it.garbled_thought() or not it.count_ok():
                return False
            return bool(it.tool_calls) or g == is_last_call

        while good and not valid(good[-1]):
            stats["tail_dropped_last_turn"] += 1
            good.pop()
        while good:
            loss_turns = [g for g in good if valid(g)]
            sample = render_sample(env, rnd, items, events_for_render(session, index, good[-1]), turns, good,
                                   loss_turns, mode="think" if et else "nothink", enable_thinking=et, stats=stats,
                                   model_event=events[index[good[-1]]])
            if sample is None:
                stats["render_failures"] += 1
                good.pop()
                continue
            if len(sample["input_ids"]) > max_tokens:
                stats["trimmed_for_length_turns"] += 1
                good.pop()
                continue
            if sample["n_turns"] < min_loss_turns:
                stats["dropped_samples_below_min_loss_turns"] += 1
                break
            sample["segment"] = pi + 1
            sample["cut_reason"] = cut
            sample["turns_after_teacher_compaction"] = sample["n_turns"]
            sample["summary_delta_tokens"] = deltas[0] if deltas else None
            sample["synthetic_summary"] = text
            samples.append(sample)
            break
    return samples


def events_for_render(session_fn, index, last):
    return session_fn(index[last] + 1)


def assistant_message(content, model: str) -> dict:
    """The OpenAI assistant message LiteLLM sends for a model event (patched _content_to_message_param)."""
    from google.adk.models import lite_llm
    msg = asyncio.run(lite_llm._content_to_message_param(copy.deepcopy(content), provider="openai", model=model))
    from adk_submission.resolvers.generation import _normalize_reasoning_message
    return _normalize_reasoning_message(dict(msg))


def render_sample(env, rnd, items, events, turns, good, loss_turns, *, mode, enable_thinking, stats,
                  model_event=None):
    """Render prompt-at-last-call + last output; loss on the outputs of loss_turns."""
    think = mode == "think"
    last = good[-1]
    it = items[last]
    t_last = turns[last]
    prompt_last = t_last.prompt
    if it.tool_calls:
        if model_event is None:
            model_event = events[env_index(events, last, items)]
        msg = assistant_message(model_event.content, env.llm_agent(model_event.author).canonical_model.model)
        conv = t_last.conv + vllm_conversation([msg])
        full = rnd.render(conv, t_last.tools, add_generation_prompt=False, enable_thinking=enable_thinking)
        if not full.startswith(prompt_last):
            log.warning("final render is not an extension of the prompt (item %d)", last)
            return None
        text = full
        if not text.endswith("<|tool_response>"):
            log.warning("tool-call turn does not end with the stop token (item %d): %r", last, text[-60:])
            return None
        if it.text.strip():
            stats["final_turn_with_text_and_toolcalls"] += 1
    else:
        out = ""
        if think and it.thought:
            out += "<|channel>thought\n" + it.thought + "\n<channel|>"
        out += it.text.strip() + "<turn|>"
        text = prompt_last + out
    spans = []
    for g in good:
        p = turns[g].prompt
        if not text.startswith(p):
            log.warning("prompt of item %d is not a prefix of the sample", g)
            return None
        if g not in loss_turns:
            continue
        start = len(p)
        if g == last:
            end = len(text)
        elif not items[g].tool_calls:
            k = text.find("<turn|>", start)  # an inner text answer (multi-agent hand-off) ends with <turn|>
            if k < 0:
                return None
            end = k + len("<turn|>")
        else:
            k = text.find("<|tool_response>", start)
            if k < 0:
                return None
            end = k + len("<|tool_response>")
            if items[g].text.strip():
                stats["inner_turns_with_text_and_toolcalls"] += 1
        spans.append((start, end))
    ids, offs = rnd.encode_offsets(text)
    mask = [0] * len(ids)
    for (s, e) in spans:
        for t, (a, b) in enumerate(offs):
            if a >= s and b <= e and b > a:
                mask[t] = 1
            elif a < s < b or a < e < b:
                log.warning("token straddles a loss boundary (item span %d-%d, token %d-%d)", s, e, a, b)
                return None
    # the context tokens before each loss span must be exactly the tokens vLLM fed for that call
    for g in good:
        pid = turns[g].prompt_ids
        if ids[: len(pid)] != pid:
            log.warning("prompt tokens of item %d are not a prefix of the sample tokens", g)
            return None
    return {"text": text, "input_ids": ids, "loss_mask": mask, "n_turns": len(loss_turns), "spans": spans}


def env_index(events, item_i, items) -> int:
    """Index of the model event of items[item_i] in the event list built by calls_to_events."""
    n = 0
    for j, it in enumerate(items):
        if j == item_i:
            return n
        n += 1
        if isinstance(it, Call) and it.responses:
            n += 1
    raise IndexError(item_i)


# ----------------------------------------------------------------------------------------------------------------
# Dataset build
# ----------------------------------------------------------------------------------------------------------------

def base_state(task: dict) -> dict:
    """Session state the harness creates (agent_runner.py): problem_description, and hints when non-empty."""
    st = {"problem_description": task.get("problem_statement", "")}
    hints = task.get("hints_text") or ""
    if isinstance(hints, str) and hints.strip():
        st["hints"] = hints.strip()
    return st


def load_tasks() -> dict[str, dict]:
    out = {}
    with open(REPO / "data/raw/tasks.jsonl") as f:
        for line in f:
            t = json.loads(line)
            out[t["instance_id"]] = t
    return out


def split_tasks(task_ids: list[str], repo_of: dict[str, str], val_frac: float, seed: int) -> set[str]:
    by_repo = collections.defaultdict(list)
    for t in sorted(task_ids):
        by_repo[repo_of[t]].append(t)
    val = set()
    for repo, ts in sorted(by_repo.items()):
        ts = sorted(ts, key=lambda x: hashlib.sha1(f"{seed}:{x}".encode()).hexdigest())
        n = int(round(len(ts) * val_frac))
        val.update(ts[:n])
    return val


def arm_submission(arm: str) -> Path:
    base = arm.split(" ")[0]
    if "_cap" in base:
        base = base[: base.rindex("_cap")]
    return REPO / "submissions" / base


def cmd_build(a):
    random.seed(a.seed)
    tasks = load_tasks()
    rows = list(csv.DictReader(open(REPO / "outputs/task_ledger/runs.csv")))
    arms = a.arms.split(",")
    cand = collections.defaultdict(list)
    skipped = collections.Counter()
    for r in rows:
        if r["arm"] not in arms or r["resolved"] != "1":
            continue
        t = tasks.get(r["instance_id"])
        if t is None or t["repo"] not in TRAIN_REPOS:
            skipped["ood_or_unknown_repo"] += 1
            continue
        cand[r["instance_id"]].append(r)
    repo_of = {i: tasks[i]["repo"] for i in cand}
    if a.val_tasks_from:
        prev = json.loads((REPO / "data/sft" / a.val_tasks_from / "meta.json").read_text())
        val = set(prev["split"]["val_tasks"]) & set(cand)
    else:
        val = split_tasks(list(cand), repo_of, a.val_frac, a.seed)
    out_dir = REPO / "data/sft" / a.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in ("READY", "train.jsonl", "val.jsonl", "index.jsonl", "meta.json"):
        if (out_dir / p).exists():
            (out_dir / p).unlink()
    rnd = Renderer(a.tokenizer, a.template)
    envs: dict[str, AgentEnv] = {}
    stats = collections.Counter()
    t0 = time.time()
    fout = {"train": open(out_dir / "train.jsonl", "w"), "val": open(out_dir / "val.jsonl", "w")}
    findex = open(out_dir / "index.jsonl", "w")
    counts = collections.Counter()
    agent_counts = collections.Counter()
    tok_counts = collections.Counter()
    loss_counts = collections.Counter()
    lengths = []
    per_traj = []
    dropped = []
    if a.trajectories_from:
        prev = json.loads((REPO / "data/sft" / a.trajectories_from / "meta.json").read_text())
        keep = {(t["run"], t["instance_id"]) for t in prev["trajectories"]
                if not a.trajectories_cut or t["cut"].split("(")[0] == a.trajectories_cut}
        for iid in list(cand):
            cand[iid] = [r for r in cand[iid] if (r["run"], iid) in keep]
            if not cand[iid]:
                del cand[iid]
        skipped["not_in_trajectories_from"] = sum(1 for r in rows if r["arm"] in arms and r["resolved"] == "1") - \
            sum(len(v) for v in cand.values())
    task_list = sorted(cand)
    if a.limit_tasks:
        task_list = task_list[: a.limit_tasks]
    for n_task, iid in enumerate(task_list):
        runs = sorted(cand[iid], key=lambda r: hashlib.sha1(f"{a.seed}:{r['run']}:{iid}".encode()).hexdigest())
        used = 0
        for r in runs:
            if used >= a.max_runs_per_task:
                skipped["over_max_runs_per_task"] += 1
                continue
            sub = arm_submission(r["arm"]) if a.mode == "think" else Path(a.target_submission)
            key = str(sub)
            if key not in envs:
                envs[key] = AgentEnv(sub)
            env = envs[key]
            trace = REPO / r["run"] / "traces" / f"trace_{iid}.json"
            try:
                system, items = parse_trace(trace)
            except Exception as e:  # noqa: BLE001
                dropped.append({"run": r["run"], "instance_id": iid, "reason": f"parse: {e}"})
                stats["parse_errors"] += 1
                continue
            agent_name = env.root.name
            state = {"problem_description": tasks[iid]["problem_statement"]}
            if (tasks[iid].get("hints_text") or "").strip():
                state["hints"] = tasks[iid]["hints_text"].strip()
            if a.mode == "think" and system is not None and not a.per_agent:
                ours = env.root.instruction.strip() if isinstance(env.root.instruction, str) else None
                if ours is not None and ours != system.strip():
                    dropped.append({"run": r["run"], "instance_id": iid, "reason": "system prompt differs from config"})
                    stats["system_prompt_mismatch"] += 1
                    continue
            recover_thought_whitespace(items, rnd, stats)
            if a.per_agent:
                samples = build_agent_sequences(env, rnd, items, base_state(tasks[iid]), max_tokens=a.max_tokens,
                                                stats=stats, min_loss_turns=a.min_loss_turns)
                cut = ",".join(sorted({x["cut_reason"].split("(")[0] for x in samples})) or "none"
            elif a.tail:
                samples = build_tail_samples(env, rnd, items, base_state(tasks[iid]), repo_of[iid],
                                             max_tokens=a.max_tokens, stats=stats, min_loss_turns=a.min_loss_turns)
                if not a.all_tails:
                    samples = [x for x in samples if x["segment"] == 1]
                cut = ",".join(sorted({x["cut_reason"] for x in samples})) or "none"
            else:
                samples, cut = build_trajectory(env, rnd, items, state, mode=a.mode, agent_name=agent_name,
                                                max_tokens=a.max_tokens, stats=stats,
                                                ignore_compaction=a.ignore_compaction,
                                                min_loss_turns=a.min_loss_turns)
            n_calls = sum(1 for x in items if isinstance(x, Call))
            per_traj.append({"run": r["run"], "instance_id": iid, "n_calls": n_calls, "cut": cut,
                             "n_samples": len(samples), "turns": sum(s["n_turns"] for s in samples)})
            if not samples:
                dropped.append({"run": r["run"], "instance_id": iid, "reason": f"no sample ({cut})"})
                stats["trajectories_without_sample"] += 1
                continue
            used += 1
            split = "val" if iid in val else "train"
            run_short = r["run"].replace("results/", "").replace("/results/", "/")
            for s in samples:
                ag = s.get("agent", agent_name)
                sid = f"{r['arm']}|{run_short}|{iid}|{ag}|s{s['segment']}" if a.per_agent else \
                    f"{r['arm']}|{run_short}|{iid}|s{s['segment']}"
                rec = {"id": sid, "instance_id": iid, "repo": repo_of[iid], "arm": r["arm"], "agent": ag,
                       "n_tokens": len(s["input_ids"]), "input_ids": s["input_ids"], "loss_mask": s["loss_mask"]}
                fout[split].write(json.dumps(rec) + "\n")
                findex.write(json.dumps({"id": sid, "split": split, "run": r["run"], "segment": s["segment"],
                                         "n_turns": s["n_turns"], "n_tokens": len(s["input_ids"]),
                                         "n_loss_tokens": sum(s["loss_mask"]), "cut_reason": s["cut_reason"],
                                         "n_calls_in_trajectory": n_calls,
                                         "turns_after_teacher_compaction": s["turns_after_teacher_compaction"],
                                         **({"summary_delta_tokens": s.get("summary_delta_tokens"),
                                             "synthetic_summary": s.get("synthetic_summary")} if a.tail else {})})
                             + "\n")
                counts[(split, repo_of[iid], r["arm"])] += 1
                agent_counts[(split, ag)] += 1
                tok_counts[split] += len(s["input_ids"])
                loss_counts[split] += sum(s["loss_mask"])
                lengths.append(len(s["input_ids"]))
        if (n_task + 1) % 10 == 0:
            log.warning("%d/%d tasks, %.0f s", n_task + 1, len(cand), time.time() - t0)
    for f in fout.values():
        f.close()
    findex.close()
    lengths.sort()
    pct = (lambda q: lengths[min(len(lengths) - 1, int(q * len(lengths)))] if lengths else 0)
    meta = {
        "dataset": a.dataset, "mode": a.mode, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": "scripts/sft_export.py", "args": vars(a) | {"tokenizer": str(a.tokenizer), "template": str(a.template)},
        "tokenizer": {"path": str(Path(a.tokenizer).relative_to(REPO)),
                      "tokenizer.json_sha256": sha256(Path(a.tokenizer) / "tokenizer.json"),
                      "tokenizer_config.json_sha256": sha256(Path(a.tokenizer) / "tokenizer_config.json")},
        "chat_template": {"path": str(Path(a.template).relative_to(REPO)), "sha256": sha256(a.template),
                          "note": "byte-identical to the E4B QAT q4_0-unquantized chat_template.jinja and the served W4A16 one"},
        "rendering": {
            "content_format": "openai (vLLM auto-detection on this template)",
            "messages": "ADK flow preprocessing + adk_submission LiteLLM patch + LiteLlm._get_completion_inputs",
            "tool_role": "tool (the model path has no 'gemma4' substring)",
            "enable_thinking": "from the config: thinking_budget>0 -> True, 0 -> False",
            "add_generation_prompt": True, "tokenize": "tokenizer(text, add_special_tokens=False)",
            "compaction_threshold": COMPACTION_THRESHOLD,
            "compaction": ("IGNORED: full history rendered as if no compaction had happened; exact up to the "
                           "teacher's first compaction, approximate after it (the teacher saw an unrecorded summary "
                           "plus the last 5 events); samples cut at max_tokens"
                           if a.ignore_compaction else
                           "samples end before the first call whose context the scorer would compact"),
            "approx_context_after_teacher_compaction": bool(a.ignore_compaction or a.mode == "nothink" or a.tail),
            "synthetic_compaction_summary": (
                "YES: each sample's context is what ADK compaction leaves (system, 'For context:[model] said: "
                "<summary>' user message, the retained events), but <summary> is synthetic_summary() (issue head, "
                "files read/edited, commands, calls used), not the LLM summary the teacher saw; it is in index.jsonl"
                if a.tail else "no"),
            "min_loss_turns": a.min_loss_turns,
            "loss": "assistant outputs incl. the stop token (<|tool_response> after tool calls, <turn|> after text)",
        },
        "split": {"rule": (f"val tasks of data/sft/{a.val_tasks_from} present here" if a.val_tasks_from else
                           f"by task, per repo, sha1('{a.seed}:'+instance_id) order, val_frac {a.val_frac}"),
                  "val_tasks": sorted(val), "train_tasks": sorted(set(cand) - val)},
        "counts": {"samples": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in sorted(counts.items())},
                   "samples_by_agent": {f"{k[0]}|{k[1]}": v for k, v in sorted(agent_counts.items())},
                   "samples_train": sum(v for k, v in counts.items() if k[0] == "train"),
                   "samples_val": sum(v for k, v in counts.items() if k[0] == "val"),
                   "tokens": dict(tok_counts), "loss_tokens": dict(loss_counts),
                   "length_p50": pct(0.5), "length_p90": pct(0.9), "length_max": lengths[-1] if lengths else 0},
        "stats": dict(stats), "skipped": dict(skipped), "dropped": dropped, "trajectories": per_traj,
        "seconds": round(time.time() - t0, 1),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta["counts"][k] for k in ("samples_train", "samples_val", "tokens", "loss_tokens",
                                                     "length_p50", "length_p90", "length_max")}))
    print(json.dumps(dict(stats)), json.dumps(dict(skipped)), "dropped", len(dropped))


# ----------------------------------------------------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------------------------------------------------

def spans_of(mask: list[int]) -> list[tuple[int, int]]:
    out, s = [], None
    for i, m in enumerate(mask + [0]):
        if m and s is None:
            s = i
        elif not m and s is not None:
            out.append((s, i))
            s = None
    return out


def cmd_verify(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(a.tokenizer))
    d = REPO / "data/sft" / a.dataset
    problems = collections.Counter()
    n = 0
    over = 0
    shown = 0
    for split in ("train", "val"):
        with open(d / f"{split}.jsonl") as f:
            for line in f:
                rec = json.loads(line)
                n += 1
                ids, mask = rec["input_ids"], rec["loss_mask"]
                if len(ids) != len(mask) or len(ids) != rec["n_tokens"]:
                    problems["length_field"] += 1
                if len(ids) > a.max_tokens:
                    over += 1
                if set(mask) - {0, 1}:
                    problems["mask_values"] += 1
                sp = spans_of(mask)
                if not sp:
                    problems["no_loss"] += 1
                for (s, e) in sp:
                    piece = tok.decode(ids[s:e], skip_special_tokens=False)
                    before = tok.decode(ids[max(0, s - 3):s], skip_special_tokens=False)
                    ok_start = (piece.startswith(("<|channel>", "<|tool_call>")) or before.endswith("model\n")
                                or (before.endswith("<tool_response|>") and piece.endswith("<turn|>")))
                    ok_end = piece.endswith(("<|tool_response>", "<turn|>"))
                    bad_inside = any(m in piece for m in ("<|turn>", "<tool_response|>", "response:"))
                    if not ok_start:
                        problems["span_start"] += 1
                    if not ok_end:
                        problems["span_end"] += 1
                    if bad_inside and "response:" in piece and "<tool_response|>" in piece:
                        problems["span_contains_tool_output"] += 1
                    if "<|turn>" in piece:
                        problems["span_contains_turn_start"] += 1
                if shown < a.show and (a.sample_id is None or rec["id"] == a.sample_id):
                    shown += 1
                    print("=" * 100)
                    print(rec["id"], "tokens", len(ids), "loss tokens", sum(mask), "spans", len(sp))
                    for k, (s, e) in enumerate(sp):
                        piece = tok.decode(ids[s:e], skip_special_tokens=False)
                        ctx = tok.decode(ids[max(0, s - 12):s], skip_special_tokens=False)
                        nxt = tok.decode(ids[e:e + 12], skip_special_tokens=False)
                        if k < a.spans or k == len(sp) - 1:
                            print(f"--- span {k} [{s}:{e}] ({e - s} tok) context-before: {ctx!r}")
                            print(piece if len(piece) < a.chars else piece[: a.chars // 2] + " ... " + piece[-a.chars // 2:])
                            print(f"--- after span: {nxt!r}")
    print(json.dumps({"samples": n, "over_max_tokens": over, "problems": dict(problems)}))


# ----------------------------------------------------------------------------------------------------------------
# vLLM request-log check
# ----------------------------------------------------------------------------------------------------------------

def parse_vllm_log(path: Path) -> list[dict]:
    """Requests logged by vLLM (VLLM_LOGGING_LEVEL=DEBUG and --enable-log-requests), in log order:
    'Request <id> details: prompt: <repr>, prompt_token_ids: [...], prompt_embeds shape: ...'."""
    reqs = []
    with open(path, errors="replace") as f:
        for line in f:
            if " details: prompt: " not in line:
                continue
            head, rest = line.split(" details: prompt: ", 1)
            rid = head.rsplit("Request ", 1)[-1].strip()
            k = rest.rfind(", prompt_token_ids: [")
            if k < 0:
                continue
            prompt = ast.literal_eval(rest[:k])
            j = rest.index("]", k)
            ids = json.loads(rest[k + len(", prompt_token_ids: "): j + 1])
            reqs.append({"id": rid, "prompt": prompt, "ids": ids})
    return reqs


def cmd_check_log(a):
    """Rebuild every model call of the traces and compare with vLLM's logged prompts, text and token ids."""
    rnd = Renderer(a.tokenizer, a.template)
    env = AgentEnv(Path(a.submission))
    reqs = parse_vllm_log(Path(a.log))
    print("logged requests", len(reqs))
    by_text = {}
    for r in reqs:
        by_text.setdefault(r["prompt"], r)
    used = set()
    tasks = load_tasks()
    res = collections.Counter()
    for trace in sorted(Path(a.traces).glob("trace_*.json")):
        iid = trace.stem[len("trace_"):]
        system, items = parse_trace(trace)
        recover_thought_whitespace(items, rnd, res)
        state = base_state(tasks.get(iid, {}))
        states = states_per_item(env, items, state)
        events, index = calls_to_events(items, env.root.name, with_thoughts=True)
        prev = None
        for i, it in enumerate(items):
            if not isinstance(it, Call):
                continue
            agent_name = it.author or env.root.name
            if agent_name not in env.agents:
                res["unknown_agent"] += 1
                continue
            msgs, tools = env.request(agent_name, events[: index[i]], states[i])
            prompt = rnd.render(vllm_conversation(msgs), vllm_tools(tools), add_generation_prompt=True,
                                enable_thinking=env.enable_thinking(agent_name))
            ids = rnd.encode(prompt)
            if a.dump:
                Path(a.dump).mkdir(parents=True, exist_ok=True)
                (Path(a.dump) / f"{iid}_{i:03d}_{agent_name}.txt").write_text(prompt)
            hit = by_text.get(prompt)
            tag = "compacted?" if (prev is not None and prev >= COMPACTION_THRESHOLD) else ""
            if hit is not None:
                used.add(hit["id"])
                res["text_match"] += 1
                res["ids_match" if ids == hit["ids"] else "ids_mismatch"] += 1
                print(f"{iid} item {i:3d} {agent_name:10s} tokens {len(ids):6d} trace {it.prompt_tokens} TEXT MATCH, ids {'match' if ids == hit['ids'] else 'MISMATCH'}")
            else:
                res["no_text_match"] += 1
                cands = [r for r in reqs if r["id"] not in used] or reqs
                same = [r for r in cands if len(r["ids"]) == len(ids)]
                best = max(same or cands, key=lambda r: len(os.path.commonprefix([r["prompt"], prompt])))
                used.add(best["id"])
                cp = len(os.path.commonprefix([best["prompt"], prompt]))
                if len(best["ids"]) == len(ids):
                    nd = sum(1 for x, y in zip(ids, best["ids"]) if x != y)
                    diff_tok = sorted({(rnd.tok.decode([x]), rnd.tok.decode([y])) for x, y in zip(ids, best["ids"]) if x != y})
                    res["same_length_token_diffs"] += nd
                    print(f"   same token count; {nd} token positions differ: {diff_tok}")
                print(f"{iid} item {i:3d} {agent_name:10s} tokens {len(ids):6d} trace {it.prompt_tokens} NO MATCH {tag} "
                      f"(best {best['id']} len {len(best['ids'])}); first difference at char {cp}:\n"
                      f"   ours   {prompt[max(0, cp - 80):cp + 160]!r}\n   logged {best['prompt'][max(0, cp - 80):cp + 160]!r}")
            prev = it.prompt_tokens if it.prompt_tokens is not None else len(ids)
    print("unmatched logged requests:", [r["id"] for r in reqs if r["id"] not in used])
    print(json.dumps(dict(res)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--dataset", required=True)
    b.add_argument("--mode", choices=["think", "nothink"], default="think")
    b.add_argument("--arms", default=",".join(SA_ARMS))
    b.add_argument("--target-submission", default=str(REPO / "submissions/e001n_baseline_nothink_e4b"),
                   help="nothink mode: config whose system prompt, tools and thinking setting are rendered")
    b.add_argument("--max-runs-per-task", type=int, default=4)
    b.add_argument("--val-frac", type=float, default=0.1)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--max-tokens", type=int, default=16384)
    b.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    b.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    b.add_argument("--limit-tasks", type=int, default=0, help="testing: only the first N tasks")
    b.add_argument("--ignore-compaction", action="store_true",
                   help="render the full history as if no compaction had happened (v2 datasets)")
    b.add_argument("--min-loss-turns", type=int, default=1, help="drop samples with fewer trained turns")
    b.add_argument("--per-agent", action="store_true",
                   help="multi-agent arms: one sample per sub-agent call sequence, each with its own context")
    b.add_argument("--val-tasks-from", help="reuse the val tasks of this dataset (keeps splits consistent)")
    b.add_argument("--tail", action="store_true",
                   help="think mode: samples for the turns after the teacher's compaction (synthetic summary)")
    b.add_argument("--all-tails", action="store_true", help="with --tail: one sample per compaction, not only the first")
    b.add_argument("--trajectories-from", help="only the trajectories (run, task) listed in this dataset's meta.json")
    b.add_argument("--trajectories-cut", help="with --trajectories-from: only those with this cut reason (e.g. compaction)")
    v = sp.add_parser("verify")
    v.add_argument("--dataset", required=True)
    v.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    v.add_argument("--max-tokens", type=int, default=16384)
    v.add_argument("--show", type=int, default=2)
    v.add_argument("--spans", type=int, default=3)
    v.add_argument("--chars", type=int, default=1500)
    v.add_argument("--sample-id")
    c = sp.add_parser("check-log")
    c.add_argument("--log", required=True)
    c.add_argument("--traces", required=True)
    c.add_argument("--submission", required=True)
    c.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    c.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    c.add_argument("--verbose", action="store_true")
    c.add_argument("--dump", help="write each rebuilt prompt to this directory")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)
    {"build": cmd_build, "verify": cmd_verify, "check-log": cmd_check_log}[a.cmd](a)


if __name__ == "__main__":
    main()
