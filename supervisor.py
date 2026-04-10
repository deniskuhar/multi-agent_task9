from __future__ import annotations

import json
import re
import uuid
import asyncio

import httpx
from fastmcp import Client as MCPClient
from langchain_openai import ChatOpenAI

from config import REPORT_REVISION_PROMPT, get_settings
from schemas import CritiqueResult, ResearchPlan

settings = get_settings()

revision_model = ChatOpenAI(
    model=settings.model_name,
    api_key=settings.openai_api_key.get_secret_value(),
    temperature=0.1,
    timeout=settings.request_timeout_seconds,
)


def _safe_filename_from_request(request: str) -> str:
    request_lower = request.lower()

    if "rag" in request_lower:
        return "rag_comparison.md"

    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", request_lower).strip("_")
    if not slug:
        slug = "research_report"
    return f"{slug[:40]}.md"


def _dedupe_queries(queries: list[str], limit: int = 3) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for q in queries:
        key = q.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(q.strip())
        if len(result) >= limit:
            break

    return result


def _extract_acp_output(data: dict) -> str:
    output = data.get("output")

    if isinstance(output, str):
        return output

    if isinstance(output, dict):
        parts = output.get("parts", [])
        texts: list[str] = []
        for part in parts:
            if isinstance(part, dict) and part.get("content"):
                texts.append(str(part["content"]))
        if texts:
            return "\n".join(texts)
        return json.dumps(output, ensure_ascii=False, indent=2)

    if isinstance(output, list):
        texts: list[str] = []
        for msg in output:
            if isinstance(msg, dict):
                for part in msg.get("parts", []):
                    if isinstance(part, dict) and part.get("content"):
                        texts.append(str(part["content"]))
        if texts:
            return "\n".join(texts)

    return json.dumps(data, ensure_ascii=False, indent=2)


def _run_agent(agent_name: str, prompt: str) -> str:
    payload = {
        "agent_name": agent_name,
        "session_id": None,
        "session": None,
        "input": [
            {
                "role": "user",
                "parts": [
                    {
                        "content_type": "text/plain",
                        "content": prompt,
                    }
                ],
            }
        ],
        "mode": "sync",
    }

    response = httpx.post(
        f"{settings.acp_base_url}/runs",
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return _extract_acp_output(data)


def plan(request: str) -> ResearchPlan:
    print(f"\n[Supervisor → ACP → Planner]\n  delegating planner for: {request!r}")
    raw = _run_agent("planner", request)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise Exception(f"Planner returned invalid JSON:\n{raw}")

    plan_obj = ResearchPlan.model_validate(data)
    print(json.dumps(plan_obj.model_dump(), indent=2, ensure_ascii=False))
    return plan_obj


def research(prompt: str) -> str:
    print(f"\n[Supervisor → ACP → Researcher]\n  delegating research")
    result = _run_agent("researcher", prompt)
    preview = result[:500]
    print(preview + ("..." if len(result) > 500 else ""))
    return result


def critique(
    *,
    original_request: str,
    plan_obj: ResearchPlan,
    findings: str,
) -> CritiqueResult:
    print(f"\n[Supervisor → ACP → Critic]\n  delegating critique")

    critique_input = f"""
Original user request:
{original_request}

Approved research plan:
{json.dumps(plan_obj.model_dump(), ensure_ascii=False, indent=2)}

Current findings:
{findings}
""".strip()

    raw = _run_agent("critic", critique_input)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise Exception(f"Critic returned invalid JSON:\n{raw}")

    critique_obj = CritiqueResult.model_validate(data)
    print(json.dumps(critique_obj.model_dump(), indent=2, ensure_ascii=False))
    return critique_obj


def _build_research_request(
    *,
    original_request: str,
    plan_obj: ResearchPlan,
    round_index: int,
    critique_obj: CritiqueResult | None = None,
    previous_findings: str | None = None,
) -> str:
    if critique_obj is not None and critique_obj.revision_requests:
        queries = _dedupe_queries(critique_obj.revision_requests, limit=2)
    else:
        queries = _dedupe_queries(plan_obj.search_queries, limit=3)

    lines = [
        f"Original user request:\n{original_request}",
        "",
        f"Research goal:\n{plan_obj.goal}",
        "",
        "Search queries:",
    ]

    for q in queries:
        lines.append(f"- {q}")

    lines.extend(
        [
            "",
            f"Preferred sources: {', '.join(plan_obj.sources_to_check)}",
            f"Expected output format: {plan_obj.output_format}",
            "",
            f"Current round: {round_index}",
        ]
    )

    if previous_findings:
        lines.extend(
            [
                "",
                "Previous findings summary:",
                previous_findings[:4000],
            ]
        )

    if critique_obj and critique_obj.revision_requests:
        lines.extend(
            [
                "",
                "You are revising existing research. Focus only on these critique requests:",
            ]
        )
        for item in critique_obj.revision_requests:
            lines.append(f"- {item}")

        if critique_obj.gaps:
            lines.extend(["", "Known gaps to address:"])
            for gap in critique_obj.gaps:
                lines.append(f"- {gap}")

        lines.extend(
            [
                "",
                "Do not restart from scratch. Improve the existing findings and fill only the missing essential gaps.",
            ]
        )

    return "\n".join(lines)


def _build_final_report(
    *,
    original_request: str,
    plan_obj: ResearchPlan,
    findings: str,
    final_critique: CritiqueResult | None,
    revision_rounds_used: int,
) -> str:
    lines = [
        "# Research Report",
        "",
        "## User Request",
        original_request,
        "",
        "## Research Goal",
        plan_obj.goal,
        "",
        "## Findings",
        findings.strip(),
        "",
        "## Process Summary",
        f"- Revision rounds used: {revision_rounds_used}",
    ]

    if final_critique is not None:
        lines.extend(
            [
                f"- Final critic verdict: {final_critique.verdict}",
                f"- Freshness: {'yes' if final_critique.is_fresh else 'no'}",
                f"- Completeness: {'yes' if final_critique.is_complete else 'no'}",
                f"- Structure: {'yes' if final_critique.is_well_structured else 'no'}",
            ]
        )

        if final_critique.strengths:
            lines.extend(["", "## Strengths"])
            for item in final_critique.strengths:
                lines.append(f"- {item}")

        if final_critique.gaps:
            lines.extend(["", "## Remaining Limitations"])
            for item in final_critique.gaps:
                lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Conclusion",
            "The comparison shows that naive RAG is the simplest but least context-aware approach, sentence-window improves local context handling, and parent-child retrieval is more structured and effective for complex document relationships. The best choice depends on the balance between simplicity, contextual quality, and implementation complexity.",
        ]
    )

    return "\n".join(lines).strip() + "\n"


def run_supervisor(user_input: str) -> dict:
    plan_obj = plan(user_input)

    findings = research(
        _build_research_request(
            original_request=user_input,
            plan_obj=plan_obj,
            round_index=1,
        )
    )

    final_critique: CritiqueResult | None = None
    revision_rounds_used = 0

    for revision_round in range(settings.max_revision_rounds + 1):
        critique_obj = critique(
            original_request=user_input,
            plan_obj=plan_obj,
            findings=findings,
        )
        final_critique = critique_obj

        if critique_obj.verdict == "APPROVE":
            revision_rounds_used = revision_round
            break

        if revision_round == settings.max_revision_rounds:
            print("\n[Supervisor] Max revision rounds reached. Proceeding with current findings.")
            revision_rounds_used = revision_round
            break

        findings = research(
            _build_research_request(
                original_request=user_input,
                plan_obj=plan_obj,
                round_index=revision_round + 2,
                critique_obj=critique_obj,
                previous_findings=findings,
            )
        )
        revision_rounds_used = revision_round + 1

    final_report = _build_final_report(
        original_request=user_input,
        plan_obj=plan_obj,
        findings=findings,
        final_critique=final_critique,
        revision_rounds_used=revision_rounds_used,
    )

    return {
        "filename": _safe_filename_from_request(user_input),
        "content": final_report,
        "plan": plan_obj,
        "findings": findings,
        "critique": final_critique,
        "revision_rounds_used": revision_rounds_used,
    }


def revise_report_with_feedback(report: dict, feedback: str) -> dict:
    prompt = f"""
{REPORT_REVISION_PROMPT}

User feedback:
{feedback}

Current filename:
{report['filename']}

Current report:
{report['content']}
""".strip()

    result = revision_model.invoke(prompt)
    revised_content = result.content if isinstance(result.content, str) else str(result.content)

    updated = dict(report)
    updated["content"] = revised_content
    return updated


def save_report_via_mcp(filename: str, content: str) -> str:
    async def _inner() -> str:
        async with MCPClient(settings.report_mcp_url) as client:
            result = await client.call_tool(
                "save_report",
                {
                    "filename": filename,
                    "content": content,
                },
            )

            data = getattr(result, "data", None)
            if isinstance(data, str):
                return data

            if isinstance(data, dict):
                if "result" in data:
                    return str(data["result"])
                return json.dumps(data, ensure_ascii=False)

            return str(result)

    return asyncio.run(_inner())


def new_thread_id() -> str:
    return str(uuid.uuid4())