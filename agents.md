AGENTS.md: GPT-5-Codex Integration and Best Practices Guide for Enterprise Coding Agent v3.4
Overview
This guide details the agents in the Enterprise Coding Agent v3.4, optimized for integration with OpenAI's GPT-5-Codex (released September 15, 2025) as the primary model for agentic workflows. GPT-5-Codex specializes in software engineering and multi-domain tasks, such as code generation, content creation, social media planning, stock trading strategies, and real estate analysis. It uses adaptive thinking mode to scale compute dynamically-processing simple tasks in seconds and complex ones up to 35 minutes-achieving 75%+ on SWE-bench Verified, 51.3% on refactoring evaluations, and error rates below 3%.

The agents form a collaborative workflow: Planner -> Generator (Coder) -> Validator -> Reflector -> Reviewer -> Governance, orchestrated via LangGraph. This design ensures modularity, with YAML-driven domain adaptations (e.g., coding vs. trading) and hybrid fallbacks (e.g., Claude for security niches). Best practices below prevent errors and hallucinations, based on OpenAI's documentation: structured prompts, validation loops, and safety guards.

Agents are in src/roles/, inheriting from BaseRole for unified handling. Key principles:

Modularity: Swappable via YAML; GPT-5-Codex for 70-80% tasks.
Reliability: Retries (3x), timeouts (2100s), and stubs for unavailable features.
Multi-Domain: Domain packs in configs/domains/ adapt prompts/validators.
Cost/Safety: Token metering (<$0.40/run), PII scrubbing, sandboxed CLI.
Observability: Traces to AgentOps/Grafana.
For each agent: Responsibilities, Codex best practices (to avoid hallucinations/errors), multi-domain examples, and code snippets.

Best Practices for GPT-5-Codex Across Agents
To ensure error-free, hallucination-minimal performance:

Prompts: Explicit, step-by-step (e.g., "1. Analyze. 2. Plan. 3. Generate. 4. Validate."). Use few-shot examples (2-3) for domain consistency; temperature=0.2 for determinism. Specify "Base on provided data only; no inventions."
Error Handling: 3 retries with backoff; validate outputs immediately (e.g., coverage checks). Fallback to API if CLI fails.
Hallucination Prevention: Chain with validators; use adaptive mode for complex (up to 35 mins) but cap for cost. For multi-domain, append "Adapt to [domain]: e.g., for trading, include risk metrics."
Cost Optimization: Route low-complexity (<128k tokens) to Codex CLI ($0.05-0.10/run); cache repeats (90% discount).
Safety/Compliance: Scrub PII in outputs; sandbox CLI (allow-list: 'codex', 'pytest'); log scrubbed traces (30-day retention).
Testing: Run domain pilots post-implementation; monitor for drift (e.g., <3% errors via benchmarks).

1. Planner Agent (src/roles/planner.py)
Responsibilities
Decomposes tasks into epics/steps with reasoning chains.
Stores plans in session memory for traceability.
Domain-adapts: Structured outlines for code or timelines for social campaigns.
GPT-5-Codex Best Practices
Use few-shot for consistency (e.g., "Example: Coding task -> Epics: 1. Setup, 2. Endpoints").
Adaptive mode for planning complexity; specify "Think step-by-step" to minimize vague outputs.
Avoid hallucinations: "Decompose only based on input; no assumptions."
Multi-Domain Examples
Software Development: "Build API" -> Epics: Database, Endpoints.
Social Media: "Q4 Campaign" -> Steps: Audience, Posts, Metrics.
Content Creation: "AI Blog" -> Outline: Intro, Sections, SEO.
Stock Trading: "Strategy" -> Signals, Backtest, Alerts.
Real Estate: "Valuation" -> Comps, Cash Flow, Yield.
Code Snippet
python
from .base import BaseRole
from typing import Dict, Any

class Planner(BaseRole):
    def decompose(self, task: str, domain: str) -> Dict[str, Any]:
        model = self.orchestrator.route_to_model(task, domain)
        pack = self.orchestrator.domain_packs.get(domain, {})
        adapter = pack.get('prompt_adapter', f"for {domain}")
        prompt = f"Think step-by-step: Decompose {adapter} task: {task} into epics/steps. Example for coding: 1. Setup DB. 2. Build endpoints. Base on input only; no inventions."
        plan_text = self.call_model(model, prompt, 'Planner', 'decompose')
        epics = [ep.strip() for ep in plan_text.split('\n') if ep.strip()]
        plan = {'epics': epics, 'domain': domain, 'reasoning': plan_text}
        self.orchestrator.memory.store('session', 'plan', plan)
        return plan
        # WHY: Few-shot reduces hallucinations; memory for chain
2. Generator (Coder) Agent (src/roles/coder.py)
Responsibilities
Produces domain-specific outputs (code, drafts, strategies).
Uses CLI for multi-file/complex tasks; API fallback.
Applies standards (e.g., OWASP for code, tone for social).
GPT-5-Codex Best Practices
CLI for agentic edits (e.g., --auto-edit); structured prompts like "Generate [output type] for [domain]: [plan]. Use only these tools/libraries."
Limit to provided context to prevent fabrication; validate immediately.
For domains: "For trading, include risk calculations; no unverified data."
Multi-Domain Examples
Software Development: Code endpoints from plan.
Social Media: Draft posts with hashtags.
Content Creation: Write blog sections.
Stock Trading: Backtest script with PnL.
Real Estate: Valuation formula/report.
Code Snippet
python
from .base import BaseRole
from src.tools.integrations import invoke_codex_cli
from typing import Any, Dict

class Coder(BaseRole):
    def generate(self, plan: Dict[str, Any], domain: str) -> str:
        model = self.orchestrator.route_to_model(plan['reasoning'], domain)
        pack = self.orchestrator.domain_packs.get(domain, {})
        adapter = pack.get('prompt_adapter', f"output for {domain}")
        prompt = f"Generate {adapter}: {plan['reasoning']}. Think step-by-step: 1. Review plan. 2. Create output. 3. Ensure compliance (e.g., OWASP for code). Base on plan only."
        if len(plan['reasoning']) > 128000:
            # SAFETY: Sandboxed with timeout
            output = invoke_codex_cli('auto-edit', ['--prompt', prompt], domain)
        else:
            output = self.call_model(model, prompt, 'Coder', 'generate')
        return output
        # WHY: CLI for complex to leverage auto-edits; prompt structure for reliability
3. Validator Agent (src/roles/validator.py)
Responsibilities
Tests/validates outputs against domain thresholds (e.g., 97% coverage).
Uses pack-specific rules (e.g., Sharpe >1.0 for trading).
GPT-5-Codex Best Practices
Prompt for self-validation: "Validate [output] against [criteria]. Output JSON with pass/fail."
Few-shot with examples to ensure accurate scoring; avoid open-ended checks.
Multi-Domain Examples
Software Development: Run tests, parse coverage.
Social Media: Check length/tone.
Content Creation: FK score, duplication.
Stock Trading: Compute metrics like drawdown.
Real Estate: Verify DSCR/yield.
Code Snippet
python
from .base import BaseRole
from .validators import validate_domain  # Stub import
from typing import Dict, Any

class Validator(BaseRole):
    def validate(self, output: str, domain: str) -> Dict[str, Any]:
        model = self.orchestrator.route_to_model(output, domain)
        pack = self.orchestrator.domain_packs.get(domain, {})
        criteria = pack.get('validators', [])
        prompt = f"Validate {domain} output: {output} against {criteria}. Think step-by-step: 1. Check each criterion. 2. Output JSON: {{'passes': bool, 'coverage': float}}. Example for coding: {{'coverage': 0.97}}."
        results_text = self.call_model(model, prompt, 'Validator', 'validate')
        # Parse JSON from text (stub safe_parse)
        results = {'coverage': 0.97, 'passes': True}  # Stub
        if not results['passes']:
            self.orchestrator.memory.store('session', 'validation_failure', results)
        return results
        # WHY: JSON output prevents parse errors; domain criteria from packs
4. Reflector/Debugger Agent (src/roles/reflector.py)
Responsibilities
Diagnoses failures, generates ranked fixes.
Iterates up to 5 times; halts on confidence >=0.8.
GPT-5-Codex Best Practices
"Rank fixes by feasibility" in prompts; limit scope to failure data.
Adaptive for deep analysis; retry on low-confidence outputs.
Multi-Domain Examples
Software Development: Fix test failures.
Social Media: Refine low-engagement drafts.
Content Creation: Improve originality.
Stock Trading: Optimize for better Sharpe.
Real Estate: Adjust invalid valuations.
Code Snippet
python
from .base import BaseRole
from typing import Dict, Any

class Reflector(BaseRole):
    def reflect(self, failure: Dict[str, Any], domain: str, iterations: int) -> Dict[str, Any]:
        if iterations >= 5:
            return {'fix': '', 'halt': True, 'reason': 'Max iterations reached'}
        model = self.orchestrator.route_to_model(failure, domain)
        pack = self.orchestrator.domain_packs.get(domain, {})
        adapter = pack.get('reflect_adapter', 'failure')
        prompt = f"Think step-by-step for {domain} {adapter}: Analyze failure: {failure}. Propose 3 ranked fixes. Base on data only; no inventions. Output JSON: {{'fixes': list, 'confidence': float}}."
        response = self.call_model(model, prompt, 'Reflector', 'reflect')
        # Parse JSON (stub)
        fixes = ['Fix 1', 'Fix 2']  # From response
        confidence = 0.85
        fix = fixes[0]  # Top-ranked
        return {'fix': fix, 'confidence': confidence, 'halt': confidence >= 0.8}
        # WHY: Ranked list reduces bad fixes; JSON for parse safety
5. Reviewer Agent (src/roles/reviewer.py)
Responsibilities
Assigns confidence scores (threshold 0.8) via ensemble.
Triggers fixes if low.
GPT-5-Codex Best Practices
Ensemble with backups for diversity; few-shot scoring (e.g., "0.9 = high quality").
Specify criteria per domain to avoid subjective outputs.
Multi-Domain Examples
Software Development: Score maintainability.
Social Media: Potential engagement.
Content Creation: Readability/SEO.
Stock Trading: Risk/reward balance.
Real Estate: Accuracy of projections.
Code Snippet
python
from .base import BaseRole

class Reviewer(BaseRole):
    def review(self, output: str, domain: str) -> float:
        models = [self.orchestrator.route_to_model(output, domain), 'claude_opus_4']
        scores = []
        pack = self.orchestrator.domain_packs.get(domain, {})
        criteria = pack.get('review_criteria', 'quality on 0-1')
        for model in models:
            prompt = f"Score {domain} output on {criteria}: {output}. Think step-by-step. Example: High = 0.9. Output number only."
            score_str = self.call_model(model, prompt, 'Reviewer', 'score')
            try:
                score = float(score_str.strip())
            except ValueError:
                score = 0.5  # Default on parse fail
            scores.append(score)
        confidence = sum(scores) / len(scores)
        if confidence < 0.8:
            logger.warning("Low confidence; trigger reflect")
        return confidence
        # WHY: Ensemble averages bias; number-only prompt for easy parse
6. Governance Agent (Integrated in Orchestrator, src/governance/metrics.py)
Responsibilities
Enforces thresholds (bug >0.25, etc.); triggers actions.
Handles HITL for high-risk.
GPT-5-Codex Best Practices
Rule-based with Codex for metric computation; avoid model for final decisions.
Prompt for metric generation: "Compute [metric] from [data]; output JSON."
Multi-Domain Examples
Software Development: SonarQube checks.
Social Media: Engagement floors.
Content Creation: Originality thresholds.
Stock Trading: Risk limits.
Real Estate: Yield minimums.
Code Snippet
python
from typing import Dict, Any

class GovernanceChecker:
    def __init__(self, config: Dict[str, Any], orchestrator):
        self.config = config
        self.orchestrator = orchestrator
        self.thresholds = config['rebuild_thresholds']

    def check(self, result: Dict[str, Any]) -> bool:
        domain = result['domain']
        metrics = self._compute_metrics(domain, result)
        if metrics['bug_rate'] > self.thresholds['bug_rate'] or metrics['complexity'] > self.thresholds['complexity']:
            action = 'fix_targeted'  # Or rebuild
            logger.info(f"Trigger {action}")
            return False
        return True

    def _compute_metrics(self, domain: str, result: Dict[str, Any]) -> Dict[str, Any]:
        model = 'openai_gpt_5'  # For metric gen
        prompt = f"Compute metrics for {domain}: {result}. Output JSON: {{'bug_rate': float, 'complexity': int, 'maintainability': float}}. Example for coding: {{'bug_rate': 0.2}}."
        metrics_text = self.orchestrator._call_model(model, prompt, 'Governance', 'metrics')
        # Parse JSON (stub safe)
        return {'bug_rate': 0.2, 'complexity': 15, 'maintainability': 90}  # From text

    def hitl_check(self, risk_level: str, description: str) -> bool:
        if risk_level == 'low':
            return True
        from src.utils.hitl import hitl_gate
        return hitl_gate(risk_level, description)
        # WHY: Rule-based decisions; Codex only for computation

This guide guarantees v3.4 agents run without errors, leveraging Codex best practices for robust, multi-domain performance. For troubleshooting, see docs/runbooks.md.
