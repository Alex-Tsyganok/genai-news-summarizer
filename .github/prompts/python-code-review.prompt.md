---
mode: 'ask'
model: GPT-5 (Preview)
description: 'Perform a Code Review'
---
Perform a Python code review tailored for a developer with a .NET/C# background. Focus on correctness, readability, performance, security, testability, and Pythonic style. Call out key differences from .NET and provide concrete suggestions with small before/after examples and .NET analogies where helpful.

Sections to produce:

1) Summary
- One-paragraph overview, risk level (Low/Medium/High), and top 3 issues.

2) Key .NET → Python differences to watch for (with quick examples)
- Types & data modeling: prefer typing and dataclasses.
	- C#: public record Article(string Title, DateTimeOffset PublishedAt);
	- Python (after):
		@dataclass
		class Article:
				title: str
				published_at: datetime
- Exceptions & error handling: EAFP over LBYL; no checked exceptions.
	- C#: if (File.Exists(path)) { /* ... */ }
	- Python (after):
		try:
				f = open(path)
		except FileNotFoundError:
				...
- Resource management: context managers (with) vs using.
	- C#: using var stream = new FileStream(...);
	- Python (after):
		with open(path) as f:
				...
- Async & concurrency: asyncio vs Task; GIL impact.
	- I/O-bound: asyncio/await (aiohttp). CPU-bound: ProcessPoolExecutor.
- Collections/iteration: list/dict/set; list comprehensions vs LINQ.
	- C#: var evens = nums.Where(n => n % 2 == 0).ToList();
	- Python (after): evens = [n for n in nums if n % 2 == 0]
- Logging: logging module vs ILogger/Serilog; structured logging via extra and dictConfig.
- Configuration: env vars/.env vs appsettings.json; consider pydantic-settings for complex configs.
- Testing: pytest fixtures/parametrize vs xUnit; keep tests fast and isolated.
- Style: PEP 8; use Black (format), Ruff (lint), Mypy (types).

3) Review checklist with actionable findings (mark each Pass/Fail/N/A)
- Correctness: input validation; edge cases (None, empty, large inputs); timezone-safe datetimes (zoneinfo); encoding (UTF-8); error handling (narrow excepts); immutable defaults (avoid {}/[] in params); resource cleanup (with/async with).
- Security: avoid eval/exec/pickle on untrusted data; safe subprocess (avoid shell=True unless necessary); path traversal checks; secrets from env/secret store (not in source); dependency scanning (pip-audit); HTTP requests include timeouts and TLS verify.
- Performance: avoid N+1 I/O/DB; prefer generators/iterators for streams; batch operations; caching (functools.lru_cache); vectorize data ops (numpy/pandas); use set/dict lookups; avoid quadratic loops.
- Maintainability: small, focused functions; clear naming; docstrings and type hints; modular layout; dependency injection via parameters/factories; minimize global state.
- Testing: pytest coverage for happy, edge, and error paths; mock I/O, time, and network; deterministic tests.
- Style: PEP 8; import order (stdlib, third-party, local); f-strings; pathlib for paths.
- Types: typing usage (TypedDict/Protocol/NamedTuple); mypy clean on practical strict settings.

4) TODO list
- Return a Markdown checklist grouped by Priority (P0 Critical, P1 Important, P2 Nice-to-have) and Issue Type (Correctness, Security, Performance, Maintainability, Style, Testing). For each item include: concise title, rationale, and suggested fix (add a short code example when useful, with a brief .NET analogy if non-obvious).

5) Quick diffs (optional)
- Provide 2–5 before → after snippets for the most impactful changes.
- Add a short “.NET analogy” line when mapping concepts (e.g., using → with; LINQ → list comprehensions/generators; ILogger → logging).

6) Recommended tooling commands (optional)
- Formatting/Linting: black .; ruff check .
- Types: mypy .
- Security: bandit -q -r src; pip-audit
- Tests: pytest -q

Return the TODO list in a Markdown format, grouped by priority and issue type.