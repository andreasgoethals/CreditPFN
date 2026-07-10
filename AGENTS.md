# Shared agent instructions

These repository-level instructions apply to Codex and any Codex subagents.

At the start of every task in this repository:

1. Read `AGENTS_MEMORY.md` completely if it exists. It is gitignored shared
   local memory containing experiment findings, prior bugs, and deliberate
   implementation choices. Preserve its compact/current maintenance rules.
2. Read `AGENTS_HISTORY.md` before editing so you know what Claude and Codex
   changed previously and why.
3. Treat the literature and upstream repository dumps as evidence, but verify
   claims against the primary paper or implementation when sources disagree.

Before finishing every user request or agent session, append one concise entry
to `AGENTS_HISTORY.md` with the date, agent (`Codex`), what changed or was
reviewed, and why. Record read-only work as "no repository changes" when
applicable. Do not put secrets, credentials, private dataset contents, or large
log excerpts in either agent file.

Keep durable scientific and operational facts in the committed documentation;
use `AGENTS_MEMORY.md` only for transient findings and pitfalls that should not
be committed.
