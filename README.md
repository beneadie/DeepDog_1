# Kiyosi Deep Research v1

Multi-agent research pipeline built with LangGraph + LangChain. It takes a prompt, runs iterative research, and generates a citation-backed report.

## What this project does

- Converts a user prompt into a structured research brief
- Runs a supervisor + researcher agent loop for evidence gathering
- Produces a final markdown report
- Optionally exports source metadata and a research trace

Core entrypoint: `run_research.py`

## Requirements

- Python 3.11+ recommended
- One LLM provider API key (Gemini/OpenAI/Anthropic/Cerebras)
- Tavily API key for web search (recommended for normal operation)

Dependencies are listed in `requirements.txt`.

## 1) Setup

### Option A: `pip` + `requirements.txt` (standard)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Option B: `uv` (optional)

```powershell
uv venv
.\venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

## 2) Configure environment variables

Create a `.env` file at the project root.

`run_research.py` loads this exact file (`./.env`), and `.env` is already gitignored in this repo, so your secrets will not be committed.

### Required keys for current default config

Current `deep_research/config.py` defaults to Gemini fallback models, so you need:

- `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) for model calls
- `TAVILY_API_KEY` for core web-search tool use

Use this template and replace placeholder values with your real keys:

```dotenv
GOOGLE_API_KEY=<YOUR_GOOGLE_OR_GEMINI_API_KEY>
TAVILY_API_KEY=<YOUR_TAVILY_API_KEY>
```

If you prefer the Gemini alias, this also works:

```dotenv
GEMINI_API_KEY=<YOUR_GOOGLE_OR_GEMINI_API_KEY>
TAVILY_API_KEY=<YOUR_TAVILY_API_KEY>
```

### If you change providers in `deep_research/config.py`

Set the provider key that matches your chosen models:

```dotenv
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
# or
ANTHROPIC_API_KEY=<YOUR_ANTHROPIC_API_KEY>
# or
CEREBRAS_API_KEY=<YOUR_CEREBRAS_API_KEY>
```

Tooling note: `PERPLEXITY_KEY` is required for Substack/Perplexity-based tool flows. If you want full tool coverage across web + Substack discovery, set both `TAVILY_API_KEY` and `PERPLEXITY_KEY`.

Optional (needed for Substack discovery tools):

```dotenv
PERPLEXITY_KEY=<YOUR_PERPLEXITY_API_KEY>
```

## 3) Run the project

### Prompt as CLI arg

```powershell
python run_research.py --prompt "What are the latest developments in AI safety?"
```

### Prompt from file

```powershell
python run_research.py --prompt-file input.txt
```

### Custom output directory

```powershell
python run_research.py --prompt-file input.txt --output-dir outputs
```

## Output files

By default (`OUTPUT_MODE = "file"` in `deep_research/config.py`), the run writes:

- `research_<timestamp>.md` (main report)
- `research_data_<timestamp>.json` (sources metadata, if collected)
- `trace_<timestamp>.md` (compressed research process trace, when available)
- `error_<timestamp>.txt` (only when a run fails)

## Config knobs

Main runtime config lives in `deep_research/config.py`:

- `DEFAULT_MODEL` and `MODEL_FALLBACK_CHAIN`
- `RESEARCH_TIME_MIN_MINUTES` / `RESEARCH_TIME_MAX_MINUTES`
- `OUTPUT_MODE` (`file`, `db`, `both`)
- `SAVE_REPORT_TO_FILE`
- `ENABLE_SUBTOPIC_GENERATION`

Recommended model setting for this repo:

- `DEFAULT_MODEL = "gemini-3-flash-preview"` for strong cost efficiency, speed, and depth-per-minute.

## Troubleshooting

### 1) Import errors for `deep_research`

Run commands from repository root so `run_research.py` can resolve imports correctly.

### 2) LLM auth/provider errors

Check:

- your API key exists in `.env`
- `DEFAULT_MODEL` in `deep_research/config.py` matches the provider key you supplied

### 3) Long stall at draft-generation stage

This repo includes a fix that makes draft generation async + timeout based, and avoids structured output wrapping for long drafts.

## Agent Flow

The system uses a **diffusion-based approach**: generate a loose draft from internal knowledge, then iteratively refine it with real research. The architecture is a hierarchical multi-agent system built with LangGraph.

### High-level pipeline

```
User Query
  |
  v
clarify_with_user          (pass-through; clarification logic currently disabled)
  |
  v
write_research_brief       (LLM converts conversation into a detailed research brief)
  |
  v
write_draft_report         (LLM writes an initial draft from internal knowledge only)
  |
  v
supervisor_subgraph        (iterative research + draft refinement loop)
  |
  v
final_report_generation    (synthesizes findings + draft into a citation-backed report)
  |
  v
subtopic_evaluation        (optional; decides if supplementary reports are needed)
  |
  v
subtopic_generation        (optional; generates detailed subtopic reports in parallel)
```

**Files**: `research_agent_scope.py` (scoping nodes), `research_agent_full.py` (full graph wiring).

### Stage 1 -- Scoping

| Node | What it does | Output |
|---|---|---|
| `clarify_with_user` | Placeholder for user clarification (currently skipped). | Routes to `write_research_brief`. |
| `write_research_brief` | Translates raw user messages into a concrete, detailed research brief using structured output. | `research_brief` string stored in state. |
| `write_draft_report` | Generates an initial draft report from the LLM's internal knowledge, guided only by the research brief. No citations -- uses `[RESEARCH_NEEDED]` placeholders where data is missing. | `draft_report` string stored in state, plus `supervisor_messages` seeded with the draft and brief. |

### Stage 2 -- Supervisor research loop

**File**: `multi_agent_supervisor.py`

The supervisor is a looping subgraph that coordinates parallel sub-agents:

```
supervisor  <-->  supervisor_tools
   |                  |
   |     (delegates)  |---> ConductResearch  (deep-dive sub-agent)
   |                  |---> DiscoverOpportunities  (broad exploratory sub-agent)
   |                  |---> refine_draft_report  (rewrites draft with new findings)
   |                  |---> think_tool  (LLM reflection)
   |
   +---> ResearchComplete  (exits loop)
```

**How each iteration works:**

1. The **supervisor** node receives the current draft, research brief, collected findings, and elapsed time. It decides what to research next.
2. The **supervisor_tools** node executes the supervisor's tool calls:
   - `ConductResearch` / `DiscoverOpportunities` spawn sub-agents that run **in parallel** (up to 4 research + 2 discovery agents concurrently).
   - Each sub-agent returns compressed findings which are appended to the shared `notes` list.
   - `refine_draft_report` rewrites the draft report incorporating the new findings.
3. Control returns to the supervisor for the next iteration, or the loop exits when `ResearchComplete` is called.

**Constraints**: max 15 iterations, hard timeout of 17 minutes, configurable min/max research time.

### Stage 2a -- Sub-agents

**File**: `researcher_agent.py`

Both research and discovery agents share the same graph structure but differ in their tool sets and prompts:

```
llm_call  <-->  tool_node
   |
   v
compress_research  (synthesizes raw findings into a structured summary)
```

**Research agent tools**: `tavily_search`, `think_tool`, `get_reddit_post`, `google_search_grounding`, `get_subreddit_posts`, `search_term_in_subreddit`

**Discovery agent tools**: all research tools plus `search_substack`, `read_substack_article`

Each sub-agent has a 10-minute timeout. Results are compressed into a findings summary with inline citations and a sources list before being returned to the supervisor.

### Stage 3 -- Final report generation

**File**: `research_agent_full.py`

The `final_report_generation` node receives the research brief, all collected findings, and the refined draft report. It produces a comprehensive markdown report with:
- Inline `[1]`, `[2]` citations
- A `CitationPlanList` for ordering sources
- A `## Sources` section at the end

After generation, a **citation validation** step checks that inline citations match the sources list. If validation fails, a single LLM repair pass is attempted. If the repair is rejected (e.g., body length shrank too much), the original report is kept as-is.

### Stage 4 -- Subtopic generation (optional)

Enabled via `ENABLE_SUBTOPIC_GENERATION` in config.

1. `subtopic_evaluation`: an LLM reviews the final report and research topics to decide whether any sub-topics deserve dedicated deep-dive reports. Uses tool calls (`GenerateSubtopicReport` / `EndSubtopicEvaluation`) to express its decisions.
2. `subtopic_generation`: for each approved subtopic, generates a full report from the collected research notes. All subtopic reports are generated **in parallel**.

### Data flow summary

```
AgentState carries these key fields through the pipeline:

  messages              -- user conversation history
  research_brief        -- structured research question (set in Stage 1)
  draft_report          -- evolving draft (set in Stage 1, refined in Stage 2)
  supervisor_messages   -- supervisor conversation log
  notes                 -- compressed findings from sub-agents (accumulated in Stage 2)
  raw_notes             -- unprocessed sub-agent notes
  final_report          -- citation-backed output (set in Stage 3)
  secondary_reports     -- optional subtopic reports (set in Stage 4)
```
