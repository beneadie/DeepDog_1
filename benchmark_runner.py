#!/usr/bin/env python3
"""
Benchmark Runner for Deep Research Agent

Runs all 100 research tasks from query.jsonl through the agent, saving each result
as an individual JSON file with the task details, final report, and citations.

Usage:
    python benchmark_runner.py
    python benchmark_runner.py --output-dir custom_benchmark_outputs
    python benchmark_runner.py --test-mode  # Run only the first task for testing
    python benchmark_runner.py --concurrency 2  # Run 2 tasks in parallel
    python benchmark_runner.py --resume  # Resume from latest run
"""

import asyncio
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Import the run_research function from run_research.py
from run_research import run_research

# Load all benchmark questions from query.jsonl
from benchmark_questions import QUESTIONS

# Delay between tasks in seconds (1 minute) — applied per-slot after each task
DELAY_BETWEEN_TASKS = 60

# Maximum retries for transient failures (timeouts, network errors)
MAX_RETRIES = 2


def aggregate_jsonl(output_dir: Path, model_name: str) -> Path:
    """
    Aggregate all completed task_*.json files into a single JSONL submission file.

    Returns the path to the generated JSONL file.
    """
    submission_dir = Path("data/test_data/raw_data")
    submission_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = submission_dir / f"{model_name}.jsonl"

    results = []
    for json_file in sorted(output_dir.glob("task_*.json")):
        if "_error" in json_file.stem:
            continue
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        results.append({
            "id": data["id"],
            "prompt": data["prompt"],
            "article": data["article"],
        })

    # Sort by id for consistent ordering
    results.sort(key=lambda x: x["id"])

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"JSONL submission written: {jsonl_path} ({len(results)} entries)")
    return jsonl_path


async def run_benchmark(
    output_base_dir: Path,
    test_mode: bool = False,
    resume_dir: Path = None,
    concurrency: int = 3,
    model_name: str = "deep_research_agent",
) -> None:
    """
    Run benchmark tasks and save results as individual JSON files.

    Args:
        output_base_dir: Base directory for benchmark outputs
        test_mode: If True, only run the first task
        resume_dir: If provided, resume from this existing run directory
        concurrency: Maximum number of tasks to run in parallel
        model_name: Name used for the output JSONL file
    """
    # Determine output directory
    if resume_dir:
        output_dir = resume_dir
        if not output_dir.exists():
            logger.error(f"Resume directory does not exist: {output_dir}")
            return
        logger.info(f"Resuming benchmark from: {output_dir}")
    else:
        # Create timestamped output directory
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        output_dir = output_base_dir / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Benchmark outputs will be saved to: {output_dir}")

    # Find already completed tasks
    completed_task_ids = set()
    for json_file in output_dir.glob("task_*.json"):
        filename = json_file.stem  # "task_3" or "task_3_error"
        if "_error" not in filename:
            try:
                task_id = int(filename.replace("task_", ""))
                completed_task_ids.add(task_id)
                logger.info(f"Found completed task: {task_id}")
            except ValueError:
                pass

    tasks_to_run = QUESTIONS[:1] if test_mode else QUESTIONS

    # Filter out already completed tasks
    pending_tasks = [t for t in tasks_to_run if t["id"] not in completed_task_ids]

    if not pending_tasks:
        logger.info("All tasks already completed!")
        # Still aggregate JSONL in case it wasn't done before
        aggregate_jsonl(output_dir, model_name)
        return

    total_tasks = len(tasks_to_run)
    completed_count = len(completed_task_ids)
    pending_count = len(pending_tasks)

    logger.info(f"Total tasks: {total_tasks}, Already completed: {completed_count}, Pending: {pending_count}")
    logger.info(f"Concurrency: {concurrency}")

    # Track progress with a lock
    progress_lock = asyncio.Lock()
    progress = {"completed": 0}

    sem = asyncio.Semaphore(concurrency)

    async def run_single_task(task: Dict[str, Any]) -> None:
        async with sem:
            task_id = task["id"]
            prompt = task["prompt"]
            language = task["language"]

            async with progress_lock:
                current = completed_count + progress["completed"] + 1

            logger.info(f"[{current}/{total_tasks}] Starting task {task_id} (language: {language})")
            logger.info(f"Prompt preview: {prompt[:100]}...")

            last_error = None
            for attempt in range(1, MAX_RETRIES + 1):
                # Create a temporary directory for this task's raw outputs
                task_temp_dir = output_dir / f"raw_task_{task_id}"
                task_temp_dir.mkdir(parents=True, exist_ok=True)

                try:
                    # Run the research
                    research_result = await run_research(prompt, task_temp_dir, clean_output=True)

                    # Get the final report and sources directly from the returned dict
                    article_content = research_result.get("final_report", "")
                    citations = research_result.get("sources", [])

                    # Create the consolidated result
                    result = {
                        "id": task_id,
                        "language": language,
                        "prompt": prompt,
                        "article": article_content,
                        "citations": citations
                    }

                    # Save to individual JSON file
                    result_json_path = output_dir / f"task_{task_id}.json"
                    with open(result_json_path, "w", encoding="utf-8") as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)

                    async with progress_lock:
                        progress["completed"] += 1

                    logger.info(f"Task {task_id} complete. Saved to: {result_json_path}")
                    last_error = None
                    break  # Success — exit retry loop

                except (TimeoutError, asyncio.TimeoutError) as e:
                    last_error = e
                    if attempt < MAX_RETRIES:
                        logger.warning(f"Task {task_id} timed out (attempt {attempt}/{MAX_RETRIES}), retrying in 30s...")
                        await asyncio.sleep(30)
                    else:
                        logger.error(f"Task {task_id} timed out after {MAX_RETRIES} attempts")

                except Exception as e:
                    last_error = e
                    logger.error(f"Task {task_id} failed (attempt {attempt}/{MAX_RETRIES}): {e}")
                    if attempt < MAX_RETRIES:
                        logger.info(f"Retrying task {task_id} in 30s...")
                        await asyncio.sleep(30)

            # If all retries exhausted, save error result
            if last_error is not None:
                error_result = {
                    "id": task_id,
                    "language": language,
                    "prompt": prompt,
                    "article": f"ERROR: {str(last_error)}",
                    "citations": []
                }
                error_json_path = output_dir / f"task_{task_id}_error.json"
                with open(error_json_path, "w", encoding="utf-8") as f:
                    json.dump(error_result, f, indent=2, ensure_ascii=False)

                async with progress_lock:
                    progress["completed"] += 1

                logger.info(f"Continuing despite error on task {task_id}...")

            # Brief delay to avoid API rate-limiting
            await asyncio.sleep(DELAY_BETWEEN_TASKS)

    # Launch all pending tasks with semaphore-based concurrency
    await asyncio.gather(*(run_single_task(t) for t in pending_tasks))

    logger.info(f"Benchmark complete! Results saved to: {output_dir}")

    # Aggregate into submission JSONL
    jsonl_path = aggregate_jsonl(output_dir, model_name)
    logger.info(f"Submission file: {jsonl_path}")


def get_latest_run_dir(output_base_dir: Path) -> Path | None:
    """Find the most recent benchmark run directory."""
    if not output_base_dir.exists():
        return None

    run_dirs = [d for d in output_base_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        return None

    # Sort by directory name (which is a timestamp)
    run_dirs.sort(key=lambda x: x.name, reverse=True)
    return run_dirs[0]


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run benchmark tasks through the Deep Research Agent"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="benchmark_outputs",
        help="Base directory for benchmark outputs (default: benchmark_outputs)"
    )
    parser.add_argument(
        "--test-mode", "-t",
        action="store_true",
        help="Run only the first task for testing"
    )
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume from the most recent benchmark run, skipping completed tasks"
    )
    parser.add_argument(
        "--resume-dir",
        type=str,
        default=None,
        help="Resume from a specific benchmark run directory (e.g., benchmark_outputs/2026-01-30_19-57-53)"
    )
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=3,
        help="Maximum number of tasks to run in parallel (default: 3)"
    )
    parser.add_argument(
        "--model-name", "-m",
        type=str,
        default="deep_research_agent",
        help="Model name for the output JSONL file (default: deep_research_agent)"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Determine resume directory
    resume_dir = None
    if args.resume_dir:
        resume_dir = Path(args.resume_dir)
    elif args.resume:
        resume_dir = get_latest_run_dir(output_dir)
        if resume_dir:
            logger.info(f"Auto-detected latest run: {resume_dir}")
        else:
            logger.info("No previous run found, starting fresh")

    try:
        asyncio.run(run_benchmark(
            output_dir,
            args.test_mode,
            resume_dir,
            concurrency=args.concurrency,
            model_name=args.model_name,
        ))
        print(f"\nBenchmark complete!")
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
        print(f"\nBenchmark interrupted. Use --resume to continue later.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        print(f"\nBenchmark failed. Use --resume to retry from where it stopped.")
        sys.exit(1)


if __name__ == "__main__":
    main()
