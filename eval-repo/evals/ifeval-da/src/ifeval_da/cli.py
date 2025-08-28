#!/usr/bin/env python3
"""
IFEval-DA CLI - Simple command-line interface for evaluating models on IFEval benchmark.
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import click
import aiohttp
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from tqdm.asyncio import tqdm

from ifeval_da import evaluation_lib_key_based as evaluation_lib

console = Console()


async def detect_model(api_base: str, api_key: Optional[str] = None) -> Optional[str]:
    """Auto-detect model from API endpoint."""
    url = f"{api_base}/models"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get("data", [])
                    if models:
                        model_id = models[0].get("id")
                        console.print(f"[green]Auto-detected model: {model_id}[/green]")
                        return model_id
    except Exception as e:
        console.print(f"[yellow]Could not auto-detect model: {e}[/yellow]")
    
    return None


async def generate_response(
    session: aiohttp.ClientSession,
    prompt: str,
    model: str,
    api_base: str,
    headers: Dict[str, str],
    temperature: float = 0.0,
    max_tokens: int = 2048,
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> Optional[str]:
    """Generate a response using the OpenAI-compatible API."""
    url = f"{api_base}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    for attempt in range(max_retries):
        try:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    error_text = await response.text()
                    raise Exception(f"API error {response.status}: {error_text}")
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (2 ** attempt))
            else:
                return None


async def process_prompt(
    semaphore: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    prompt_data: Dict[str, Any],
    model: str,
    api_base: str,
    headers: Dict[str, str],
    temperature: float,
    max_tokens: int,
    index: int
) -> Dict[str, Any]:
    """Process a single prompt with semaphore control."""
    async with semaphore:
        prompt_text = prompt_data.get("prompt", "")
        
        response = await generate_response(
            session=session,
            prompt=prompt_text,
            model=model,
            api_base=api_base,
            headers=headers,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return {
            "key": prompt_data.get("key", index),
            "instruction_id_list": prompt_data.get("instruction_id_list", []),
            "prompt": prompt_text,
            "response": response if response is not None else "",
            "kwargs": prompt_data.get("kwargs", [])
        }


async def process_dataset(
    prompts: List[Dict[str, Any]],
    model: str,
    api_base: str,
    api_key: Optional[str],
    temperature: float,
    max_tokens: int,
    max_concurrent: int
) -> List[Dict[str, Any]]:
    """Process all prompts concurrently with semaphore control."""
    semaphore = asyncio.Semaphore(max_concurrent)
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    headers["Content-Type"] = "application/json"
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            process_prompt(
                semaphore=semaphore,
                session=session,
                prompt_data=prompt_data,
                model=model,
                api_base=api_base,
                headers=headers,
                temperature=temperature,
                max_tokens=max_tokens,
                index=i
            )
            for i, prompt_data in enumerate(prompts)
        ]
        
        results = []
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating responses"):
            result = await f
            results.append(result)
        
        results.sort(key=lambda x: x["key"])
        return results


def load_prompts(input_file: Path) -> List[Dict[str, Any]]:
    """Load prompts from a JSONL file."""
    prompts = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    return prompts


def save_results(results: List[Dict[str, Any]], output_file: Path):
    """Save results to a JSONL file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


def run_evaluation(response_file: Path, input_file: Path, output_dir: Path):
    """Run evaluation on responses."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    inputs = evaluation_lib.read_prompt_list(str(input_file))
    key_to_responses = evaluation_lib.read_key_to_responses_dict(str(response_file))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Extract model name without timestamp (if response file has timestamp)
    model_name = response_file.stem.replace("responses_", "")
    # Remove timestamp pattern from model name if present
    import re
    model_name = re.sub(r'_\d{8}_\d{6}$', '', model_name)
    
    results_summary = {}
    
    for eval_name, eval_func in [
        ("STRICT", evaluation_lib.test_instruction_following_strict),
        ("LOOSE", evaluation_lib.test_instruction_following_loose),
    ]:
        console.print(f"\n[bold]{eval_name} Evaluation:[/bold]")
        
        eval_results = []
        for prompt_data in inputs:
            result = eval_func(prompt_data, key_to_responses)
            eval_results.append(result)
        
        follow_all = [r.follow_all_instructions for r in eval_results]
        accuracy = sum(follow_all) / len(eval_results) if eval_results else 0
        
        console.print(f"Accuracy: {accuracy:.2%} ({sum(follow_all)}/{len(eval_results)})")
        
        output_file = output_dir / f"eval_results_{model_name}_{eval_name.lower()}_{timestamp}.jsonl"
        evaluation_lib.write_outputs(str(output_file), eval_results)
        console.print(f"[dim]Results saved to: {output_file}[/dim]")
        
        results_summary[eval_name] = {
            "accuracy": accuracy,
            "correct": sum(follow_all),
            "total": len(eval_results)
        }
    
    return results_summary


@click.group()
def cli():
    """IFEval-DA CLI - Evaluate models on instruction following benchmark."""
    pass


@cli.command()
@click.argument('model', required=False)
@click.option('--api-base', default='http://localhost:8000/v1', help='API base URL')
@click.option('--api-key', envvar='OPENAI_API_KEY', help='API key')
@click.option('--sample', type=int, help='Only evaluate on N samples')
@click.option('--concurrent', type=int, default=50, help='Max concurrent requests')
@click.option('--temperature', type=float, default=0.0, help='Generation temperature')
@click.option('--max-tokens', type=int, default=2048, help='Max tokens to generate')
@click.option('--skip-eval', is_flag=True, help='Skip evaluation, only generate responses')
def eval(model, api_base, api_key, sample, concurrent, temperature, max_tokens, skip_eval):
    """Evaluate a model via OpenAI-compatible API."""
    
    async def run():
        # Auto-detect model if not specified
        if not model:
            detected_model = await detect_model(api_base, api_key)
            if not detected_model:
                console.print("[red]Error: Could not detect model and none specified[/red]")
                sys.exit(1)
            model_name = detected_model
        else:
            model_name = model
        
        # Select dataset
        input_file = Path("data/danish.jsonl")
        if not input_file.exists():
            console.print("[red]Error: Danish dataset not found at data/danish.jsonl[/red]")
            sys.exit(1)
        
        # Load prompts
        console.print(f"Loading prompts from {input_file}...")
        prompts = load_prompts(input_file)
        
        if sample:
            prompts = prompts[:sample]
            console.print(f"[yellow]Sampling {sample} prompts[/yellow]")
        
        console.print(f"Loaded {len(prompts)} prompts")
        
        # Generate responses
        console.print(f"Generating responses using [bold]{model_name}[/bold]...")
        console.print(f"API: {api_base}, Concurrency: {concurrent}")
        
        import time
        start_time = time.time()
        
        results = await process_dataset(
            prompts=prompts,
            model=model_name,
            api_base=api_base.rstrip('/'),
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_concurrent=concurrent
        )
        
        elapsed_time = time.time() - start_time
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_safe = model_name.replace("/", "_").replace(":", "_")
        output_file = Path(f"data/responses_{model_name_safe}_{timestamp}.jsonl")
        
        console.print(f"Saving responses to {output_file}...")
        save_results(results, output_file)
        
        # Print statistics
        successful = sum(1 for r in results if r["response"])
        console.print(f"\n[bold]Generation Summary:[/bold]")
        console.print(f"  Total: {len(prompts)}")
        console.print(f"  Successful: {successful}")
        console.print(f"  Failed: {len(prompts) - successful}")
        console.print(f"  Time: {elapsed_time:.2f}s ({len(prompts)/elapsed_time:.2f} prompts/s)")
        
        # Run evaluation
        if not skip_eval:
            console.print("\n" + "=" * 64)
            console.print("[bold]Running evaluation...[/bold]")
            
            output_dir = Path("results")
            summary = run_evaluation(output_file, input_file, output_dir)
            
            # Display summary table
            table = Table(title="\nEvaluation Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Strict", style="magenta")
            table.add_column("Loose", style="magenta")
            
            table.add_row(
                "Accuracy",
                f"{summary['STRICT']['accuracy']:.2%}",
                f"{summary['LOOSE']['accuracy']:.2%}"
            )
            table.add_row(
                "Correct",
                f"{summary['STRICT']['correct']}/{summary['STRICT']['total']}",
                f"{summary['LOOSE']['correct']}/{summary['LOOSE']['total']}"
            )
            
            console.print(table)
    
    asyncio.run(run())


@cli.command()
@click.argument('response_file', type=click.Path(exists=True))
def analyze(response_file):
    """Analyze existing response file."""
    response_path = Path(response_file)
    
    # Select dataset
    input_file = Path("data/danish.jsonl")
    if not input_file.exists():
        console.print("[red]Error: Danish dataset not found at data/danish.jsonl[/red]")
        sys.exit(1)
    
    console.print(f"Analyzing responses from {response_path}...")
    console.print(f"Using dataset: {input_file}")
    
    output_dir = Path("results")
    summary = run_evaluation(response_path, input_file, output_dir)
    
    # Display summary table
    table = Table(title="\nEvaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Strict", style="magenta")
    table.add_column("Loose", style="magenta")
    
    table.add_row(
        "Accuracy",
        f"{summary['STRICT']['accuracy']:.2%}",
        f"{summary['LOOSE']['accuracy']:.2%}"
    )
    table.add_row(
        "Correct",
        f"{summary['STRICT']['correct']}/{summary['STRICT']['total']}",
        f"{summary['LOOSE']['correct']}/{summary['LOOSE']['total']}"
    )
    
    console.print(table)


@cli.command()
@click.argument('file1', type=click.Path(exists=True))
@click.argument('file2', type=click.Path(exists=True))
@click.option('--detailed', is_flag=True, help='Show detailed category comparison')
@click.option('--export', type=click.Path(), help='Export detailed comparison to markdown file')
def compare(file1, file2, detailed, export):
    """Compare two evaluation results."""
    
    def load_eval_results(filepath):
        """Load evaluation results and calculate metrics."""
        results = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        
        follow_all = [r.get("follow_all_instructions", False) for r in results]
        accuracy = sum(follow_all) / len(results) if results else 0
        
        # Calculate category breakdown
        category_stats = {}
        for r in results:
            instruction_ids = r.get("instruction_id_list", [])
            follow_list = r.get("follow_instruction_list", [])
            
            for i, inst_id in enumerate(instruction_ids):
                if i < len(follow_list):
                    category = inst_id.split(":")[0] if ":" in inst_id else inst_id
                    if category not in category_stats:
                        category_stats[category] = {"correct": 0, "total": 0}
                    category_stats[category]["total"] += 1
                    if follow_list[i]:
                        category_stats[category]["correct"] += 1
        
        return {
            "accuracy": accuracy,
            "correct": sum(follow_all),
            "total": len(results),
            "results": results,
            "categories": category_stats
        }
    
    # Load both results
    results1 = load_eval_results(file1)
    results2 = load_eval_results(file2)
    
    # Extract model names from filenames
    model1 = Path(file1).stem.replace("eval_results_", "").rsplit("_", 2)[0]
    model2 = Path(file2).stem.replace("eval_results_", "").rsplit("_", 2)[0]
    
    # Main comparison table
    console.print(f"\n[bold]Comparing Models:[/bold]")
    console.print(f"Model 1: {model1}")
    console.print(f"Model 2: {model2}\n")
    
    table = Table(title="Overall Comparison")
    table.add_column("Model", style="cyan")
    table.add_column("Accuracy", style="magenta")
    table.add_column("Correct", style="green")
    table.add_column("Total", style="yellow")
    
    table.add_row(
        model1[:40],
        f"{results1['accuracy']:.2%}",
        str(results1['correct']),
        str(results1['total'])
    )
    
    table.add_row(
        model2[:40],
        f"{results2['accuracy']:.2%}",
        str(results2['correct']),
        str(results2['total'])
    )
    
    diff = results2['accuracy'] - results1['accuracy']
    diff_style = "[green]" if diff > 0 else "[red]" if diff < 0 else ""
    table.add_row(
        "Difference",
        f"{diff_style}{diff:+.2%}[/]",
        f"{diff_style}{results2['correct'] - results1['correct']:+d}[/]",
        "-"
    )
    
    console.print(table)
    
    # Category comparison
    if detailed and results1['categories'] and results2['categories']:
        console.print("\n[bold]Category Comparison:[/bold]")
        
        cat_table = Table()
        cat_table.add_column("Category", style="cyan", width=20)
        cat_table.add_column(f"{model1[:20]} Acc", style="yellow", justify="right")
        cat_table.add_column(f"{model2[:20]} Acc", style="yellow", justify="right")
        cat_table.add_column("Difference", style="magenta", justify="right")
        
        all_categories = set(results1['categories'].keys()) | set(results2['categories'].keys())
        
        category_diffs = []
        for category in all_categories:
            cat1 = results1['categories'].get(category, {"correct": 0, "total": 0})
            cat2 = results2['categories'].get(category, {"correct": 0, "total": 0})
            
            acc1 = cat1["correct"] / cat1["total"] if cat1["total"] > 0 else 0
            acc2 = cat2["correct"] / cat2["total"] if cat2["total"] > 0 else 0
            diff = acc2 - acc1
            
            category_diffs.append((category, acc1, acc2, diff))
        
        # Sort by absolute difference (biggest changes first)
        category_diffs.sort(key=lambda x: abs(x[3]), reverse=True)
        
        for category, acc1, acc2, diff in category_diffs:
            diff_str = f"{diff:+.1%}" if diff != 0 else "0.0%"
            if diff > 0.05:
                diff_str = f"[green]{diff_str}[/green]"
            elif diff < -0.05:
                diff_str = f"[red]{diff_str}[/red]"
            
            cat_table.add_row(
                category,
                f"{acc1:.1%}",
                f"{acc2:.1%}",
                diff_str
            )
        
        console.print(cat_table)
    
    # Analyze instruction-level changes
    if len(results1['results']) == len(results2['results']):
        improved_prompts = 0
        regressed_prompts = 0
        instruction_improvements = {}
        instruction_regressions = {}
        
        for r1, r2 in zip(results1['results'], results2['results']):
            prompt_improved = not r1.get('follow_all_instructions') and r2.get('follow_all_instructions')
            prompt_regressed = r1.get('follow_all_instructions') and not r2.get('follow_all_instructions')
            
            if prompt_improved:
                improved_prompts += 1
            elif prompt_regressed:
                regressed_prompts += 1
            
            # Track which specific instructions changed
            inst_ids = r1.get("instruction_id_list", [])
            follow1 = r1.get("follow_instruction_list", [])
            follow2 = r2.get("follow_instruction_list", [])
            
            for i, inst_id in enumerate(inst_ids):
                if i < len(follow1) and i < len(follow2):
                    if not follow1[i] and follow2[i]:
                        # This instruction improved
                        instruction_improvements[inst_id] = instruction_improvements.get(inst_id, 0) + 1
                    elif follow1[i] and not follow2[i]:
                        # This instruction regressed
                        instruction_regressions[inst_id] = instruction_regressions.get(inst_id, 0) + 1
        
        # Show net change summary
        net_change = improved_prompts - regressed_prompts
        console.print(f"\n[bold]Prompt-level Changes:[/bold]")
        
        change_style = "[green]" if net_change > 0 else "[red]" if net_change < 0 else "[yellow]"
        console.print(f"  Improved: [green]+{improved_prompts}[/green]")
        console.print(f"  Regressed: [red]-{regressed_prompts}[/red]")
        console.print(f"  Net change: {change_style}{net_change:+d}[/]")
        
        # Show specific instruction types only if detailed (since categories are already shown above)
        if detailed and (instruction_improvements or instruction_regressions):
            # Only show if there are specific instruction subtypes (with colons)
            specific_changes = []
            for inst_id, count in instruction_improvements.items():
                if ":" in inst_id:  # Only specific subtypes
                    net = count - instruction_regressions.get(inst_id, 0)
                    if net != 0:
                        specific_changes.append((inst_id, net))
            for inst_id, count in instruction_regressions.items():
                if ":" in inst_id and inst_id not in instruction_improvements:
                    specific_changes.append((inst_id, -count))
            
            if specific_changes:
                specific_changes.sort(key=lambda x: abs(x[1]), reverse=True)
                console.print(f"\n[bold]Specific Instruction Changes:[/bold]")
                console.print("Most significant changes by specific instruction type:")
                
                # Show top 6 specific changes
                for inst_id, change in specific_changes[:6]:
                    if change > 0:
                        console.print(f"  [green]↑ {inst_id}: +{change}[/green]")
                    else:
                        console.print(f"  [red]↓ {inst_id}: {change}[/red]")
                
                if len(specific_changes) > 6:
                    console.print(f"  [dim]... and {len(specific_changes) - 6} more specific types[/dim]")
    
    if not detailed:
        console.print("\n[dim]Use --detailed to see category-level comparison[/dim]")
    
    # Export detailed comparison if requested
    if export and len(results1['results']) == len(results2['results']):
        console.print(f"\n[yellow]Exporting detailed comparison to {export}...[/yellow]")
        
        with open(export, 'w', encoding='utf-8') as f:
            f.write(f"# Model Comparison Report\n\n")
            f.write(f"**Model 1:** {model1}\n")
            f.write(f"**Model 2:** {model2}\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            
            f.write(f"## Summary\n\n")
            f.write(f"- Model 1 Accuracy: {results1['accuracy']:.2%} ({results1['correct']}/{results1['total']})\n")
            f.write(f"- Model 2 Accuracy: {results2['accuracy']:.2%} ({results2['correct']}/{results2['total']})\n")
            f.write(f"- Difference: {diff:+.2%}\n\n")
            
            # Write improved prompts with details
            improved_details = []
            regressed_details = []
            
            for r1, r2 in zip(results1['results'], results2['results']):
                prompt_text = r1.get('prompt', '')
                if not r1.get('follow_all_instructions') and r2.get('follow_all_instructions'):
                    # Improved
                    improved_details.append({
                        'prompt': prompt_text,
                        'instructions': r1.get('instruction_id_list', []),
                        'model1_failed': [inst for i, inst in enumerate(r1.get('instruction_id_list', [])) 
                                        if i < len(r1.get('follow_instruction_list', [])) and not r1['follow_instruction_list'][i]],
                        'response1': r1.get('response', ''),
                        'response2': r2.get('response', '')
                    })
                elif r1.get('follow_all_instructions') and not r2.get('follow_all_instructions'):
                    # Regressed
                    regressed_details.append({
                        'prompt': prompt_text,
                        'instructions': r2.get('instruction_id_list', []),
                        'model2_failed': [inst for i, inst in enumerate(r2.get('instruction_id_list', [])) 
                                        if i < len(r2.get('follow_instruction_list', [])) and not r2['follow_instruction_list'][i]],
                        'response1': r1.get('response', ''),
                        'response2': r2.get('response', '')
                    })
            
            # Write improvements
            if improved_details:
                f.write(f"## Improved Prompts ({len(improved_details)})\n\n")
                for i, item in enumerate(improved_details[:10], 1):  # Show first 10
                    f.write(f"### {i}. Improvement\n\n")
                    f.write(f"**Prompt:** {item['prompt'][:200]}{'...' if len(item['prompt']) > 200 else ''}\n\n")
                    f.write(f"**Instructions that Model 1 failed:** {', '.join(item['model1_failed'])}\n\n")
                    f.write(f"**Model 1 Response:**\n```\n{item['response1'][:500]}{'...' if len(item['response1']) > 500 else ''}\n```\n\n")
                    f.write(f"**Model 2 Response (Correct):**\n```\n{item['response2'][:500]}{'...' if len(item['response2']) > 500 else ''}\n```\n\n")
                    f.write("---\n\n")
                
                if len(improved_details) > 10:
                    f.write(f"*... and {len(improved_details) - 10} more improvements*\n\n")
            
            # Write regressions
            if regressed_details:
                f.write(f"## Regressed Prompts ({len(regressed_details)})\n\n")
                for i, item in enumerate(regressed_details[:10], 1):  # Show first 10
                    f.write(f"### {i}. Regression\n\n")
                    f.write(f"**Prompt:** {item['prompt'][:200]}{'...' if len(item['prompt']) > 200 else ''}\n\n")
                    f.write(f"**Instructions that Model 2 failed:** {', '.join(item['model2_failed'])}\n\n")
                    f.write(f"**Model 1 Response (Correct):**\n```\n{item['response1'][:500]}{'...' if len(item['response1']) > 500 else ''}\n```\n\n")
                    f.write(f"**Model 2 Response:**\n```\n{item['response2'][:500]}{'...' if len(item['response2']) > 500 else ''}\n```\n\n")
                    f.write("---\n\n")
                
                if len(regressed_details) > 10:
                    f.write(f"*... and {len(regressed_details) - 10} more regressions*\n\n")
        
        console.print(f"[green]Detailed comparison exported to {export}[/green]")


@cli.command()
@click.argument('result_file', type=click.Path(exists=True), required=False)
@click.option('--latest', is_flag=True, help='Show only the latest result')
def results(result_file, latest):
    """List evaluation results or show detailed breakdown of a specific result.
    
    Examples:
        ifeval results                    # List all results
        ifeval results --latest            # Show latest result
        ifeval results path/to/result.jsonl  # Show detailed breakdown
    """
    results_dir = Path("results")
    
    if not results_dir.exists():
        console.print("[yellow]No results directory found[/yellow]")
        return
    
    # Find all result files
    result_files = sorted(results_dir.glob("eval_results_*.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not result_files:
        console.print("[yellow]No evaluation results found[/yellow]")
        return
    
    def parse_filename(filename):
        """Extract model name, evaluation type, and timestamp from filename."""
        # Pattern: eval_results_{model}_{strict/loose}_{timestamp}.jsonl
        # Handle both Path objects and strings
        if hasattr(filename, 'stem'):
            stem = filename.stem
        else:
            stem = filename.replace('.jsonl', '')
        parts = stem.replace("eval_results_", "").rsplit("_", 3)
        if len(parts) >= 3:
            # Check if second-to-last is strict/loose
            if parts[-3].lower() in ["strict", "loose"]:
                model = "_".join(parts[:-3])
                eval_type = parts[-3]
                timestamp = f"{parts[-2]}_{parts[-1]}"
            else:
                model = "_".join(parts[:-2])
                eval_type = "unknown"
                timestamp = f"{parts[-2]}_{parts[-1]}"
        else:
            model = "_".join(parts[:-1]) if len(parts) > 1 else parts[0]
            eval_type = "unknown"
            timestamp = parts[-1] if parts else ""
        return model, eval_type, timestamp
    
    def load_accuracy(filepath):
        """Load file and calculate accuracy."""
        results = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        follow_all = [r.get("follow_all_instructions", False) for r in results]
        accuracy = sum(follow_all) / len(results) if results else 0
        return accuracy, sum(follow_all), len(results), results
    
    # Determine which file(s) to show details for
    if result_file:
        # User specified a specific file
        target_file = Path(result_file)
        if not target_file.exists():
            console.print(f"[red]File not found: {result_file}[/red]")
            return
        detailed = True
        # Find matching strict/loose pair if exists
        stem = target_file.stem
        if "_strict_" in stem:
            loose_stem = stem.replace("_strict_", "_loose_")
            loose_file = target_file.parent / f"{loose_stem}.jsonl"
            files_to_analyze = [(target_file, "strict"), (loose_file, "loose") if loose_file.exists() else None]
        elif "_loose_" in stem:
            strict_stem = stem.replace("_loose_", "_strict_")
            strict_file = target_file.parent / f"{strict_stem}.jsonl"
            files_to_analyze = [(strict_file, "strict") if strict_file.exists() else None, (target_file, "loose")]
        else:
            files_to_analyze = [(target_file, "unknown")]
        files_to_analyze = [f for f in files_to_analyze if f is not None]
    elif latest:
        detailed = True
        # Show details of latest result(s)
        # Group by timestamp to show both strict and loose together
        timestamp_groups = {}
        for f in result_files[:10]:  # Look at recent files
            model, eval_type, timestamp = parse_filename(f)
            if timestamp not in timestamp_groups:
                timestamp_groups[timestamp] = {}
            timestamp_groups[timestamp][eval_type.lower()] = (f, model)
        
        # Get the most recent timestamp group
        if timestamp_groups:
            latest_timestamp = sorted(timestamp_groups.keys(), reverse=True)[0]
            group = timestamp_groups[latest_timestamp]
            files_to_analyze = []
            if 'strict' in group:
                files_to_analyze.append((group['strict'][0], 'strict'))
            if 'loose' in group:
                files_to_analyze.append((group['loose'][0], 'loose'))
            model_name = group.get('strict', group.get('loose', (None, 'unknown')))[1]
        else:
            console.print("[yellow]No results found[/yellow]")
            return
    else:
        detailed = False
    
    # Show detailed analysis if requested
    if detailed:
        # Extract model name from first file
        if result_file:
            model_name, _, timestamp = parse_filename(files_to_analyze[0][0])
        console.print(f"\n[bold]Model: {model_name}[/bold]")
        if not result_file:
            console.print(f"Timestamp: {latest_timestamp}")
            
        # Show both strict and loose results
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Evaluation", style="cyan")
        table.add_column("Accuracy", style="green")
        table.add_column("Correct", style="yellow")
        
        strict_results_data = None
        for filepath, eval_type in files_to_analyze:
            accuracy, correct, total, results_data = load_accuracy(filepath)
            table.add_row(
                eval_type.upper(),
                f"{accuracy:.2%}",
                f"{correct}/{total}"
            )
            if eval_type == 'strict':
                strict_results_data = results_data
        
        console.print(table)
        
        # Show detailed breakdown
        if strict_results_data:
            category_counts = {}
            instruction_type_counts = {}
            
            for r in strict_results_data:
                instruction_ids = r.get("instruction_id_list", [])
                follow_list = r.get("follow_instruction_list", [])
                
                # Make sure we have matching lengths
                for i, inst_id in enumerate(instruction_ids):
                    if i < len(follow_list):
                        # Extract category from instruction_id (e.g., "keywords:existence" -> "keywords")
                        category = inst_id.split(":")[0] if ":" in inst_id else inst_id
                        
                        # Track by category
                        if category not in category_counts:
                            category_counts[category] = {"correct": 0, "total": 0}
                        category_counts[category]["total"] += 1
                        if follow_list[i]:
                            category_counts[category]["correct"] += 1
                        
                        # Track by full instruction type
                        if inst_id not in instruction_type_counts:
                            instruction_type_counts[inst_id] = {"correct": 0, "total": 0}
                        instruction_type_counts[inst_id]["total"] += 1
                        if follow_list[i]:
                            instruction_type_counts[inst_id]["correct"] += 1
            
            if category_counts:
                console.print("\n[bold]Instruction Category Breakdown (Strict):[/bold]")
                
                cat_table = Table(show_header=True)
                cat_table.add_column("Category", style="cyan", width=20)
                cat_table.add_column("Accuracy", style="green", justify="right")
                cat_table.add_column("Correct/Total", style="yellow", justify="right")
                
                # Sort by accuracy (worst first)
                sorted_categories = sorted(
                    category_counts.items(),
                    key=lambda x: x[1]["correct"] / x[1]["total"] if x[1]["total"] > 0 else 0
                )
                
                for category, counts in sorted_categories:
                    acc = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
                    cat_table.add_row(
                        category,
                        f"{acc:.1%}",
                        f"{counts['correct']}/{counts['total']}"
                    )
                console.print(cat_table)
                
                # Show worst performing specific instruction types
                console.print("\n[bold]Hardest Specific Instructions:[/bold]")
                
                inst_table = Table(show_header=True)
                inst_table.add_column("Instruction Type", style="cyan", width=35)
                inst_table.add_column("Accuracy", style="green", justify="right")
                inst_table.add_column("Count", style="yellow", justify="right")
                
                # Sort by accuracy (worst first) and show top 10
                sorted_instructions = sorted(
                    instruction_type_counts.items(),
                    key=lambda x: x[1]["correct"] / x[1]["total"] if x[1]["total"] > 0 else 0
                )[:10]
                
                for inst_type, counts in sorted_instructions:
                    acc = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
                    inst_table.add_row(
                        inst_type,
                        f"{acc:.1%}",
                        f"{counts['total']}"
                    )
                console.print(inst_table)
    
    else:
        # List all results grouped by model
        model_results = {}
        for f in result_files:
            model, eval_type, timestamp = parse_filename(f)
            accuracy, correct, total, _ = load_accuracy(f)
            
            if model not in model_results:
                model_results[model] = {"strict": None, "loose": None, "timestamp": timestamp}
            
            # Keep only the latest result for each model/type combination
            if eval_type.lower() in ["strict", "loose"]:
                if model_results[model][eval_type.lower()] is None or timestamp > model_results[model]["timestamp"]:
                    model_results[model][eval_type.lower()] = accuracy
                    model_results[model]["timestamp"] = timestamp
        
        # Create summary table
        table = Table(title="Evaluation Results Summary")
        table.add_column("Model", style="cyan")
        table.add_column("Strict", style="green")
        table.add_column("Loose", style="green")
        table.add_column("Date", style="yellow")
        
        for model, results in sorted(model_results.items(), key=lambda x: x[1]["timestamp"], reverse=True)[:20]:
            timestamp = results["timestamp"]
            # Parse timestamp (YYYYMMDD_HHMMSS)
            if len(timestamp) == 15 and "_" in timestamp:
                date_part = timestamp.split("_")[0]
                date_str = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
            else:
                date_str = timestamp[:10] if len(timestamp) >= 10 else timestamp
            
            table.add_row(
                model[:30],  # Truncate long model names
                f"{results['strict']:.1%}" if results['strict'] is not None else "-",
                f"{results['loose']:.1%}" if results['loose'] is not None else "-",
                date_str
            )
        
        console.print(table)
        
        if len(model_results) > 20:
            console.print(f"\n[dim]Showing 20 most recent models out of {len(model_results)} total[/dim]")
        
        console.print("\n[dim]Use --latest to see the most recent result in detail[/dim]")
        console.print("[dim]Use --detailed to see category breakdown[/dim]")


if __name__ == "__main__":
    cli()