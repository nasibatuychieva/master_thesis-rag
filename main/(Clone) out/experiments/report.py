import typing as t
from collections import defaultdict
from pathlib import Path
import json
from copilot_rag.pipelines.log import Logger
from copilot_rag.tools.llm_judge.evaluator import RAGEvaluator
from copilot_rag.tools.llm_judge.common import EvaluationResult


def create_evaluation_report(
        evaluator: RAGEvaluator,
        evaluations: t.List[EvaluationResult]
) -> dict:
    """Create a report from a list of evaluations by computing averages.

    Args:
        evaluations: List of evaluation dictionaries, where each evaluation
        contains metrics, durations, and document_precision

    Returns:
        Dictionary containing averaged metrics, durations, and document
        precision
    """
    if not evaluations:
        return {}

    duration_sums = defaultdict(float)
    duration_counts = defaultdict(int)
    doc_precision_sum = 0
    doc_precision_count = 0
    first_page_precision_sum = 0
    last_page_precision_sum = 0
    page_precision_count = 0

    procedure_evaluations = [
        e for e in evaluations
        if e.category == "procedure_description"
    ]
    service_request_evaluations = [
        e for e in evaluations
        if e.category == "service_request"
    ]

    for evaluation in evaluations:
        for component, elapsed_time in evaluation.durations.items():
            duration_sums[component] += elapsed_time
            duration_counts[component] += 1

        if evaluation.document_precision is not None:
            doc_precision_sum += evaluation.document_precision
            doc_precision_count += 1

        # Aggregate page precision
        if evaluation.page_precision:
            first_page_precision_sum += evaluation.page_precision.get(
                'first_page_precision', 0
            )
            last_page_precision_sum += evaluation.page_precision.get(
                'last_page_precision', 0
            )
            page_precision_count += 1

    duration_averages = {
        component: duration_sums[component] / duration_counts[component]
        for component in duration_sums
    }

    doc_precision_avg = (
        doc_precision_sum / doc_precision_count
        if doc_precision_count > 0 else None
    )

    # Calculate average page precision
    first_page_precision_avg = (
        first_page_precision_sum / page_precision_count
        if page_precision_count > 0 else None
    )
    last_page_precision_avg = (
        last_page_precision_sum / page_precision_count
        if page_precision_count > 0 else None
    )

    overall_metrics = evaluator.aggregate_metrics(evaluations)

    procedure_metrics = evaluator.aggregate_metrics_by_category(
        procedure_evaluations
    )
    service_request_metrics = evaluator.aggregate_metrics_by_category(
        service_request_evaluations
    )

    procedure_metrics.update({
        "count": len(procedure_evaluations),
        "type": "procedure_description",
    })

    service_request_metrics.update({
        "count": len(service_request_evaluations),
        "type": "service_request",
    })

    # Calculate summary metrics
    summary = {
        "total_questions": len(evaluations),
        "evaluations_completed": len(evaluations),
        "completion_rate": {
            "total": len(evaluations),
            "completed": len(evaluations)
        },
        "procedures_count": len(procedure_evaluations),
        "service_requests_count": len(service_request_evaluations),
        "document_precision_stats": {
            "cases_included": doc_precision_count,
            "cases_excluded": len(evaluations) - doc_precision_count,
            "total_cases": len(evaluations),
            "inclusion_rate": (doc_precision_count / len(evaluations) * 100) if len(evaluations) > 0 else 0.0  # noqa: E501
        }
    }

    return {
        "overall_metrics": overall_metrics,
        "summary": summary,
        "document_precision": doc_precision_avg,
        "first_page_precision": first_page_precision_avg,
        "last_page_precision": last_page_precision_avg,
        "procedure_metrics": procedure_metrics,
        "service_request_metrics": service_request_metrics,
        "durations": duration_averages,

    }


def _is_safe_path(base_path: Path, path: Path) -> bool:
    """Check if the path is safe (no directory traversal).

    :param base_path: The allowed base directory.
    :param path: The path to check.
    :return: True if path is safe, False otherwise.
    """
    try:
        # Resolve any symlinks and normalize path
        real_base = base_path.resolve()
        real_path = path.resolve()
        # Check if the path is within the base directory
        return real_base in real_path.parents
    except (RuntimeError, ValueError):
        return False


def save_evaluation_report(
    evaluations: t.List[EvaluationResult],
    report_summary: dict,
    output_dir: str
) -> None:
    """Save evaluation results to JSON files.

    Creates JSON files for metrics, durations and summary report.

    :param evaluations: List of evaluation results
    :param report_summary: Dictionary containing the computed averages
    :param output_dir: Directory where to save the reports
    """
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    Logger.info(
        "Saving evaluation report",
        output_path=output_path
    )

    # Save answers.
    answers_path = output_path / "answers.json"
    if not _is_safe_path(output_path, answers_path):
        raise ValueError("Invalid path for answers.json")
    with open(answers_path, "w") as f:
        json.dump({
            res.case_id: res.actual_answer
            for res in evaluations
        }, f, indent=2)

    # Save summary report
    summary_path = output_path / "summary.json"
    if not _is_safe_path(output_path, summary_path):
        raise ValueError("Invalid path for summary.json")
    with open(summary_path, "w") as f:
        json.dump(report_summary, f, indent=2)

    # Create consolidated metrics data
    consolidated_metrics = {}
    for evaluation in evaluations:
        # Extract correctness category if available
        correctness_category = None
        if 'correctness' in evaluation.metrics and isinstance(evaluation.metrics['correctness'], dict):  # noqa: E501
            if 'category' in evaluation.metrics['correctness']:
                correctness_category = evaluation.metrics['correctness']['category'].lower()  # noqa: E501

        # Calculate completeness precision if available
        completeness_precision = None
        total_expected_cause = None
        if 'completeness' in evaluation.metrics and isinstance(evaluation.metrics['completeness'], dict):  # noqa: E501
            completeness = evaluation.metrics['completeness']

            total_expected_cause = completeness.get(
                'total_expected_cause_count', None
            )

            total_steps = completeness.get('total_actual_steps', 0)
            wrong_step_count = completeness.get('wrong_step_count', 0)

            if total_steps > 0:
                steps_precision = (
                    total_steps - wrong_step_count
                    ) / total_steps
            else:
                steps_precision = 0.0

            total_causes = completeness.get('total_actual_cause_count', 0)
            wrong_causes = completeness.get('wrong_cause_count', 0)

            if total_causes > 0:
                causes_precision = (total_causes - wrong_causes) / total_causes
            else:
                causes_precision = 0.0

            completeness_precision = {
                "steps_precision": steps_precision,
                "causes_precision": causes_precision
            }

        consolidated_metrics[evaluation.case_id] = {
            "expected_answer": evaluation.expected_answer,
            "actual_answer": evaluation.actual_answer,
            "metrics": evaluation.metrics,
            "correctness_category": correctness_category,
            "completeness_precision": completeness_precision,
            "total_expected_cause": total_expected_cause,
            "expected_docs": evaluation.expected_docs,
            "retrieved_docs": evaluation.retrieved_documents,
            "document_precision": evaluation.document_precision,
            "document_omission_rate": evaluation.document_omission_rate,
            "page_precision": evaluation.page_precision,
            "expected_pages": evaluation.expected_pages,
            "actual_pages": evaluation.actual_pages,
            "durations": evaluation.durations,
            "machine_type": evaluation.machine_type,
            "category": evaluation.category,
        }

    # Save consolidated metrics
    consolidated_path = output_path / "consolidated_metrics.json"
    if not _is_safe_path(output_path, consolidated_path):
        raise ValueError("Invalid path for consolidated_metrics.json")
    with open(consolidated_path, "w") as f:
        json.dump(consolidated_metrics, f, indent=2)

    # Collect and save metric data
    metric_data = defaultdict(list)
    for evaluation in evaluations:
        for metric_name, dimensions in evaluation.metrics.items():
            # Sanitize metric name to prevent path traversal
            safe_metric_name = "".join(
                c for c in metric_name if c.isalnum() or c in "_-"
            )
            metric_data[safe_metric_name].append({
                "case_id": evaluation.case_id,
                "dimensions": dimensions
            })

    # Save each metric to its own JSON file
    for metric_name, values in metric_data.items():
        metric_path = output_path / f"{metric_name}.json"
        if not _is_safe_path(output_path, metric_path):
            Logger.warning(
                "Skipping unsafe metric file",
                metric_name=metric_name
            )
            continue
        with open(metric_path, "w") as f:
            json.dump(values, f, indent=2)

    # Save durations
    duration_records = []
    for evaluation in evaluations:
        for component, elapsed_time in evaluation.durations.items():
            duration_records.append({
                "case_id": evaluation.case_id,
                "component": component,
                "elapsed_time": elapsed_time
            })

    if duration_records:
        duration_path = output_path / "durations.json"
        if not _is_safe_path(output_path, duration_path):
            raise ValueError("Invalid path for durations.json")
        with open(duration_path, "w") as f:
            json.dump(duration_records, f, indent=2)
