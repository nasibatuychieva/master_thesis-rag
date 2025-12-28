import re
import json
import httpx
import asyncio
import traceback
import argparse
from copy import deepcopy
from datetime import datetime


from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv, find_dotenv
from collections import defaultdict


project_root = Path(__file__).resolve().parent.parent.parent.parent
env_path = find_dotenv()
if not Path(env_path).exists():
    raise FileNotFoundError(
        f"Environment file not found at {env_path}. "
        "Please ensure .env file exists in the project root directory."
    )

load_dotenv(env_path)

parser = argparse.ArgumentParser()
parser.add_argument("--test_set", type=str, required=True)
parser.add_argument("--output_dir", type=str, default=".", required=False)
parser.add_argument(
    "--division",
    type=str,
    help="Filter test cases by division name (e.g. 'ft', 'fht')",
    required=False
)
parser.add_argument(
    "--skip_evaluation",
    action="store_true",
    help="Skip evaluation and only output raw results with retrieved documents"
)
parser.add_argument(
    "--output_format",
    type=str,
    choices=["json", "csv"],
    default="json",
    help="Output format for raw results when skip_evaluation is true"
)
parser.add_argument(
    "--generated_qa_set",
    action="store_true",
    help="Use generated Q&A format test set (id, question, machine_group)"
)
parser.add_argument(
    "--division_id",
    type=int,
    default=1,  # FT
    help="The ID of the division which owns the testset."
)
parser.add_argument(
    "--division_name",
    type=str,
    default="ft",
    help="The abbreviated division name."
)
parser.add_argument(
    "--api_version",
    type=str,
    default="1.0.0",
    help="The api version to test."
)
parser.add_argument(
    "--relaxed",
    action="store_true",
    help="Use relaxed TP/FP criteria (current definition). When not set, "
    "uses strict criteria where TP requires all expected steps/causes to be "
    "present."
)
args = parser.parse_args()


import pandas as pd  # noqa: E402

from copilot_rag.config import DatabricksOptions, PipelineOptions  # noqa: E402
from copilot_rag.config import (  # noqa: E402
    CopilotDatabricksSettings,  # noqa: E402
    CopilotPipelineSettings  # noqa: E402
)
from copilot_rag.pipelines.service_request import SupportQnAPipeline  # noqa: E402 E501

from copilot_rag.pipelines.log import Logger  # noqa: E402
from copilot_rag.pipelines.tracer import PipelineRunTracer  # noqa: E402
from copilot_rag.tools.llm_judge.common import TestCase  # noqa: E402
from copilot_rag.tools.llm_judge.evaluator import RAGEvaluator  # noqa: E402
from copilot.orm import get_session  # noqa: E402
from copilot_rag.databricks import DatabricksServicePrincipal  # noqa: E402
from copilot_rag.tools.llm_judge.report import (  # noqa: E402
    create_evaluation_report,  # noqa: E402
    save_evaluation_report  # noqa: E402
)  # noqa: E402


class CombinedSettings(CopilotDatabricksSettings, CopilotPipelineSettings):
    """Combined settings class that includes both Databricks and Pipeline settings."""  # noqa: E501
    pass


# Machine type groups mapping
MACHINE_TYPE_GROUPS = {
    "DairyRobot": ["DairyRobot 9500", "DairyRobot 9650", "Monobox"],
    "DPQ": ["DPQ"],
    "MIOne": ["MIOne"],
    "DairyFeed": ["DairyFeed 4500"],
    "ProManure": ["ProManure"],
    "AutorRotor": ["AutoRotor -  Global90/70", "AutorRotor - Performer", "AutorRotor - T8800", "AutorRotor - T8900", "AutorRotor - Magnum 40"],  # noqa: E501
    "Herignbone": ["Herignbone - Challenger 40", "Herignbone - Challenger 45", "Herignbone - Euroclass 800", "Herignbone - Euroclass 1200", "Herignbone - Global 45", "Herignbone - Magnum 40"],  # noqa: E501
    "SbS": ["SbS - Challenger 80", "SbS - Comfort Top", "SbS - Global 90i", "SbS - Global 90VL", "SbS - Johanna", "SbS - Magnum 90i", "SbS - Magnum 90VL"],  # noqa: E501
    "RMA": ["RMA"],
    "Barn Equipment": ["Barn Equipment"],
    "Cooling Equipment": ["Cooling Equipment"],
    "Software/Herd Management": ["Software/Herd Management"]
}


def normalize_newlines(text):
    """Replace multiple consecutive newlines with a single newline."""
    if not text:
        return text
    return re.sub(r'\n{2,}', '\n', text)


async def load_test_cases(
    file_path: Path,
    is_generated_qa: bool = False
) -> List[TestCase] | Dict[str, List[TestCase]]:
    """Load and parse questions from CSV file.

    Args:
        file_path: Path to the CSV file
        is_generated_qa: If True, expects generated Q&A format
                        (id, question, machine_group).
                        If False, expects the evaluation format.
    """
    # Validate file path to prevent path traversal
    safe_path = Path(file_path).resolve()
    expected_parent = project_root
    if not safe_path.exists():
        raise FileNotFoundError(f"Question file not found: {safe_path}")

    if (expected_parent not in safe_path.parents and
            expected_parent != safe_path):
        raise ValueError(
            f"File path {safe_path} "
            f"is outside of project root {expected_parent}"
        )

    try:
        df = pd.read_csv(
            safe_path,
            delimiter=';',
            quotechar='"',
            on_bad_lines='warn',
            encoding='utf-8',
        )

        if df.empty:
            raise ValueError(f"No questions found in {safe_path}")

        cases = []

        if is_generated_qa:
            # Validate required columns
            required_columns = ["id", "question", "machine_group"]
            missing_columns = [
                col for col in required_columns if col not in df.columns
            ]
            if missing_columns:
                raise ValueError(
                    "Generated Q&A format requires columns:",
                    required_columns=required_columns,
                    missing_columns=missing_columns
                )

            for _, row in df.iterrows():
                test_case = TestCase(
                    case_id=str(row["id"]),
                    machine_type=row["machine_group"],
                    question=str(row["question"]),
                    expected_answers=[],
                    expected_docs="",
                    expected_pages="",
                    category="service_request"
                )
                cases.append(test_case)

                Logger.info(
                    f"Loaded generated Q&A question {test_case.case_id}",
                    machine_group=test_case.machine_type,
                    question=test_case.question
                )
        else:
            # Validate required columns for evaluation format
            required_columns = ["No.", "Issue", "Machine Typ"]
            missing_columns = [
                col for col in required_columns if col not in df.columns
            ]
            if missing_columns:
                raise ValueError(
                    "Evaluation format requires columns:",
                    required_columns=required_columns,
                    missing_columns=missing_columns
                )

            # Original evaluation format loading
            for idx, row in df.iterrows():
                try:
                    if pd.notna(row["No."]):
                        # Handle step number, defaulting to 1 if NaN
                        try:
                            step = (
                                1 if "Step" not in row or pd.isna(row["Step"])
                                else int(row["Step"])
                            )
                            if step < 1:
                                raise ValueError(
                                    f"Step number must be positive, got {step}"
                                )
                        except (ValueError, TypeError) as e:
                            Logger.error(
                                f"Invalid step number at row {idx}: {str(e)}"
                            )
                            continue

                        expected_docs = (
                            normalize_newlines(
                                str(row["Expected Documents"])
                            ) if pd.notna(row["Expected Documents"])
                            else ""
                        )
                        expected_pages = (
                            normalize_newlines(
                                str(row["Expected pages"])
                            ) if pd.notna(row["Expected pages"])
                            else ""
                        )
                        expected_causes = (
                            normalize_newlines(
                                str(row["Expected no of Causes"])
                            ) if pd.notna(row["Expected no of Causes"])
                            else ""
                        )
                        expected_answer = (
                            normalize_newlines(
                                str(row["Expected No of Workarounds"])
                            ) if pd.notna(row["Expected No of Workarounds"])
                            else ""
                        )

                        machine_types = (
                            row["Machine Typ"].split("\n")
                            if pd.notna(row["Machine Typ"])
                            else [""]
                        )

                        machine_types = [
                            mt.split("=")[0].strip()
                            for mt in machine_types
                            if mt.strip()
                        ]

                        if "Category" in row and pd.notna(row["Category"]):
                            category = str(row["Category"]).strip().lower()

                        for machine_type in machine_types:
                            test_case = TestCase(
                                category=category,
                                case_id=f"{int(row['No.'])}_{machine_type}",
                                step=step,
                                machine_type=machine_type,
                                question=(
                                    normalize_newlines(
                                        str(row["Issue"])
                                    ) if pd.notna(row["Issue"])
                                    else ""
                                ),
                                expected_answers=(
                                    [expected_answer] if not expected_causes
                                    else [expected_answer, expected_causes]
                                ),
                                expected_docs=expected_docs,
                                expected_pages=expected_pages
                            )
                            cases.append(test_case)

                            Logger.info(
                                "Loaded evaluation format question",
                                test_case_id=test_case.case_id,
                                machine_type=test_case.machine_type,
                                category=test_case.category,
                                question=test_case.question,
                                expected_answers=test_case.expected_answers,
                                expected_docs=test_case.expected_docs,
                                expected_pages=test_case.expected_pages
                            )
                except Exception as e:
                    Logger.error(
                        "Error loading question",
                        row=idx,
                        error=str(e)
                    )
                    traceback.print_exc()
                    exit()

        if not cases:
            raise ValueError(
                f"No valid questions could be loaded from {safe_path}"
            )

        # For evaluation format, group test cases by case_id and sort by step
        if not is_generated_qa:
            case_groups = defaultdict(list)
            for case in cases:
                case_groups[case.case_id].append(case)

            # Sort each group by step number
            for case_id in case_groups:
                case_groups[case_id].sort(key=lambda x: x.step)

            # Count single-step and multi-step cases
            single_step_cases = sum(
                1 for cases in case_groups.values()
                if len(cases) == 1
            )
            multi_step_cases = sum(
                1 for cases in case_groups.values()
                if len(cases) > 1
            )
            total_steps = sum(len(cases) for cases in case_groups.values())

            Logger.info(
                "Question loading complete",
                total_cases=len(case_groups),
                single_step_cases=single_step_cases,
                multi_step_cases=multi_step_cases,
                total_steps=total_steps,
                format_type="evaluation",
                machine_types=list(set(
                    q.machine_type
                    for cases in case_groups.values()
                    for q in cases
                ))
            )
        else:
            Logger.info(
                "Question loading complete",
                total_questions=len(cases),
                format_type="generated_qa",
                machine_types=list(set(q.machine_type for q in cases))
            )

        return cases

    except Exception as e:
        Logger.error(f"Error reading CSV file: {str(e)}")
        try:
            with open(safe_path, 'r', encoding='utf-8') as f:
                first_lines = ''.join(f.readlines()[:5])
                Logger.error(f"First few lines of the file:\n{first_lines}")
        except Exception as read_error:
            Logger.error(
                f"Could not read file for debugging: {str(read_error)}"
            )
        raise


def expand_machine_groups(test_cases: List[TestCase]) -> List[TestCase]:
    """Expand test cases with machine groups into individual machine types."""
    expanded_cases = []

    for case in test_cases:
        # Check if the machine_type is actually a group
        if case.machine_type in MACHINE_TYPE_GROUPS:
            # Create a new test case for each machine type in the group
            for machine_type in MACHINE_TYPE_GROUPS[case.machine_type]:
                new_case = deepcopy(case)
                new_case.machine_type = machine_type
                new_case.case_id = (
                    f"{case.case_id}_{machine_type.replace(' ', '_')}"
                )
                expanded_cases.append(new_case)
        else:
            # If it's not a group, keep the original case
            expanded_cases.append(case)

    return expanded_cases


async def run_evaluation(args: argparse.Namespace):
    """Run the evaluation process."""
    Logger.info("Starting RAG evaluation")

    if not args.test_set:
        script_dir = Path(__file__).parent
        test_set_path = script_dir / "generated_qa_dataset.csv"
    else:
        test_set_path = Path(args.test_set)

    if not test_set_path.exists():
        raise FileNotFoundError(f"Test set file not found: {test_set_path}")

    # Create combined settings for the service
    combined_settings = CombinedSettings(
        ADB_SERVING_HOSTNAME=DatabricksOptions.ADB_SERVING_HOSTNAME,
        ADB_FILES_HOSTNAME=DatabricksOptions.ADB_FILES_HOSTNAME,
        ADB_CLIENT_ID=DatabricksOptions.ADB_CLIENT_ID,
        ADB_CLIENT_SECRET=DatabricksOptions.ADB_CLIENT_SECRET,
        ADB_FILES_CATALOG=DatabricksOptions.ADB_FILES_CATALOG,
        TIMEOUT=DatabricksOptions.TIMEOUT,
        EMBEDDING_MODEL_NAME=PipelineOptions.EMBEDDING_MODEL_NAME,
        GENERATOR_MODEL_NAME=PipelineOptions.GENERATOR_MODEL_NAME,
        RAG_XENC_NAME=PipelineOptions.RAG_XENC_NAME,
        RAG_INDEX_ENDPOINT=PipelineOptions.RAG_INDEX_ENDPOINT,
        RAG_INDEX_NAME=PipelineOptions.RAG_INDEX_NAME,
        RAG_INDEX_TYPE=PipelineOptions.RAG_INDEX_TYPE,
        RAG_NUM_DOCS=PipelineOptions.RAG_NUM_DOCS,
        RAG_MAX_TOKENS=PipelineOptions.RAG_MAX_TOKENS,
        RAG_MIN_SCORE=PipelineOptions.RAG_MIN_SCORE,
        RAG_RELEVANCE_THRESHOLD=PipelineOptions.RAG_RELEVANCE_THRESHOLD,
        RAG_DESCRIPTION_LEN=PipelineOptions.RAG_DESCRIPTION_LEN,
        RAG_DESCRIPTION_MIN_LINE_LEN=PipelineOptions.RAG_DESCRIPTION_MIN_LINE_LEN,  # noqa: E501
        RAG_TEMPERATURE=PipelineOptions.RAG_TEMPERATURE,
        REPHRASE_TEMPERATURE=PipelineOptions.REPHRASE_TEMPERATURE,
        EVALUATOR_TEMPERATURE=PipelineOptions.EVALUATOR_TEMPERATURE,
        PROMPT_CHECK_FREQ=PipelineOptions.PROMPT_CHECK_FREQ,
        LANG_PROB_THRESH=PipelineOptions.LANG_PROB_THRESH,
    )

    adb_spn = DatabricksServicePrincipal(
        DatabricksOptions.ADB_SERVING_HOSTNAME,
        DatabricksOptions.ADB_CLIENT_ID,
        DatabricksOptions.ADB_CLIENT_SECRET
    )

    api_base_url = DatabricksOptions.openai_base_url

    # Create database session
    session = next(get_session())

    qna_pipeline = SupportQnAPipeline(
        http_client=httpx.Client(),
        adb_spn=DatabricksServicePrincipal(
            DatabricksOptions.ADB_SERVING_HOSTNAME,
            DatabricksOptions.ADB_CLIENT_ID,
            DatabricksOptions.ADB_CLIENT_SECRET
        ),
        sa_session=session,
        settings=combined_settings,
        division_id=args.division_id,
        division_name=args.division_name,
        api_version=args.api_version
    )
    qna_pipeline.tracer = PipelineRunTracer()

    evaluator = RAGEvaluator(
        qna_pipeline=qna_pipeline,
        openai_spn=adb_spn,
        openai_baseurl=api_base_url,
        openai_model_name=(PipelineOptions.EVALUATOR_MODEL_NAME
                           or PipelineOptions.GENERATOR_MODEL_NAME),
        openai_embeddings_name=PipelineOptions.EMBEDDING_MODEL_NAME,
        openai_temperature=PipelineOptions.EVALUATOR_TEMPERATURE,
        session=session,
        relaxed=args.relaxed
    )

    # Load test cases based on format
    test_cases = await load_test_cases(
        test_set_path,
        is_generated_qa=args.generated_qa_set
    )

    # For generated Q&A set or skip_evaluation, expand machine groups
    if args.generated_qa_set or args.skip_evaluation:
        test_cases = expand_machine_groups(test_cases)
        Logger.info(
            "Expanded test cases for machine groups",
            total_cases=len(test_cases)
        )

    # Filter test cases by division if specified
    if args.division:
        filtered_cases = []
        for case in test_cases:
            division_id, division_name = evaluator._get_division_info(
                case.machine_type
            )
            if division_name.lower() == args.division.lower():
                filtered_cases.append(case)
        test_cases = filtered_cases
        Logger.info(
            f"Filtered test cases by division '{args.division}'",
            count=len(test_cases)
        )

    Logger.info(f"Processing {len(test_cases)} test cases")

    # Handle raw results mode (generated Q&A set or skip_evaluation)
    if args.skip_evaluation or args.generated_qa_set:
        raw_results = []
        for test_case in test_cases:
            try:
                raw_result = await evaluator.get_raw_response(test_case)
                if raw_result:
                    # Add original machine group to the raw result
                    original_group = next(
                        (group for group, types in MACHINE_TYPE_GROUPS.items()
                         if test_case.machine_type in types),
                        test_case.machine_type
                    )
                    raw_result["machine_group"] = original_group
                    raw_results.append(raw_result)
                    Logger.info(
                        f"Got raw response for question {test_case.case_id}"
                    )
                else:
                    Logger.warning(
                        f"Question {test_case.case_id} returned no raw result"
                    )
            except Exception as e:
                Logger.error(
                    "Failed to get raw response for question",
                    test_case_id=test_case.case_id,
                    error=str(e),
                    traceback=traceback.format_exc()
                )

        # Save raw results
        output_dir = (
            Path(args.output_dir) /
            f"judge_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.output_format == "json":
            output_file = output_dir / "raw_results.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(raw_results, f, indent=2, ensure_ascii=False)
        else:  # csv
            output_file = output_dir / "raw_results.csv"
            df = pd.DataFrame(raw_results)
            df.to_csv(output_file, index=False, encoding="utf-8")

        Logger.info(f"Raw results saved to {output_file}")
        return

    # Normal evaluation mode (only for evaluation format)
    if args.generated_qa_set:
        raise ValueError(
            "Cannot run full evaluation on generated Q&A format test set."
            "Use --skip_evaluation."
        )

    # Group test cases by case_id
    case_groups = defaultdict(list)
    for case in test_cases:
        case_groups[case.case_id].append(case)

    # Sort each group by step number
    for case_id in case_groups:
        case_groups[case_id].sort(key=lambda x: x.step)

    total_cases = len(case_groups)
    total_steps = sum(len(cases) for cases in case_groups.values())
    Logger.info(
        f"Loaded {total_cases} test cases with {total_steps} total steps"
    )

    results = []
    # Process each group of test cases
    for case_id, cases in case_groups.items():
        try:
            case_results = await evaluator.evaluate_test_case(cases)
            if case_results:
                results.extend(case_results)
                Logger.info(
                    f"Successfully evaluated case {case_id}",
                    steps=len(cases),
                    results_count=len(case_results)
                )
            else:
                Logger.warning(
                    f"Failed to evaluate case {case_id}",
                    steps=len(cases)
                )

        except Exception as e:
            Logger.error(
                f"Failed to evaluate case {case_id}",
                error=str(e),
                traceback=traceback.format_exc()
            )

    # Count successful evaluations
    successful_single_step = sum(
        1 for case_id, cases in case_groups.items()
        if len(cases) == 1 and any(
            r.case_id.startswith(case_id + "_step") or r.case_id == case_id
            for r in results
        )
    )
    successful_multi_step = sum(
        1 for case_id, cases in case_groups.items()
        if len(cases) > 1 and any(
            r.case_id.startswith(case_id + "_step") or r.case_id == case_id
            for r in results
        )
    )

    report = create_evaluation_report(evaluator, results)

    output_dir = (
        Path(args.output_dir) /
        f"judge_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    save_evaluation_report(results, report, output_dir)

    # Save missing documents that were not found in latest_mapped_filename_ids
    evaluator.save_missing_documents(output_dir)

    # Generate the evaluation summary using direct string formatting
    try:
        def format_number(number):
            if isinstance(number, (int, float)):
                return (
                    f"{number:.2f}" if isinstance(number, float)
                    else str(number)
                )
            return str(number)

        agg_results = report.get("overall_metrics")
        doc_precision_stats = report.get("summary", {}).get(
            "document_precision_stats", {})

        # Generate the evaluation summary
        summary_text = f"""Evaluation Summary
 =================

 Overall Metrics:
 ---------------
 Total Cases: {format_number(total_cases)}
 Total Steps: {format_number(total_steps)}
 Successful Single-Step Cases: {format_number(successful_single_step)}
 Successful Multi-Step Cases: {format_number(successful_multi_step)}
 Total Successful: {format_number(len(results))}
 Success Rate: {format_number(len(results) / total_steps * 100 if total_steps > 0 else 0)}%

 Document Precision Analysis:
 ---------------------------
 Cases Included in Precision Calc: {format_number(doc_precision_stats.get('cases_included', 0))}
 Cases Excluded (Missing Expected Docs): {format_number(doc_precision_stats.get('cases_excluded', 0))}
 Inclusion Rate: {format_number(doc_precision_stats.get('inclusion_rate', 0))}%
 Average Document Precision: {format_number(report.get('document_precision', 'N/A'))}

 Page Precision Analysis:
 -----------------------
 Average First Page Precision: {format_number(report.get('first_page_precision', 'N/A'))}
 Average Last Page Precision: {format_number(report.get('last_page_precision', 'N/A'))}

 Answer Quality:
 --------------
 Answer Relevance: {format_number(agg_results.get('answer_relevance', 0))}%
 Answer Completeness: {format_number(agg_results.get('answer_completeness', 0))}%
 Answer Accuracy: {format_number(agg_results.get('answer_accuracy', 0))}%
 """  # noqa

        # Save the summary text
        summary_text_path = output_dir / "evaluation_summary.txt"
        with open(summary_text_path, 'w') as f:
            f.write(summary_text)

        Logger.info(
            f"Evaluation summary saved to {summary_text_path}"
        )
        print(summary_text)
    except Exception as e:
        Logger.error(f"Error generating evaluation summary: {str(e)}")

    Logger.info(
        "Evaluation complete",
        total_cases=total_cases,
        total_steps=total_steps,
        successful_single_step=successful_single_step,
        successful_multi_step=successful_multi_step,
        total_successful=len(results),
        output_dir=output_dir
    )


def main():
    """Entry point for poetry script."""
    args = parser.parse_args()
    asyncio.run(run_evaluation(args))


if __name__ == "__main__":
    main()
