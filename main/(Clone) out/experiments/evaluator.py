import re
import typing as t

import traceback
from pathlib import Path

from copilot_rag.pipelines.log import Logger
from copilot_rag.tools.llm_judge.common import (
    TestCase,
    EvaluationResult,
    is_apology_response
)

from copilot_rag.tools.llm_judge.metrics import (
    ResponseStyle,
    Correctness,
    Completeness,
    Faithfulness,
    AnswerRelevance,
    PromptInjection,
    ExcessiveAgency
)

from copilot.orm import Session
from copilot.models.str import MachineTypeModel, DivisionModel

from copilot_rag.pipelines.service_request import SupportQnAPipeline
from copilot_rag.databricks import DatabricksServicePrincipal

from copilot_rag.tools.llm_judge.mock_models import (
    MockMessage,
    MockConversation
)


class RAGEvaluator:
    """Evaluates RAG system responses using LLM judges."""

    def __init__(
        self,
        qna_pipeline: SupportQnAPipeline,
        openai_spn: DatabricksServicePrincipal,
        openai_baseurl: str,
        openai_model_name: str,
        openai_embeddings_name: str,
        openai_temperature: float,
        session: Session,
        relaxed: bool = False
    ):
        """Initialize an evaluator.

        Args:
            qna_pipeline: The QnA pipeline to evaluate
            openai_spn: Databricks service principal for authentication
            openai_baseurl: Base URL for OpenAI API
            openai_model_name: Name of the model to use for judging
            openai_embeddings_name: Name of the embeddings model
            openai_temperature: Sampling temperature
            session: Database session
            relaxed: Whether to use relaxed TP/FP criteria
                (default True for backward compatibility)
        """
        self._qna = qna_pipeline
        self._session = session
        self._relaxed = relaxed
        self._filename_to_id: t.Optional[t.Dict[str, str]] = None

        kwargs = {
            "openai_spn": openai_spn,
            "openai_baseurl": openai_baseurl,
            "openai_model_name": openai_model_name,
            "openai_embeddings_name": openai_embeddings_name,
            "openai_temperature": openai_temperature
        }

        correctness_kwargs = {
            **kwargs,
            "relaxed": relaxed
        }

        # TODO configure this
        self._metrics = {
            "service_request": [
                ResponseStyle(**kwargs),
                Correctness(**correctness_kwargs),
                Completeness(**kwargs),
                Faithfulness(**kwargs),
                AnswerRelevance(**kwargs)
            ],
            "procedure_description": [
                ResponseStyle(**kwargs),
                Correctness(**correctness_kwargs),
                Completeness(**kwargs),
                Faithfulness(**kwargs),
                AnswerRelevance(**kwargs)
            ],
            "prompt_injection": [
                PromptInjection(**kwargs)
            ],
            "excessive_agency": [
                ExcessiveAgency(**kwargs)
            ]
        }
        # Track documents not found in mapping file
        self.missing_documents = set()

    def _get_division_info(self, machine_type: str) -> tuple[int, str]:
        """Get division ID and name for a machine type."""
        query = self._session.query(
            MachineTypeModel,
            DivisionModel
        ).join(
            DivisionModel, MachineTypeModel.division_id == DivisionModel.id
        ).filter(
            MachineTypeModel.name == machine_type
        ).first()

        if not query:
            Logger.warning(
                "No division found for machine type",
                machine_type=machine_type
            )
            return (1, "ft")  # Fallback to default values

        machine, division = query
        Logger.info(
            "Found division info",
            division_id=division.id,
            division_name=division.name,
            machine_type=machine.name
        )
        return (division.id, division.name)

    def _doc_ids_to_str(
        self,
        doc_ids: t.List[int]
    ) -> t.List[str]:
        """Convert document IDs to strings."""
        return [str(doc_id) for doc_id in doc_ids]

    def _get_doc_id(
        self,
        doc_names: t.List[str],
        check_all_found: bool = False
    ) -> t.Union[t.List[str], t.Tuple[t.List[str], bool]]:
        """Get document names from IDs.

        Args:
            doc_names: List of document names to convert
            check_all_found: If True, return tuple with (converted_ids,
            all_found_flag)

        Returns:
            If check_all_found is False: List of converted document IDs
            If check_all_found is True: Tuple of (converted_ids,
            all_found_flag)
        """
        try:
            filename_to_id = self._load_filename_to_id()

            # Normalize unicode dashes to ASCII hyphen
            def normalize_dashes(s: str) -> str:
                return s.replace('\u2013', '-').replace('\u2014', '-')

            # Extract common id formats like 1234-5678-901
            pattern = r'(\d{4}-\d{4}-\d{3})'

            converted_ids = []
            all_found = True

            for doc_name in doc_names:
                doc_name = normalize_dashes(str(doc_name).strip())

                # Treat numeric (possibly float-formatted) as direct doc IDs
                if re.match(r'^\d+(?:\.0+)?$', doc_name):
                    try:
                        as_int = int(float(doc_name))
                        converted_ids.append(str(as_int))
                        found = True
                    except Exception:
                        pass
                found = False

                # If it's already a numeric ID, keep it as is
                if doc_name.isdigit():
                    converted_ids.append(doc_name)
                    found = True
                else:
                    match = re.search(pattern, doc_name)
                    if match:
                        matching_files = [
                            f for f in filename_to_id.keys()
                            if normalize_dashes(f).startswith(match.group(1))
                        ]
                        if matching_files:
                            converted_ids.append(
                                str(filename_to_id[matching_files[0]])
                            )
                            found = True
                        else:
                            Logger.warning(
                                "No mapping found for document ID",
                                doc_name=doc_name
                            )
                            # Track missing document
                            self.missing_documents.add(doc_name)
                    else:
                        # Fuzzy match:
                        # capture 4-digit, 2-4 digit prefix, 3-digit
                        fuzzy = re.search(
                            r'^(?:.*?)(\d{4})\D+(\d{2,4})\D+(\d{3})(?:.*?)$',
                            doc_name
                        )
                        if fuzzy:
                            g1, g2pref, g3 = fuzzy.group(1), fuzzy.group(2), fuzzy.group(3)  # noqa: E501
                            # Scan for first token where groups match on prefix
                            for f in filename_to_id.keys():
                                nf = normalize_dashes(f)
                                token = re.search(
                                    r'(\d{4})-(\d{2,4})-(\d{3})',
                                    nf
                                )
                                if not token:
                                    continue
                                tg1, tg2, tg3 = token.group(1), token.group(2), token.group(3)  # noqa: E501
                                if tg1 == g1 and tg3 == g3 and tg2.startswith(g2pref):  # noqa: E501
                                    converted_ids.append(
                                        str(filename_to_id[f])
                                    )
                                    found = True
                                    break
                            if not found:
                                Logger.warning(
                                    "Invalid document ID format",
                                    doc_name=doc_name
                                )
                                # Track invalid format document
                                self.missing_documents.add(doc_name)

                if not found:
                    all_found = False

            Logger.info("Converted document IDs", converted_ids=converted_ids)

            if check_all_found:
                return converted_ids, all_found
            return converted_ids

        except Exception as e:
            Logger.error(f"Error getting document names: {str(e)}")
            if check_all_found:
                return [], False
            return []

    def _load_filename_to_id(self) -> t.Dict[str, str]:
        """Load filename->doc_id mapping from CSV using a robust parser.

        The mapping file is large and may contain extra semicolons in later
        fields. We only need the first two fields: file_name and doc_id.
        """
        if self._filename_to_id is not None:
            return self._filename_to_id

        mapping_path = Path(__file__).parent / 'latest_mapped_filename_ids.csv'
        mapping: t.Dict[str, str] = {}
        try:
            with open(mapping_path, 'r', encoding='utf-8', errors='replace') as f:  # noqa: E501
                first = True
                for line in f:
                    if first:
                        first = False
                        # skip header
                        continue
                    line = line.strip('\n').lstrip('\ufeff')
                    if not line:
                        continue
                    parts = line.split(';')
                    if len(parts) < 2:
                        continue
                    file_name = parts[0].strip()
                    doc_id = parts[1].strip()
                    if not file_name or not doc_id:
                        continue
                    mapping[file_name] = doc_id
            self._filename_to_id = mapping
            return mapping
        except Exception as e:
            Logger.error(f"Error loading mapping file: {str(e)}")
            return {}

    def _calculate_document_precision(
        self,
        expected_docs: t.List[str],
        retrieved_docs: t.List[str]
    ) -> t.Tuple[t.Optional[float], t.Optional[float]]:
        """Calculate document precision and omission rate.

        Returns tuple of (precision, omission_rate). Both are None
        if expected documents are not found in
        latest_mapped_filename_ids.
        """
        if not expected_docs or not retrieved_docs:
            return 0.0, 0.0 if expected_docs else None

        # Ensure expected_docs is a list of strings
        if isinstance(expected_docs, str):
            expected_docs = [
                doc.strip() for doc in expected_docs.split('\n') if doc.strip()
            ]
        elif isinstance(expected_docs, list):
            expected_docs = [str(doc).strip() for doc in expected_docs if doc]

        # Ensure retrieved_docs is a list of strings
        retrieved_docs = [str(doc).strip() for doc in retrieved_docs if doc]

        # Convert all document IDs to their numeric form
        # Check if all expected documents are found in the mapping
        expected_docs, all_expected_found = self._get_doc_id(
            expected_docs,
            check_all_found=True
        )
        retrieved_docs = self._get_doc_id(retrieved_docs)

        # If not all expected documents were found in the mapping, return None
        # to exclude this test case from document precision computation
        if not all_expected_found:
            Logger.info(
                "Excluding test case from document precision computation: "
                "not all expected documents found in "
                "latest_mapped_filename_ids.csv"
            )
            return None, None

        # Filter out any non-numeric IDs
        expected_docs = [doc for doc in expected_docs if doc.isdigit()]
        retrieved_docs = [doc for doc in retrieved_docs if doc.isdigit()]

        if not expected_docs or not retrieved_docs:
            Logger.warning(
                "No valid numeric document IDs found for precision calculation"
            )
            return 0.0, 1.0 if expected_docs else 0.0

        expected_set = set(expected_docs)
        retrieved_set = set(retrieved_docs)

        Logger.info(f"Expected documents (numeric): {expected_set}")
        Logger.info(f"Retrieved documents (numeric): {retrieved_set}")

        # Calculate precision using string comparison
        correct = len(expected_set.intersection(retrieved_set))
        precision = correct / len(retrieved_set) if retrieved_set else 0.0

        # Calculate omission rate: percentage of expected docs that are missing
        missing = len(expected_set - retrieved_set)
        omission_rate = missing / len(expected_set) if expected_set else 0.0

        return precision, omission_rate

    def _calculate_page_precision(
        self,
        expected_pages: str,
        retrieved_refs: t.List[dict]
    ) -> t.Dict[str, float]:
        """Calculate first and last page precision.

        Args:
            expected_pages: String containing expected page ranges
            (e.g. "1,3-5,7")
            retrieved_refs: List of DocReference objects from RAG response

        Returns:
            Dictionary with first_page_precision and last_page_precision
        """
        if not expected_pages or not retrieved_refs:
            return {
                "first_page_precision": 0.0,
                "last_page_precision": 0.0
            }

        # Extract expected pages
        expected_ranges = []
        # Split by both commas and newlines
        for entry in re.split(r'[,|\n]', expected_pages):
            entry = entry.strip()
            if not entry:
                continue
            if "-" in entry:
                try:
                    start, end = map(int, entry.split("-"))
                    expected_ranges.append((start, end))
                except ValueError:
                    Logger.warning(f"Invalid page range format: {entry}")
                    continue
            else:
                try:
                    page = int(entry)
                    expected_ranges.append((page, page))
                except ValueError:
                    Logger.warning(f"Invalid page number format: {entry}")
                    continue

        if not expected_ranges:
            Logger.warning(f"No valid page ranges found in: {expected_pages}")
            return {
                "first_page_precision": 0.0,
                "last_page_precision": 0.0
            }

        expected_first = min(r[0] for r in expected_ranges)
        expected_last = max(r[1] for r in expected_ranges)

        retrieved_first = min(
            ref["start_page"] for ref in retrieved_refs
        ) if retrieved_refs else 0
        retrieved_last = max(
            ref["end_page"] for ref in retrieved_refs
        ) if retrieved_refs else 0

        Logger.info(
            "Expected page range",
            expected_first=expected_first,
            expected_last=expected_last
        )
        Logger.info(
            "Retrieved page range",
            retrieved_first=retrieved_first, retrieved_last=retrieved_last)

        return {
            "first_page_precision": 1.0 if retrieved_first == expected_first else 0.0,  # noqa: E501
            "last_page_precision": 1.0 if expected_last <= retrieved_last <= expected_last + 2 else 0.0  # noqa: E501
        }

    async def evaluate_test_case(
        self,
        test_cases: t.List[TestCase]
    ) -> t.List[EvaluationResult]:
        """Evaluate a list of test cases in order of step number.

        Args:
            test_cases: List of test cases to evaluate, will be sorted by step
                       number if not already sorted

        Returns:
            List[EvaluationResult]: List of evaluation results for all test
                          cases if all steps complete successfully, or []
        """
        if not test_cases:
            return []

        # Sort test cases by step number
        sorted_cases = sorted(test_cases, key=lambda x: x.step)

        # Verify steps are consecutive
        steps = [case.step for case in sorted_cases]
        if steps != list(range(min(steps), max(steps) + 1)):
            Logger.error(
                f"Test cases have non-consecutive steps: {steps}",
                case_id=sorted_cases[0].case_id
            )
            return []

        try:
            # Initialize conversation memory for this sequence
            conversation_memory = MockConversation()
            all_results = []

            division_id, division_name = self._get_division_info(
                sorted_cases[0].machine_type
            )

            # Process each step in order
            for test_case in sorted_cases:
                solution = await self._qna.answer(
                    query=test_case.question,
                    machine_type=test_case.machine_type,
                    conversation=conversation_memory,
                )

                # Normalize solution to dict-based structure
                if isinstance(solution, dict):
                    answer_text = str(solution.get("answer", "") or "")
                    raw_refs = solution.get("references", []) or []
                else:
                    answer_text = str(getattr(solution, "answer", "") or "")
                    raw_refs = getattr(solution, "references", []) or []

                # Normalize references into list[dict]
                normalized_refs = []
                for ref in raw_refs:
                    try:
                        if isinstance(ref, dict):
                            doc_id_val = ref.get("doc_id")
                            start_page_val = ref.get("start_page")
                            end_page_val = ref.get("end_page")
                            description_val = ref.get("description", "")
                        else:
                            doc_id_val = getattr(ref, "doc_id", None)
                            start_page_val = getattr(ref, "start_page", None)
                            end_page_val = getattr(ref, "end_page", None)
                            description_val = getattr(ref, "description", "")

                        # Require a doc identifier;
                        # tolerate missing one of the page bounds
                        if doc_id_val is None:
                            continue

                        # If one of start/end is missing, default end=start
                        if start_page_val is None and end_page_val is None:
                            # No usable pages
                            continue
                        if start_page_val is None:
                            start_page_val = end_page_val
                        if end_page_val is None:
                            end_page_val = start_page_val

                        normalized_refs.append({
                            "doc_id": str(doc_id_val),
                            "start_page": int(start_page_val),
                            "end_page": int(end_page_val),
                            "description": str(description_val or ""),
                        })
                    except Exception:
                        continue

                test_case.answer = answer_text
                test_case.context = self._qna.documents

                # Map retrieved document identifiers to numeric IDs
                raw_doc_ids = [
                    str(ref["doc_id"]).strip() for ref in normalized_refs
                ]
                mapped_doc_ids: t.List[str] = []
                for doc in raw_doc_ids:
                    if doc.isdigit():
                        mapped_doc_ids.append(doc)
                    else:
                        mapped = self._get_doc_id([doc])
                        if mapped:
                            mapped_doc_ids.extend(
                                [m for m in mapped if m.isdigit()]
                            )
                # Deduplicate while preserving order
                seen = set()
                doc_ids = []
                for d in mapped_doc_ids:
                    if d not in seen:
                        seen.add(d)
                        doc_ids.append(d)

                # Handle expected documents
                if isinstance(test_case.expected_docs, str):
                    parts = re.split(r"[\n,;]", test_case.expected_docs)
                    expected_docs = [
                        doc.strip() for doc in parts if doc and doc.strip()
                    ]
                else:
                    # If it's a list, also explode any entries
                    # that contain multiple IDs
                    exploded: t.List[str] = []
                    for doc in test_case.expected_docs:
                        if not doc:
                            continue
                        for part in re.split(r"[\n,;]", str(doc)):
                            if part and part.strip():
                                exploded.append(part.strip())
                    expected_docs = exploded

                # Handle expected pages
                if isinstance(test_case.expected_pages, str):
                    expected_pages = test_case.expected_pages
                elif isinstance(test_case.expected_pages, list):
                    expected_pages = ','.join(
                        str(page).strip() for page in test_case.expected_pages if page  # noqa: E501
                    )
                else:
                    expected_pages = str(test_case.expected_pages) if test_case.expected_pages else ""  # noqa: E501

                # Check if this is an apology response
                is_apology = is_apology_response(test_case.answer)

                # Skip certain metrics for apology responses
                metrics_to_skip_for_apology = [
                    "completeness", "answer_relevance"
                ]

                if not is_apology:
                    # Calculate document precision
                    doc_precision, omission_rate = self._calculate_document_precision(  # noqa: E501
                        expected_docs,
                        doc_ids
                    )

                    page_precision = self._calculate_page_precision(
                        expected_pages,
                        [
                            {
                                "start_page": ref["start_page"],
                                "end_page": ref["end_page"],
                            }
                            for ref in normalized_refs
                        ],
                    )
                else:
                    # For apology responses, set precision values to None
                    doc_precision = None
                    omission_rate = None
                    page_precision = {}
                    Logger.info(
                        f"Skipping document/page precision for "
                        f"apology response in case {test_case.case_id}"
                    )

                metric_results = {}
                for metric in self._metrics[test_case.category]:
                    # Skip certain metrics for apology responses
                    if (is_apology and
                            metric.name in metrics_to_skip_for_apology):
                        Logger.info(
                            f"Skipping {metric.name} metric for "
                            f"apology response in case {test_case.case_id}"
                        )
                        continue

                    try:
                        result = metric.evaluate(test_case)
                        Logger.info(metric.__class__.__name__, result=result)
                        metric_results.update(result)
                    except Exception as e:
                        Logger.error(
                            "Failed to compute metric",
                            metric=metric.__class__.__name__.lower(),
                            error=str(e),
                        )
                        traceback.print_stack()

                run_times = self._qna.tracer.results.copy()
                self._qna.tracer.clear()

                # Convert expected docs to numeric IDs
                expected_docs = self._get_doc_id(expected_docs)
                expected_docs = [doc for doc in expected_docs if doc.isdigit()]

                # Convert actual pages: merge to a single range per document
                # and drop description
                actual_pages: t.List[t.Dict] = []
                if not is_apology:
                    per_doc_ranges: t.Dict[str, t.Tuple[int, int]] = {}
                    for ref in normalized_refs:
                        # Map ref doc id to numeric if needed
                        ref_doc = str(ref["doc_id"]).strip()
                        if ref_doc.isdigit():
                            doc_key = ref_doc
                        else:
                            mapped = self._get_doc_id([ref_doc])
                            if not mapped:
                                continue
                            doc_key = mapped[0]

                        start_page = int(ref["start_page"]) if ref.get("start_page") is not None else 0  # noqa: E501
                        end_page = int(ref["end_page"]) if ref.get("end_page") is not None else start_page  # noqa: E501
                        # Ensure last page has a range of +2 max from start page # noqa: E501
                        constrained_end_page = min(end_page, start_page + 2)

                        if doc_key in per_doc_ranges:
                            min_start, max_end = per_doc_ranges[doc_key]
                            per_doc_ranges[doc_key] = (
                                min(min_start, start_page),
                                max(max_end, constrained_end_page)
                            )
                        else:
                            per_doc_ranges[doc_key] = (
                                start_page,
                                constrained_end_page
                            )

                    for doc_key, (min_start, max_end) in per_doc_ranges.items():  # noqa: E501
                        actual_pages.append({
                            "start_page": min_start,
                            "end_page": max_end,
                        })
                else:
                    actual_pages = []

                result = EvaluationResult(
                    case_id=f"{test_case.case_id}_step{test_case.step}",
                    prompt=test_case.question,
                    step=test_case.step,
                    machine_type=test_case.machine_type,
                    category=test_case.category,
                    actual_answer=test_case.answer,
                    expected_answer=(
                        (
                            (
                                f"Expected Causes:\n{test_case.expected_answers[1]}\n\n"  # noqa: E501
                                f"Expected Workarounds:\n{test_case.expected_answers[0]}"  # noqa: E501
                            )
                            if (len(test_case.expected_answers) > 1 and test_case.expected_answers[1])  # noqa: E501
                            else test_case.expected_answers[0]
                        )
                        if test_case.expected_answers
                        else ""
                    ),
                    document_precision=doc_precision,
                    document_omission_rate=omission_rate,
                    page_precision=page_precision,
                    retrieved_documents=doc_ids,
                    expected_docs=expected_docs,
                    expected_pages=expected_pages,
                    actual_pages=actual_pages,
                    durations=run_times,
                    metrics=metric_results
                )

                all_results.append(result)

                Logger.info(
                    f"Evaluated step {test_case.step} for case "
                    f"{test_case.case_id}",
                    result=result
                )

                # Add message to conversation memory for next step
                conversation_memory.append(
                    MockMessage(
                        prompt=test_case.question,
                        answer=answer_text,
                        documents=normalized_refs,
                    )
                )

            return all_results

        except Exception as e:
            Logger.error(
                f"Error evaluating case {sorted_cases[0].case_id}: {e}"
            )
            traceback.print_exc()
            return []

    async def get_raw_response(
        self,
        test_case: TestCase
    ) -> t.Optional[t.Dict]:
        """Get raw response without evaluation.

        Args:
            test_case: The test case to process

        Returns:
            Dictionary containing raw response data
        """
        try:
            conversation_memory = MockConversation()

            division_id, division_name = self._get_division_info(
                test_case.machine_type
            )
            # Get response from RAG system
            response = await self._qna.answer(
                query=test_case.question,
                machine_type=test_case.machine_type,
                conversation=conversation_memory,
            )

            if not response:
                Logger.warning(
                    "No response received from RAG system",
                    test_case_id=test_case.case_id
                )
                return None

            # Normalize response to dict structure
            if isinstance(response, dict):
                answer_text = str(response.get("answer", "") or "")
                refs = response.get("references", []) or []
                normalized_refs = []
                for ref in refs:
                    try:
                        if isinstance(ref, dict):
                            normalized_refs.append({
                                "doc_id": str(ref.get("doc_id", "")),
                                "start_page": int(ref.get("start_page", 0)),
                                "end_page": int(ref.get("end_page", 0)),
                                "description": str(ref.get("description", "")),
                            })
                        else:
                            normalized_refs.append({
                                "doc_id": str(getattr(ref, "doc_id", "")),
                                "start_page": int(getattr(ref, "start_page", 0)),  # noqa: E501
                                "end_page": int(getattr(ref, "end_page", 0)),
                                "description": str(getattr(ref, "description", "")),  # noqa: E501
                            })
                    except Exception:
                        continue
            else:
                answer_text = str(getattr(response, "answer", "") or "")
                refs = getattr(response, "references", []) or []
                normalized_refs = []
                for ref in refs:
                    try:
                        normalized_refs.append({
                            "doc_id": str(getattr(ref, "doc_id", "")),
                            "start_page": int(getattr(ref, "start_page", 0)),
                            "end_page": int(getattr(ref, "end_page", 0)),
                            "description": str(getattr(ref, "description", "")),  # noqa: E501
                        })
                    except Exception:
                        continue

            # Create raw result with full document information
            raw_result = {
                "id": test_case.case_id,
                "machine_type": test_case.machine_type,
                "question": test_case.question,
                "answer": answer_text,
                "references": normalized_refs,
            }

            return raw_result

        except Exception as e:
            Logger.error(
                "Error getting raw response for test case",
                test_case_id=test_case.case_id,
                error=str(e)
            )
            traceback.print_exc()
            return None

    def aggregate_metrics(
            self, results: t.List[EvaluationResult]
    ) -> t.Dict[str, float]:
        """Compute metrics over a set of evaluation results."""
        metrics = [m for _, mlist in self._metrics.items() for m in mlist]
        return {
            metric.name: metric.aggregate(results) for metric in metrics
        }

    def aggregate_metrics_by_category(
            self, results: t.List[EvaluationResult]
    ) -> t.Dict[str, float]:
        """Compute metrics over a set of evaluation results for a specific category."""  # noqa: E501
        if not results:
            return {"message": "No evaluations in this category"}

        if results and results[0].category in self._metrics:
            category_metrics = self._metrics[results[0].category]
        else:
            return {"message": "Unknown category or no metrics defined"}

        metrics_dict = {}
        for metric in category_metrics:
            try:
                metric_result = metric.aggregate(results)
                if metric_result:
                    metrics_dict[metric.name] = metric_result
            except Exception as e:
                Logger.error(
                    f"Error aggregating {metric.name} "
                    f"for category {results[0].category if results else 'unknown'}: {str(e)}"  # noqa: E501
                )

        return metrics_dict

    def save_missing_documents(self, output_dir: t.Union[str, Path]) -> None:
        """Save list of documents not found in latest_mapped_filename_ids
        to a file.

        Args:
            output_dir: Directory where to save the missing documents file
        """
        output_path = Path(output_dir)
        missing_docs_file = output_path / "missing_documents_from_mapping.txt"

        try:
            with open(missing_docs_file, 'w') as f:
                f.write(
                    "Documents from LLM_Judge_184.csv not "
                    "found in latest_mapped_filename_ids.csv:\n")
                f.write("=" * 80 + "\n\n")

                if not self.missing_documents:
                    f.write("No missing documents found - all expected "
                            "documents were present in the mapping file.\n")
                else:
                    f.write(f"Total missing documents: "
                            f"{len(self.missing_documents)}\n\n")

                    # Sort documents for consistent output
                    sorted_missing = sorted(self.missing_documents)

                    for i, doc in enumerate(sorted_missing, 1):
                        f.write(f"{i:3d}. {doc}\n")

            Logger.info(
                "Missing documents saved",
                file_path=str(missing_docs_file),
                count=len(self.missing_documents)
            )

        except Exception as e:
            Logger.error(f"Error saving missing documents: {str(e)}")
