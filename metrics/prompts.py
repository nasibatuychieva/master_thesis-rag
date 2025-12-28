"""Prompts used for judging answer correctness."""


JUDGE_SINGLE_ANSWER_COMPLETENESS_PROMPT = """
You are an expert evaluator tasked with assessing the correctness
of responses in a retrieval-augmented generation (RAG) pipeline.
Evaluate the response based on the following framework:

## Inputs:

### Question:
{question}

### Expected Answer:
{expected_answer}

### Actual Answer:
{actual_answer}

## Evaluation Framework:

Compare the actual answer with the expected answer using the
instructions below.

### Response completeness:
1. First, create semantic summaries of both the expected and actual
troubleshooting steps:
   - Group similar steps together
   - Focus on the core actions and their sequence
   - Ignore minor wording differences
   - Consider equivalent phrasings as matches

2. Measure completeness by comparing the semantic summaries:
   - Count the number of unique troubleshooting steps in each summary
   - Compare the semantic meaning rather than exact wording
   - Allow for equivalent but differently worded instructions
   - Consider steps that cover the same action in different ways as matches

3. Calculate error rates based on semantic matching:
   - The 'step error rate' is the percentage of troubleshooting steps in the
   actual response that have no semantic match with expected steps
   - Formula: (wrong_step_count / total_actual_steps) * 100
   - The 'step omission rate' is the percentage of expected steps missing from
   the response
   - Formula: (missing_step_count / total_expected_steps) * 100

4. Calculate overall completeness score:
   - The 'overall completeness score' is a percentage from 0-100 representing
   how complete the response is overall, considering both accuracy and coverage
   - Formula: (correct_steps / total_expected_steps) * 100
   - Where correct_steps = total_expected_steps - missing_step_count
   - This represents the percentage of expected steps that were
   correctly provided

## Evaluation Process:
1. Create semantic summaries of both expected and actual steps
2. Compare the summaries to identify matches and differences
3. Compute the error rates based on semantic matching (use the formulas above)
4. Calculate the overall completeness score

## Expected Output Format:
Your response must be a valid JSON object with the following structure.
Do not include any text before or after the JSON object:

{{
    "total_actual_steps": <number of steps in actual response>,
    "total_expected_steps": <number of steps in expected answer>,
    "wrong_step_count": <number of steps in response that are
    semantically different>,
    "missing_step_count": <number of expected steps not present in response>,
    "step_error_rate": <(wrong_step_count / total_actual_steps) * 100>,
    "step_error_rate_justification": "<explanation>",
    "step_omission_rate": <(missing_step_count / total_expected_steps) * 100>,
    "step_omission_rate_justification": "<explanation>",
    "overall_completeness_score": <number>
}}

IMPORTANT:
1. The response must be a single valid JSON object
2. Do not include any text before or after the JSON object
3. All rate values must be numbers between 0 and 100 representing percentages
4. total_actual_steps = count of steps in the actual response
5. total_expected_steps = count of steps in the expected answer
6. wrong_step_count = steps in response that don't match any expected step
7. missing_step_count = expected steps that are not found in the response
8. Overall completeness score must be between 0 and 100
9. Justification fields must be brief explanations of how the rates were
calculated
10. Do not include any newlines or extra whitespace before or after the
JSON object

---

Now, provide your evaluation for the above inputs following the exact output
format specified above.
"""


JUDGE_MULTI_ANSWER_COMPLETENESS_PROMPT = """
You are an expert evaluator tasked with assessing the quality and
correctness of responses in a retrieval-augmented generation (RAG)
pipeline. Evaluate the response based on the following framework.

## Inputs:

### Question:
{question}

### Expected troubleshooting causes:
{expected_causes}

### Expected troubleshooting steps:
{expected_steps}

### Actual Answer:
{actual_answer}

## Evaluation Framework:

Compare the actual answer with the expected causes and steps, using the
instructions below.

### Response completeness:
1. First, create semantic summaries of both the expected and actual
troubleshooting steps:
   - Group similar steps together
   - Focus on the core actions and their sequence
   - Ignore minor wording differences
   - Consider equivalent phrasings as matches

2. Measure completeness by comparing the semantic summaries:
   - Count the number of unique troubleshooting causes and
   steps in each summary
   - Compare the semantic meaning rather than exact wording
   - Allow for equivalent but differently worded instructions
   - Consider steps that cover the same action in different
   ways as matches

3. Calculate error rates based on semantic matching:
   - The 'step error rate' is the percentage of troubleshooting
   steps in the actual response that have no semantic match with expected steps
   - Formula: (wrong_step_count / total_actual_steps) * 100
   - The 'step omission rate' is the percentage of expected steps
   missing from the response
   - Formula: (missing_step_count / total_expected_steps) * 100
   - The 'cause error rate' is the percentage of troubleshooting
   causes in the actual response that have no semantic match with
   expected causes
   - Formula: (wrong_cause_count / total_actual_cause_count) * 100
   - The 'cause omission rate' is the percentage of expected causes
   missing from the response
   - Formula: (missing_cause_count / total_expected_cause_count) * 100

4. Calculate overall completeness score:
   - The 'overall completeness score' is a percentage from 0-100 representing
   how complete the response is overall, considering both causes and steps
   - Formula:
   ((correct_steps + correct_causes) / (
   total_expected_steps + total_expected_cause_count)) * 100
   - Where correct_steps = total_expected_steps - missing_step_count
   - Where correct_causes = total_expected_cause_count - missing_cause_count
   - This represents the percentage of expected content (both steps and causes)
   that were correctly provided

## Evaluation Process:
1. Create semantic summaries of both expected and actual steps
2. Compare the summaries to identify matches and differences
3. Compute the error rates based on semantic matching (use the formulas above)
4. Calculate the overall completeness score

## Expected Output Format:
Your response must be a valid JSON object with the following
structure. Do not include any text before or after the JSON object:

{{
    "total_actual_steps": <number of steps in actual response>,
    "total_expected_steps": <number of steps in expected answer>,
    "wrong_step_count": <number of steps in response that are
    semantically different>,
    "missing_step_count": <number of expected steps not present in response>,
    "step_error_rate": <(wrong_step_count / total_actual_steps) * 100>,
    "step_error_rate_justification": "<explanation>",
    "step_omission_rate": <(missing_step_count / total_expected_steps) * 100>,
    "step_omission_rate_justification": "<explanation>",
    "total_actual_cause_count": <number of causes in actual response>,
    "total_expected_cause_count": <number of expected causes>,
    "wrong_cause_count": <number of causes in response that are
    semantically different>,
    "missing_cause_count": <number of expected causes not present in response>,
    "cause_error_rate": <(wrong_cause_count / total_actual_cause_count) * 100>,
    "cause_error_rate_justification": "<explanation>",
    "cause_omission_rate": <(
    missing_cause_count / total_expected_cause_count) * 100>,
    "cause_omission_rate_justification": "<explanation>",
    "overall_completeness_score": <number>
}}

IMPORTANT:
1. The response must be a single valid JSON object
2. Do not include any text before or after the JSON object
3. All rate values must be numbers between 0 and 100 representing
percentages
4. total_actual_steps = count of steps in the actual response
5. total_expected_steps = count of steps in the expected answer
6. total_actual_cause_count = count of causes in the actual response
7. total_expected_cause_count = count of expected causes
8. wrong_step_count = steps in response that don't match any expected step
9. missing_step_count = expected steps that are not found in the response
10. wrong_cause_count = causes in response that don't match any expected cause
11. missing_cause_count = expected causes that are not found in the response
12. Overall completeness score must be between 0 and 100
13. Justification fields must be brief explanations of how the
rates were calculated
14. Do not include any newlines or extra whitespace before or after
the JSON object

---

Now, provide your evaluation for the above inputs following the exact
output format specified above.
"""
