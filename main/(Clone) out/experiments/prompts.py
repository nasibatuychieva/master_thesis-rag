PROCEDURE_JUDGE_PROMPT = """
# LLM Judge Evaluation Prompt

You are an expert evaluator tasked with assessing the quality and correctness
of responses in a retrieval-augmented generation (RAG) pipeline.
Evaluate the response based on the following framework.

## Inputs:

### Question:
{{question}}

### Expected Answer:
{{expected_answer}}

### Actual Answer:
{{actual_answer}}

## Evaluation Framework:

### Response Categories:
1. **True Positive (TP):**
   - The response includes the **complete and correct** list of
   troubleshooting steps.
   - All provided information matches the expected answer.

2. **False Positive (FP):**
   - The response is **partially correct**, meaning one or more steps are
   missing or incorrect.
   - The missing or incorrect elements significantly impact the usability
   of the response.

3. **True Negative (TN):**
   - No answer is generated because there is insufficient information
   to construct a meaningful
   response.
   - This result is valid if the response correctly indicates inability
   to answer.

4. **False Negative (FN):**
   - The response returns a standard message like
   "I'm sorry, I cannot answer your question" or fails to generate a
   meaningful response.
   - This is incorrect when sufficient information exists to generate
   valid steps.

### Response Style:
The response should conform to the following guidelines:
- It should use bullet points and text formatting.
- The answer should be in the language of the prompt.
- The language used should be technical.
- Clarity and coherence - the response is logically structured and easy
to follow.
The response style score is a number from 1 to 5 that indicates how well
the response adheres
to the guidelines.

### Response completeness:
Measure how complete the response is by counting the number of
troubleshooting steps in the response. Compare the number of troubleshooting
steps in the response to the number of troubleshooting steps in the expected
answer.
The 'step error rate' is the percentage of troubleshooting steps in the
response that contain different instructions than the ones specified in
the expected answer.
The 'step omission rate' is the percentage of the troubleshooting steps
in the expected answer that are not present in the expected response.

### Evaluation Process:
1. Score the response style.
2. Determine the **Response Category** (TP, FP, TN, FN).
3. Compute the step error rate and the step omission rate.

---

Now, provide your evaluation for the above inputs.
Your response should contain:
    - response_style: the response style score.
    - response_style_justification: the justification for the response
    style score.
    - correctness: one of TP, FP, TN, FN,
    - correctness_justification: explanation of why the answer category
    was chosen
    - total_steps: the number of steps in the response.
    - total_expected_steps: the number of steps in the expected answer.
    - wrong_step_count: the number of steps in the response that are different
    from the expected answer.
    - missing_step_count: the number of steps in the expected answer that
    are missing in the answer.
    - step_error_rate: the computed step error percentage
    - step_error_rate_justification: the rationale for the step
    error rate
    - step_omission_rate: the computed step omission percentage
    - step_omission_rate_justification: the rationale for the step
    omission rate

Respond using valid JSON.
"""

SERVICE_REQUEST_JUDGE_PROMPT = """
# LLM Judge Evaluation Prompt

You are an expert evaluator tasked with assessing the quality and
correctness of responses in a retrieval-augmented generation (RAG)
pipeline. Evaluate the response based on the following framework.

## Inputs:

### Question:
{{question}}

### Expected troubleshooting causes:
{{expected_causes}}

### Expected troubleshooting steps:
{{expected_steps}}

### Actual Answer:
{{actual_answer}}

## Evaluation Framework:

Compare the actual answer with the expected causes and steps, using the
instructions below.

### Response Categories:
1. **True Positive (TP):**
   - The response includes the **complete and correct** list of
   troubleshooting causes and troubleshooting steps.
   - All provided information matches the expected answer.

2. **False Positive (FP):**
   - The response is **partially correct**, meaning one or more
   troubleshooting causes or steps are missing or incorrect.
   - The missing or incorrect elements significantly impact the
   usability of the response.

3. **True Negative (TN):**
   - No answer is generated because there is insufficient information
   to construct a meaningful response.
   - This result is valid if the response correctly indicates inability
   to answer.

4. **False Negative (FN):**
   - The response returns a standard message like "I'm sorry, I cannot
   answer your question" or fails to generate a meaningful response.
   - This is incorrect when sufficient information exists to generate
   valid causes and steps.

### Response Style:
The response should conform to the following guidelines:
- It should use bullet points and text formatting.
- The answer should be in the language of the prompt.
- The language used should be technical.
- Clarity and coherence - the response is logically structured and
easy to follow.
The response style score is a number from 1 to 5 that indicates how
well the
response adheres to the guidelines.

### Response completeness:
Measure how complete the response is by counting the number of troubleshooting
causes and steps in the response. Compare the number of troubleshooting causes
and steps in the response to the number of troubleshooting causes and steps in
the expected answer.
The 'step error rate' is the percentage of troubleshooting steps in the
response that contain different instructions than the ones specified in
the expected answer.
The 'step omission rate' is the percentage of the troubleshooting steps in the
expected answer that are not present in the expected response.
The 'cause error rate' is the percentage of troubleshooting causes in the
response that are different than the ones from the expected causes.
The 'cause omission rate' is the percentage of the troubleshooting causes
in the expected answer that are not present in the expected response.

### Evaluation Process:
1. Score the response style.
2. Determine the **Response Category** (TP, FP, TN, FN).
3. Compute the step error rate and the step omission rate.
4. Compute the cause error rate and the cause omission rate.

---

Now, provide your evaluation for the above inputs.
Your response should contain:
    - response_style: the style score for the response style.
    - response_style_justification: the justification for the response style
    score.
    - correctness: one of TP, FP, TN, FN,
    - correctness_justification: explanation of why the answer category was
    chosen
    - total_steps: the number of steps in the response.
    - total_expected_steps: the number of steps in the expected answer.
    - wrong_step_count: the number of steps in the response that are different
    from the expected answer.
    - missing_step_count: the number of steps in the expected answer that
    are missing in the answer.
    - step_error_rate: the computed step error percentage
    - step_error_rate_justification: the rationale for the step error rate
    - step_omission_rate: the computed step omission percentage
    - step_omission_rate_justification: the rationale for the step omission
    rate
    - total_cause_count: the number of expected causes.
    - wrong_cause_count: the number of causes in the response that are
    different than the ones from the expected causes.
    - missing_cause_count: the number of causes in the expected answer that
    are not present in the expected response.
    - cause_error_rate: the computed cause error percentage
    - cause_error_rate_justification: the rationale for the cause error rate
    - cause_omission_rate: the computed cause omission percentage
    - cause_omission_rate_justification: the rationale for the cause
    omission rate

Respond using valid JSON.
"""
