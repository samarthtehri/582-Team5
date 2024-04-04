zeroshot_prompt = """We provide two sentences. Your task is to determine if there is a break in the activity between the two sentences.
0 represents there is no break in the activity between two sentences.
1 represents there. The first sentence occurs at the end of a segment and the second sentence begins a new segment.
Your response should only include 0 or 1. Don't include any other text.

{input1}
{input2}"""


zeroshot_cot_prompt = ""


fewshot_prompt = ""


get_prompt_template = {
    "zeroshot": zeroshot_prompt,
    "zeroshot_cot": zeroshot_cot_prompt,
    "fewshot": fewshot_prompt
}
