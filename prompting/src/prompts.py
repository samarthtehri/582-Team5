zeroshot_prompt = """We provide two sentences. Your task is to determine if there is a break in the activity between the two sentences.
0 represents there is no break in the activity between two sentences.
1 represents there. The first sentence occurs at the end of a segment and the second sentence begins a new segment.
Your response should only include 0 or 1. Don't include any other text.

{input1}
{input2}"""


fewshot_prompt = """We provide two sentences. Your task is to determine if there is a break in the activity between the two sentences.
0 represents there is no break in the activity between two sentences.
1 represents there. The first sentence occurs at the end of a segment and the second sentence begins a new segment.
Your response should only include 0 or 1. Don't include any other text.

Alien: Got it.
Human: Are you lost?
1

Alien: How are you doing today?
Human: I am good, thank you for asking
0

Alien: I see.
Human: I opened the TV.
1

Alien: How to open the TV?
Human: Hit the open button on the corner of the screen
0

Alien: Let's go! I am excited!
Human: I have reached living room
1

Alien: No, I guess I should turn on the TV but I do not know how to do it.
Human: you should walk towards it and click on its power button\\
0

{input1}
{input2}"""

zeroshot_cot_prompt = ""


get_prompt_template = {
    "zeroshot": zeroshot_prompt,
    "fewshot": fewshot_prompt,
    "zeroshot_cot": zeroshot_cot_prompt
}
