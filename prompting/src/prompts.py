# this definition is used in all prompts
definition = """We provide two sentences in a dialogue. Your task is to determine if there is a break in the activity between the two sentences.
0 represents that the two sentences are in the same segment in the dialogue. There is no break in the activity between two sentences.
1 represents that the two sentences are in different segments in the dialogue. The first sentence occurs at the end of a segment and the second sentence begins a new segment."""


# zero-shot prompt
zeroshot_sentence = "Your response should only include 0 or 1. Don't include any other text."
zeroshot_prompt = definition + "\n" + zeroshot_sentence + """

{input1}
{input2}"""


# few-shot examples
few_shot_examples = [
    {
        "utterance1": {'user': 'Alien', 'text': "Got it. Let's go", 'intent': 'Ask_get_started'},
        "utterance2": {'user': 'Human', 'text': 'Now you have found the TV, can you open it', 'intent': 'Inform_next_step'},
        "label": "1",
        "category": "watch_tv_1",
        "explanation": "There is a time gap between the two sentences. They are in different segments in the dialogue.",
    },
    {
        "utterance1": {'user': 'Alien', 'text': 'Okay is this the computer?', 'intent': 'Ask_about_object'},
        "utterance2": {'user': 'Human', 'text': 'No thats not the computer, that is the stove', 'intent': 'Inform_object'},
        "label": "0",
        "category": "using_computer_1",
        "explanation": "The second sentence is a response to the first sentence. They are in the same segment in the dialogue.",
    },
    {
        "utterance1": {'user': 'Alien', 'text': "Sure. I'm on my way.", 'intent': 'Inform_understanding'},
        "utterance2": {'user': 'Human', 'text': 'Are you ok Zara? Why you standing next to the refrigerator?', 'intent': 'Ask_okay'},
        "label": "1",
        "category": "wash_glass_3",
        "explanation": "There is a time gap between the two sentences. They are in different segments in the dialogue.",
    },
    {
        "utterance1": {'user': 'Human', 'text': 'Yes, please.', 'intent': 'Yes-no_positive'},
        "utterance2": {'user': 'Human', 'text': 'Can you set a 20 second heating timer for the microwave?', 'intent': 'Inform_next_step'},
        'label': '1',
        "category": "using_microwave_3",
        "explanation": "The two sentences are about different activities. They are in different segments in the dialogue.",
    },
    {
        "utterance1": {'user': 'Human', 'text': 'Do you see the black object on the white mat?', 'intent': 'Ask_know_object'},
        "utterance2": {'user': 'Alien', 'text': 'I see it', 'intent': 'Yes-no_positive'},
        'label': '0',
        "category": "using_computer_3",
        "explanation": "The second sentence is a response to the first sentence. They are in the same segment in the dialogue.",
    },
    {
        "utterance1": {'user': 'Alien', 'text': 'This book is interesting. Did the same author also write the other book?', 'intent': 'Ask_about_object'},
        "utterance2": {'user': 'Human', 'text': 'I believe so. Glad your enjoying them.', 'intent': 'Yes-no_positive'},
        'label': '0',
        "category": "read_book_0",
        "explanation": "The second sentence is a response to the first sentence. They are in the same segment in the dialogue.",
    }
]


few_shot_examples_sentence = "The following are some examples of the task (in a random order):"
your_task_placeholder = """New input (your task):

{input1}
{input2}"""


# few-shot prompt (only final answer)
fewshot_prompt = definition + "\n" + zeroshot_sentence + "\n\n" + few_shot_examples_sentence + "\n\n" + "\n\n".join(
    [f"{example['utterance1']['user']}: {example['utterance1']['text']}\n{example['utterance2']['user']}: {example['utterance2']['text']}\n{example['label']}" for example in few_shot_examples]
) + "\n\n" + your_task_placeholder


# few-shot prompt with chain-of-thought
fewshot_cot_prompt = definition + "\n" + zeroshot_sentence + "\n\n" + few_shot_examples_sentence + "\n\n" + "\n\n".join(
    [f"{example['utterance1']['user']}: {example['utterance1']['text']}\n{example['utterance2']['user']}: {example['utterance2']['text']}\nExplanation: {example['explanation']}\nLabel: {example['label']}" for example in few_shot_examples]
) + "\n\n" + your_task_placeholder


# few-shot prompt with chain-of-thought and intent
fewshot_cot_prompt_intent = definition + "\n" + zeroshot_sentence + "\n\n" + few_shot_examples_sentence + "\n\n" + "\n\n".join(
    [f"{example['utterance1']['user']}: {example['utterance1']['text']} ({example['utterance1']['intent']})\n{example['utterance2']['user']}: {example['utterance2']['text']} ({example['utterance2']['intent']})\nExplanation: {example['explanation']}\nLabel: {example['label']}" for example in few_shot_examples]
) + "\n\n" + your_task_placeholder


# few-shot prompt with chain-of-thought and category
fewshot_cot_prompt_category = definition + "\n" + zeroshot_sentence + "\n\n" + few_shot_examples_sentence + "\n\n" + "\n\n".join(
    [f"{example['utterance1']['user']}: {example['utterance1']['text']}\n{example['utterance2']['user']}: {example['utterance2']['text']}\nCategory: {example['category']}\nExplanation: {example['explanation']}\nLabel: {example['label']}" for example in few_shot_examples]
) + "\n\n" + your_task_placeholder


# this dictionary will be used in run_llms.py
get_prompt_template = {
    "zeroshot": zeroshot_prompt,
    "fewshot": fewshot_prompt,
    "fewshot_cot": fewshot_cot_prompt,
    "fewshot_cot_intent": fewshot_cot_prompt_intent,
    "fewshot_cot_category": fewshot_cot_prompt_category,
}


# check prompts
if __name__ == "__main__":
    print("Zero-shot prompt:")
    print(get_prompt_template["zeroshot"])
    
    print("\n\nFew-shot prompt:")
    print(get_prompt_template["fewshot"])
    
    print("\n\nFew-shot COT prompt:")
    print(get_prompt_template["fewshot_cot"])
    
    print("\n\nFew-shot COT with intent prompt:")
    print(get_prompt_template["fewshot_cot_intent"])
    
    print("\n\nFew-shot COT with category prompt:")
    print(get_prompt_template["fewshot_cot_category"])
