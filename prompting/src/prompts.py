definition = """We provide two sentences in a dialogue. Your task is to determine if there is a break in the activity between the two sentences.
0 represents that the two sentences are in the same segment in the dialogue. There is no break in the activity between two sentences.
1 represents that the two sentences are in different segments in the dialogue. The first sentence occurs at the end of a segment and the second sentence begins a new segment."""

zeroshot_sentence = "Your response should only include 0 or 1. Don't include any other text."

zeroshot_prompt = definition + "\n" + zeroshot_sentence + """

{input1}
{input2}"""


few_shot_examples = [
    [
        "Alien: Ok, that's fine with me.",
        "Human: What is the first thing we have to do in order to use this dishwasher properly?",
        "Explanation: Alien finished the previous conversation. Human starts a new conversation about the dishwasher. The two sentences are in different segments.",
        "1",
    ],
    [
        "Alien: How are you doing today?",
        "Human: I am good, thank you for asking",
        "Explanation: Human answers to the question asked by Alien. The two sentences are in the same segment.",
        "0",
    ],
    [
        "Alien: How to open the TV?",
        "Human: Hit the open button on the corner of the screen",
        "Explanation: Human answers to the question asked by Alien. The two sentences are in the same segment.",
        "0",
    ],
    [
        "Human: Yup! Great job! Now we just have to wait until the dishwasher finishes washing the plate.",
        "Human: Now that the dishwasher is done cleaning the plate, you can take it out of the dishwasher",
        "Explanation: There is a time gap between the two sentences. The first sentence is about the dishwasher washing the plate. The second sentence is about taking the plate out of the dishwasher. The two sentences are in different segments.",
        "1",
    ],
    [
        "Alien: No, I guess I should turn on the TV but I do not know how to do it.",
        "Human: you should walk towards it and click on its power button",
        "Explanation: Human answers to the question asked by Alien about how to turn on the TV. The two sentences are in the same segment.",
        "0",
    ],
    [
        "Alien: Ah of course! I definitely won't forget to do that next time.",
        "Alien: Where do you want me to put this clean plate?",
        "Explanation: Alien finishes the previous conversation. Alien starts a new conversation about the clean plate. The two sentences are in different segments.",
        "1",
    ]
]

few_shot_examples_sentence = "The following are some examples of the task (in a random order):"
your_task_placeholder = """New input (your task):

{input1}
{input2}"""


fewshot_prompt = definition + "\n" + zeroshot_sentence + "\n\n" + few_shot_examples_sentence + "\n\n" + "\n\n".join(
    [f"{example[0]}\n{example[1]}\n{example[3]}" for example in few_shot_examples]
) + "\n\n" + your_task_placeholder


fewshot_cot_prompt = definition + "\n" + zeroshot_sentence + "\n\n" + few_shot_examples_sentence + "\n\n" + "\n\n".join(
    [f"{example[0]}\n{example[1]}\n{example[2]}\nLabel: {example[3]}" for example in few_shot_examples]
) + "\n\n" + your_task_placeholder


get_prompt_template = {
    "zeroshot": zeroshot_prompt,
    "fewshot": fewshot_prompt,
    "fewshot_cot": fewshot_cot_prompt
}


if __name__ == "__main__":
    print("Zero-shot prompt:")
    print(get_prompt_template["zeroshot"])
    
    print("\n\nFew-shot prompt:")
    print(get_prompt_template["fewshot"])
    
    print("\n\nFew-shot COT prompt:")
    print(get_prompt_template["fewshot_cot"])
