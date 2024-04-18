import ast


def preprocess_utterance(data, input_format) -> dict:
    output = {}
    for idx in [1, 2]:
        utterance = ast.literal_eval(data[f"utterance{idx}"])
        input_str = ""
        if "user" in input_format:
            input_str += utterance["user"] + ": "
        if "text" in input_format:
            input_str += utterance["text"]
        
        if "intent" in input_format:
            input_str += f" ({utterance['intent']})"
        
        output[f"utterance{idx}"] = input_str
    
    return output
