
from tools.stream_conversation.conversation_convo_v2 import covert_prompt_to_input_ids_with_history


prompts = "你好，世界"
history = [["USER", "你好"], ["SYSTEM", "你好！有什么我可以帮助你的吗？"], ["USER", "flagOpen"], ["SYSTEM", "您好，flagOpen()是PHP中的一种函数，用于打开一个包含用户定义标记的文件。您可以使用它来读取或写入文件中的标记。希望对您有所帮助！\n"]]


def get_tokenizer():
    from megatron.tokenizer.tokenizer import _AquilaTokenizer
    vocab_file = "../examples/aquila/tokenizer/vocab.json"
    merge_file = "../examples/aquila/tokenizer/merges.txt"
    special = "../examples/aquila/tokenizer/special_tokens.txt"
    tokenizer = _AquilaTokenizer(vocab_file, merge_file, special)

    return tokenizer

tokenizer = get_tokenizer()

input_ids = covert_prompt_to_input_ids_with_history(prompts, history, tokenizer, 2048)

print(input_ids)
