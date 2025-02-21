from contextlib import contextmanager

import torch


@contextmanager
def suspend_nn_inits():
    """
    see https://github.com/huggingface/transformers/issues/26258
    """
    skip = lambda *args, **kwargs: None
    saved_inits = (
        torch.nn.init.kaiming_uniform_,
        torch.nn.init.uniform_,
        torch.nn.init.normal_,
        torch.nn.init.xavier_uniform_,
    )  # saving
    torch.nn.init.kaiming_uniform_ = torch.nn.init.uniform_ = torch.nn.init.normal_ = (
        torch.nn.init.xavier_uniform_
    ) = skip  # replacing
    try:
        yield
    finally:
        (
            torch.nn.init.kaiming_uniform_,
            torch.nn.init.uniform_,
            torch.nn.init.normal_,
            torch.nn.init.xavier_uniform_,
        ) = saved_inits  # restoring


def validate_args(args):
    pass


def padding_vocab_size(orig_word_embed, md, args, attr_name="padded_vocab_size"):

    vocab_size_attr = eval(f"args.{attr_name}")
    if md.true_vocab_size is not None:
        orig_vocab_size = orig_word_embed.shape[0]
        # Cut out extra padding we don't need
        if orig_vocab_size > vocab_size_attr:
            full_word_embed = orig_word_embed[0:vocab_size_attr, :]

        # Expanding embedding to larger size by replicating final entry
        elif orig_vocab_size < vocab_size_attr:
            padding_size = vocab_size_attr - orig_vocab_size

            full_word_embed = torch.cat(
                (
                    orig_word_embed,
                    orig_word_embed[-1].unsqueeze(0).expand(padding_size, -1),
                )
            )

        # Same size!
        else:
            full_word_embed = orig_word_embed
    else:
        print(
            "Original vocab size not specified, leaving embedding table as-is. "
            "If you've changed the tensor parallel size this could cause problems."
        )
        setattr(args, attr_name, orig_word_embed.shape[0])
        full_word_embed = orig_word_embed
    return full_word_embed
