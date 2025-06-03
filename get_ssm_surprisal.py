import sys
import configparser
import logging
import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
from preprocess.preprocess import read_ids


def read_vocab(fn):
    idx_to_tokens = {}
    space_idx = []
    subword_idx = []

    with open(fn, "r+") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            token = line.strip().split()[0]
            idx_to_tokens[i] = token
            if token.startswith("▁"):
                space_idx.append(i)
            else:
                subword_idx.append(i)

    return idx_to_tokens, space_idx, subword_idx


def main():
    model = MambaLMHeadModel.from_pretrained(sys.argv[1], device="cuda")
    model.eval()
    model.half()

    train_data, n_toks, n_examples, seq_lens = read_ids(
        sys.argv[2],
        model.config.vocab_size-1,
        1
    )

    assert (len(train_data) == len(n_toks))

    idx_to_tokens, space_idx, subword_idx = read_vocab(sys.argv[3])

    sm = torch.nn.Softmax(dim=-1)

    print("word totsurp bori bprob iprob")
    for i, (batch, batch_n_toks) in enumerate(zip(train_data, n_toks)):
        input_idx = batch[:, :].cuda()
        target_idx = batch[:, 1:].cuda()
        with torch.no_grad():
            output = model(input_idx)
        probs = sm(output.logits.double()).squeeze(0)
        index = torch.arange(0, target_idx.shape[1])
        all_surp = -1 * torch.log2(probs)[:, :-1]
        actual_surp = all_surp[index, target_idx.squeeze()]

        for j in index:
            tok = idx_to_tokens[target_idx.squeeze()[j].item()]
            if tok.startswith("▁"):
                print(tok.replace("▁", ""), actual_surp.squeeze()[j].item(), "B", torch.log2(torch.sum(probs[j][space_idx])).item(), torch.log2(torch.sum(probs[j][subword_idx])).item())
            else:
                print(tok.replace("▁", ""), actual_surp.squeeze()[j].item(), "I", torch.log2(torch.sum(probs[j][space_idx])).item(), torch.log2(torch.sum(probs[j][subword_idx])).item())

        print("<eos>", -1 * torch.log2(torch.sum(probs[-1][space_idx])).item(), "B", torch.log2(torch.sum(probs[-1][space_idx])).item(), torch.log2(torch.sum(probs[-1][subword_idx])).item())


if __name__ == "__main__":
    main()
