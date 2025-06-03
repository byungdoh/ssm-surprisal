import sys
import torch


def read_ids(fn, vocab_size, n_examples_per_step):
    dtype = torch.int64
    ids = []
    n_toks = []
    with open(fn, "r+") as f:
        lines = f.readlines()
        n_examples = len(lines)
        curr_ids = []
        curr_n_toks = []
        seq_lens = []
        for i, line in enumerate(lines):
            if (i+1) % n_examples_per_step == 0 or (i+1) == n_examples:
                int_line = [int(j) for j in line.strip().split()]
                curr_ids.append(int_line)
                seq_lens.append(len(int_line))
                curr_ids = torch.tensor(
                    # <s> and padding to accommodate variable length
                    list(map(lambda x: [1] + x + [vocab_size]*(max(map(len, curr_ids))-len(x)), curr_ids)),
                    dtype=dtype
                )
                ids.append(curr_ids)
                n_toks.append(sum(curr_n_toks))
                curr_ids = []
                curr_n_toks = []
            else:
                int_line = [int(j) for j in line.strip().split()]
                curr_ids.append(int_line)
                curr_n_toks.append(len(int_line))
                seq_lens.append(len(int_line))

    return ids, n_toks, n_examples, seq_lens


def read_ids_list(fn):
    ids = []
    with open(fn, "r+") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            int_line = [int(j) for j in line.strip().split()]
            ids.append(int_line)
            
    return ids
