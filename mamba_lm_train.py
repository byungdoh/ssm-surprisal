import sys
import configparser
import logging
import torch
import random
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
from preprocess.preprocess import read_ids


def main():
    # configuration and initialization
    ini = configparser.ConfigParser()
    ini.read(sys.argv[1])
    dtype_dict = {
        "bf16": torch.bfloat16,
        "f16": torch.float16,
        "f32": torch.float32
    }
    d_model = ini.getint("mamba2", "d_model")
    d_intermediate = ini.getint("mamba2", "d_intermediate")
    n_layer = ini.getint("mamba2", "n_layer")
    vocab_size = ini.getint("mamba2", "vocab_size")
    ssm_cfg = {
        "layer": ini.get("mamba2", "layer"),
        "d_state": ini.getint("mamba2", "d_state"),
        "headdim": ini.getint("mamba2", "headdim"),
        "ngroups": ini.getint("mamba2", "ngroups")
    }
    rms_norm = ini.getboolean("mamba2", "rms_norm")
    pad_vocab_size_multiple = ini.getint("mamba2", "pad_vocab_size_multiple")
    device = ini.get("mamba2", "device")
    dtype = ini.get("mamba2", "dtype")
    train_fn = ini.get("mamba2", "train_fn")
    n_examples_per_step = ini.getint("mamba2", "n_examples_per_step")
    n_steps_per_update = ini.getint("mamba2", "n_steps_per_update")
    max_lr = ini.getfloat("mamba2", "max_lr")
    min_lr = ini.getfloat("mamba2", "min_lr")
    max_grad_norm = ini.getfloat("mamba2", "max_grad_norm")
    seed = ini.getint("mamba2", "seed")

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config = MambaConfig(
        d_model=d_model,
        d_intermediate=d_intermediate,
        n_layer=n_layer,
        vocab_size=vocab_size+1,  # to account for <pad> idx
        ssm_cfg=ssm_cfg,
        rms_norm=rms_norm,
        pad_vocab_size_multiple=pad_vocab_size_multiple
    )

    model = MambaLMHeadModel(config=config, device=device, dtype=dtype_dict[dtype])
    model.train()

    # logging
    ini_idx = sys.argv[1].split("/")[-1]
    logging.basicConfig(
        filename=f"logs/{ini_idx}.log",
        format="%(asctime)s %(message)s",
        filemode="w"
    )

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info({section: dict(ini[section]) for section in ini.sections()})
    logger.info(config)
    logger.info(model)

    n_params = 0
    for name, weight in model.named_parameters():
        curr_n_params = weight.numel()
        logger.info(f"{name}, {weight.shape}, {curr_n_params}, {weight.dtype}")
        n_params += curr_n_params

    logger.info(f"Total number of parameters: {n_params}")

    train_data, n_toks, n_examples, seq_lens = read_ids(
        train_fn,
        vocab_size,
        n_examples_per_step
    )

    assert(len(train_data) == len(n_toks))

    logger.info(f"Minimum sequence length: {min(seq_lens)}")
    logger.info(f"Maximum sequence length: {max(seq_lens)}")

    n_total_steps = len(train_data)
    n_total_updates = (len(train_data)//ini.getint("mamba2", "n_steps_per_update"))

    logger.info(f"Total number of examples: {n_examples}")
    logger.info(f"Total number of steps: {n_total_steps}")
    logger.info(f"Total number of updates: {n_total_updates}")

    loss_fn = torch.nn.NLLLoss(reduction="sum")
    sm = torch.nn.LogSoftmax(dim=-1)
    scaler = torch.cuda.amp.GradScaler()
    optim = torch.optim.AdamW(
        params=model.parameters(),
        lr=max_lr,
        betas=(0.9, 0.95),
        eps=1e-08,
        weight_decay=0.1
    )

    n_warmup_steps = (n_total_updates // 20)
    logger.info(f"Total number of warmup steps: {n_warmup_steps}")
    # LR scheduling...
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optim,
        start_factor=1/n_warmup_steps,
        end_factor=1.,
        total_iters=n_warmup_steps
    )
    decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim,
        T_max=n_total_updates-n_warmup_steps-1,
        eta_min=min_lr
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optim,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[n_warmup_steps]
    )

    # training loop
    n_updates = 0
    seq_start_idx = n_updates * n_examples_per_step * n_steps_per_update
    if (n_updates + 1) * n_examples_per_step * n_steps_per_update < n_examples:
        seq_end_idx = (n_updates + 1) * n_examples_per_step * n_steps_per_update
    else:
        seq_end_idx = n_examples
    curr_n_update_toks = sum(seq_lens[seq_start_idx:seq_end_idx])

    # initial save
    save_dir = f"models/{ini_idx}_{n_updates}"
    logger.info(f"Model checkpoint saved to models/{ini_idx}_{n_updates}!")
    model.save_pretrained(save_dir)
    t = torch.cuda.get_device_properties(0).total_memory

    save_at = {0}
    for i, (batch, batch_n_toks) in enumerate(zip(train_data, n_toks)):
        input_idx = batch[:, :-1].cuda()
        logger.info(f"Input batch size: {input_idx.shape}")
        target_idx = batch[:, 1:].cuda()
        with torch.autocast(device_type="cuda"):
            output = model(input_idx)
            log_probs = sm(output.logits).view(-1, vocab_size+1)
            target_idx = target_idx.view(-1)
            mask = (target_idx != vocab_size)
            loss = loss_fn(log_probs[mask].float(), target_idx[mask])
            # division by n_steps_per_update assumes each batch is weighted identically (regardless of batch_n_toks)
            # loss /= n_steps_per_update
            loss /= curr_n_update_toks 
        scaler.scale(loss).backward()
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        logger.info(f"Training step: {(i+1)}/{n_total_steps}, LR: {scheduler.get_last_lr()}, Loss: {loss.item()}, "
                    f"Update tokens: {curr_n_update_toks}, Total: {t}, Reserved: {r}, Allocated: {a}")

        if (i + 1) % n_steps_per_update == 0 or (i + 1) == n_total_steps:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()
            scheduler.step()
            logger.info(f"Weights updated after training step {(i+1)}!")
            n_updates += 1
            seq_start_idx = n_updates * n_examples_per_step * n_steps_per_update
            if (n_updates + 1) * n_examples_per_step * n_steps_per_update < n_examples:
                seq_end_idx = (n_updates + 1) * n_examples_per_step * n_steps_per_update
            else:
                seq_end_idx = n_examples
            curr_n_update_toks = sum(seq_lens[seq_start_idx:seq_end_idx])

            # if n_updates in save_at or n_updates == n_total_updates:
            if n_updates in save_at:
                save_dir = f"models/{ini_idx}_{n_updates}"
                logger.info(f"Model checkpoint saved to models/{ini_idx}_{n_updates}!")
                model.save_pretrained(save_dir)

    save_dir = f"models/{ini_idx}_{n_updates}"
    logger.info(f"Model checkpoint saved to models/{ini_idx}_{n_updates}!")
    model.save_pretrained(save_dir)


if __name__ == "__main__":
    main()
