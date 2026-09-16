"""The train/validate/early-stop loop both torch baselines run."""
import torch


def fit(model, optim, tr_ldr, val_ldr, step, validate, save_path, epochs, patience):
    """Train until the val metric stops improving, checkpointing each improvement.

    `step(batch) -> loss` and `validate(loader) -> (metric, text)`, where a
    higher metric is better and `text` is appended to the per-epoch line. ->
    best metric.
    """
    best, pat = 0.0, 0
    for ep in range(1, epochs + 1):
        model.train()
        tloss, nb = 0, 0
        for batch in tr_ldr:
            loss = step(batch)
            optim.zero_grad()
            loss.backward()
            optim.step()
            tloss += loss.item()
            nb += 1

        model.eval()
        with torch.no_grad():
            metric, text = validate(val_ldr)
        print(f"  ep={ep:3d}  tr_loss={tloss/nb:.4f}  {text}")
        if metric > best:
            best, pat = metric, 0
            torch.save(model.state_dict(), save_path)
        else:
            pat += 1
            if pat >= patience:
                print(f"  early stop ep={ep}")
                break
    return best
