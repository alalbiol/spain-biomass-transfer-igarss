"""
Fine-tune a Finland-trained AGB checkpoint on Spanish IFN4 pine plots with
province-grouped cross-validation (every plot is predicted by a model that
never saw any plot of its province).

Supervision is sparse: one number per chip, the plot AGB, compared with the
model's mean over the 25 m plot disk. To stop the network from learning
"the label is at the chip centre" (CNNs can read absolute position from
zero padding), training uses random 96 px crops that keep the disk inside,
plus random flips / 90-degree rotations; the disk weights move with them.
Evaluation uses the full 128 px chip with the plot at the centre.

Regimes (--regime):
  head     decoder + segmentation head trained, encoder and its attention frozen
  full     everything trained, encoder at lr/10
BatchNorm layers stay in eval mode (frozen running stats) in both: batches are
small and the statistics carry the Finland normalisation the weights expect.

No-pretraining baseline (--init imagenet|random, regime full only): the same
architecture, data, folds, crops, loss and epoch selection, but without the
Finland weights. Inputs are raw, so BatchNorm must learn their statistics:
it is trained (--train_bn, on by default for these inits).
Hyperparameters are fixed up front; the only selection is the epoch, chosen on
a random 10% of the TRAINING provinces' plots (never the held-out fold).

Output: <DATA>/finetune/<run>/oof.csv with the out-of-fold plot predictions
(and the per-fold model weights unless --no_save_weights).
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from ifn_common import DATA, N_FOLDS, PX, disk_weights, load_chip, load_model, metrics, plots

CROP = 96


class PlotDS(torch.utils.data.Dataset):
    def __init__(self, df, train, max_cloud=None):
        self.df, self.train, self.max_cloud = df.reset_index(drop=True), train, max_cloud
        self.w_full = disk_weights(PX)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        x, miss = load_chip(r.chip_id, max_cloud=self.max_cloud)
        w = self.w_full
        if self.train:
            oy, ox = np.random.randint(0, PX - CROP + 1, size=2)
            x, w = x[..., oy:oy + CROP, ox:ox + CROP], w[oy:oy + CROP, ox:ox + CROP]
            k = np.random.randint(4)
            x, w = np.rot90(x, k, axes=(-2, -1)), np.rot90(w, k)
            if np.random.rand() < 0.5:
                x, w = x[..., ::-1], w[:, ::-1]
        return (torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(miss),
                torch.from_numpy(np.ascontiguousarray(w)), torch.tensor(r.agb_mg_ha, dtype=torch.float32))


def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()


def predict(model, df, args):
    dl = torch.utils.data.DataLoader(PlotDS(df, False, args.max_cloud), batch_size=args.batch, num_workers=args.workers)
    model.eval()
    out = []
    with torch.no_grad():
        for x, miss, w, _ in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(x.cuda(), miss.cuda().clone()).float().squeeze(1)
            out.append((pred * w.cuda()).sum((1, 2)).cpu().numpy())
    return np.concatenate(out)


def train_fold(name, tr, va, args):
    model = load_model(name, init=args.init)
    enc = list(model.encoder.parameters()) + list(model.attn.parameters())
    dec = list(model.decoder.parameters()) + list(model.segmentation_head.parameters())
    if args.regime == "head":
        for p_ in enc:
            p_.requires_grad_(False)
        groups = [{"params": dec, "lr": args.lr}]
    else:
        groups = [{"params": enc, "lr": args.lr * args.enc_lr_mult}, {"params": dec, "lr": args.lr}]
    opt = torch.optim.AdamW(groups, weight_decay=args.wd)
    dl = torch.utils.data.DataLoader(PlotDS(tr, True, args.max_cloud), batch_size=args.batch, shuffle=True,
                                     num_workers=args.workers, drop_last=True, persistent_workers=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups],
                                                total_steps=args.epochs * len(dl) + 1, pct_start=0.1)
    best, best_state, hist = np.inf, None, []
    va_pred = predict(model, va, args)
    hist.append({"epoch": 0, "val_rmse": metrics(va_pred, va.agb_mg_ha)["rmse"]})
    print(f"    epoch 0 (zero-shot) inner-val RMSE {hist[-1]['val_rmse']:.2f}", flush=True)
    for ep in range(1, args.epochs + 1):
        model.train()
        if not args.train_bn:
            freeze_bn(model)
        t0, losses = time.time(), []
        for x, miss, w, y in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(x.cuda(non_blocking=True), miss.cuda().clone()).float().squeeze(1)
            plot_pred = (pred * w.cuda()).sum((1, 2))
            loss = torch.nn.functional.mse_loss(plot_pred, y.cuda())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p_ for g in groups for p_ in g["params"]], 5.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        va_pred = predict(model, va, args)
        rm = metrics(va_pred, va.agb_mg_ha)["rmse"]
        hist.append({"epoch": ep, "train_rmse": float(np.sqrt(np.mean(losses))), "val_rmse": rm})
        print(f"    epoch {ep} train RMSE {hist[-1]['train_rmse']:.2f} inner-val RMSE {rm:.2f} ({time.time()-t0:.0f}s)", flush=True)
        if rm < best:
            best, best_state = rm, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, hist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="efficientnet_early")
    p.add_argument("--regime", choices=["head", "full"], default="full")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max_cloud", type=float, default=None)
    p.add_argument("--folds", default=",".join(str(f) for f in range(N_FOLDS)))
    p.add_argument("--train_frac", type=float, default=1.0,
                   help="Subsample the training provinces' plots (learning-curve runs)")
    p.add_argument("--tag", default="")
    p.add_argument("--seed", type=int, default=0,
                   help="Training seed (init, augmentation, batch order); folds, inner-val plots and "
                        "train_frac subsets stay fixed. 0 reproduces the original runs; >0 adds _seed<n> to the run name")
    p.add_argument("--no_save_weights", action="store_true")
    p.add_argument("--init", choices=["finland", "imagenet", "random"], default="finland",
                   help="Starting weights; imagenet/random are the no-pretraining baselines")
    p.add_argument("--enc_lr_mult", type=float, default=0.1, help="Encoder+attention lr = lr * this (full regime)")
    p.add_argument("--train_bn", action=argparse.BooleanOptionalAction, default=None,
                   help="Train BatchNorm statistics (default: only for --init imagenet/random)")
    args = p.parse_args()
    if args.train_bn is None:
        args.train_bn = args.init != "finland"
    if args.init != "finland" and args.regime != "full":
        p.error("--init imagenet/random needs --regime full (a fresh encoder cannot be frozen)")
    torch.set_float32_matmul_precision("medium")
    if args.init == "finland":
        run = f"{args.model}_{args.regime}"
    else:
        run = f"{args.model}_scratch_{args.init}"
    run += (f"_frac{args.train_frac:g}" if args.train_frac < 1 else "") + args.tag
    run += f"_seed{args.seed}" if args.seed else ""
    out = DATA / "finetune" / run
    out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / "args.json", "w"), indent=1)
    df = plots()
    print(f"{run}: {len(df)} plots with chips", flush=True)
    oof, hists = [], {}
    for fold in map(int, args.folds.split(",")):
        test = df[df.fold == fold]
        rest = df[df.fold != fold].sample(frac=1.0, random_state=fold)
        n_va = int(0.1 * len(rest))
        va, tr = rest.iloc[:n_va], rest.iloc[n_va:]
        if args.train_frac < 1:
            tr = tr.sample(frac=args.train_frac, random_state=fold)
        print(f"  fold {fold}: test provinces {sorted(test.province_code.unique())} n={len(test)}, train {len(tr)}, inner-val {len(va)}", flush=True)
        torch.manual_seed(fold + 1000 * args.seed)
        np.random.seed(fold + 1000 * args.seed)
        model, hist = train_fold(args.model, tr, va, args)
        hists[fold] = hist
        pred = predict(model, test, args)
        part = test[["chip_id", "plot_id", "province_code", "fold", "main_pine", "agb_mg_ha"]].assign(pred=pred)
        oof.append(part)
        m = metrics(pred, test.agb_mg_ha)
        print(f"  fold {fold} held-out: RMSE {m['rmse']:.2f} bias {m['bias']:.2f} R2 {m['r2']:.3f}", flush=True)
        if not args.no_save_weights:
            torch.save(model.state_dict(), out / f"fold{fold}.pt")
        pd.concat(oof).to_csv(out / "oof.csv", index=False)
        json.dump(hists, open(out / "history.json", "w"), indent=1)
        del model
        torch.cuda.empty_cache()
    allp = pd.concat(oof)
    print(json.dumps({k: round(float(v), 3) for k, v in metrics(allp.pred, allp.agb_mg_ha).items()}))


if __name__ == "__main__":
    main()
