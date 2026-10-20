import argparse
from pathlib import Path

import open_clip

from configs.config import PROMPTS, Config
from eval.eval import evaluate
from eval.retrieval import embed_tiles, load_model, score_prompts
from results.results import fmt, make_run_name, print_summary, save_run

def run_zs():
    cfg = Config()

    # Argument Parsing
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default=cfg.tsv_path, help="TSV written by prep_data")
    ap.add_argument("--prompt", nargs="+", choices=list(PROMPTS), default=list(PROMPTS), help="which PROMPTS keys to run (default: all of them)")
    ap.add_argument("--overwrite", action="store_true", help="replace run folders that already exist")
    args = ap.parse_args()

    prompts = {key: PROMPTS[key] for key in args.prompt}
    names = {key:make_run_name("zs", cfg.model_name, cfg.pretrained, key) for key in prompts}

    device = cfg.device
    model, preprocess, tokenizer = load_model(cfg.model_name, cfg.pretrained, device)

    # Push tsv files through image encoder -- just once
    emb = embed_tiles(model, preprocess, args.tsv, device, cfg.batch_size, cfg.num_workers)
    counts = dict(zip(emb.type_names, emb.labels.sum(axis=0).astype(int).tolist()))

    # Score each prompt in PROMPTS
    rows = []
    for key, template in prompts.items():
        scores = score_prompts(model, tokenizer, emb, template, device)
        metrics = evaluate(scores, emb.labels, emb.type_names, cfg.ks)
        settings = {
            "train": "zs", # zero shot
            "model_name": cfg.model_name,
            "pretrained": cfg.pretrained, 
            "prompt_key": key,
            "template": template,
            "tsv_path": args.tsv,
            "n_tiles": len(emb.tile_ids),
            "type_names": emb.type_names,
            "ks": cfg.ks,
            "batch_size": cfg.batch_size,
            "device": device,
            "open_clip_version": getattr(open_clip, "__version__", None),
        }
        save_run(names[key], settings, metrics, cfg.runs_dir, overwrite=args.overwrite)
        print_summary(metrics)
        rows.append((names[key], metrics["mAP"], metrics["image_to_text_recall@1"]))

if __name__ == "__main__":
    run_zs()