from train import OUT_DIR, FASTA_PATH, GO_NPZ, prepare_clinvar_splits, RunCfg
from util import get_device

device = get_device()
cfg = RunCfg(
    out_dir=OUT_DIR,
    fasta_path=FASTA_PATH,
    go_npz=GO_NPZ,
    epochs=1,
    force_dna=False,
)
cfg.out_dir.mkdir(parents=True, exist_ok=True)
print(f"Device: {device}")


train_df, val_df, test_df = prepare_clinvar_splits(force=False)
splits = {"train": train_df, "val": val_df, "test": test_df}
print(f"Splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")


from src.protein_utils_runner import build_protein_caches

protein_cache = build_protein_caches(
    cfg,
    device,
    splits,
    vep_cache_dir="/Users/j.kuiper01/vep_cache",
    vep_fasta_relpath="homo_sapiens/115_GRCh38/Homo_sapiens.GRCh38.dna.toplevel.fa.gz",
    image="ensemblorg/ensembl-vep",
    filter_mode="all",  # "protein_changing" / "patchable" / "all"
    chunk_size=1000,
    jobs=6,
    vep_fork=0,
    esm_model_id="facebook/esm2_t12_35M_UR50D",  # esm2_t33_650M_UR50D = much bigger
    bs_cuda=16,
    bs_cpu=8,
    max_len=2048,
    pool="mean",
)
