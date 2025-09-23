# PathoLens

⚠️ This is a learning experiment; please don’t take it too seriously.

## Task
Given [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/), predict the pathogenicity of a variant. Many tools attempt to predict pathogenicity (e.g., CADD), and it is generally considered a hard problem because signal comes from diverse molecular and biological factors.

Here we reduce the task to a binary label (pathogenic/benign) and fine-tune an LLM to consume *virtual conditioning tokens* derived from a [1000 Genomes nucleotide transformer](https://huggingface.co/InstaDeepAI/nucleotide-transformer-500m-1000g) difference vector and protein-level descriptors built from Ensembl VEP + ESM2 embeddings.

## Intuition and basic idea

PathoLens separates “feature extraction” from “decision making.” The feature extractor produces a compact numeric summary of the variant: a nucleotide-transformer difference vector (`dna_alt – dna_ref`) that approximates local sequence perturbation, optionally concatenated with protein embeddings (`prot_mt - prot_wt`) aligned per variant. This joint vector is projected into a small set of *virtual tokens* that are prepended to the natural-language prompt. A decoder-only LLM is then fine-tuned (via LoRA) to read those tokens alongside a short textual context (gene symbol, HGVS, optional coarse consequence heuristics) and emit a single-word label.

Why this could work:

1. **Transformers attend over tokens.** Encoding the biology as a few learned embeddings lets the model attend to it at every layer, without adding a separate classifier head.
2. **The conditioning is informative.** `alt – ref` captures a local perturbation signal, and protein embeddings inject coarse functional context.
3. **LoRA is enough to retarget attention.** Low-rank adapters let the model learn to route from the textual scaffold to the conditioning without full retraining.
4. **Probabilities fall out naturally.** The target is literally a vocabulary word, so we can score `log P("Benign")` vs `log P("Pathogenic")` directly.

This is not a claim that the approach *works*. It’s a test of whether modest, structured biological signal, inserted in a way that matches how transformers operate, can move the needle on a hard classification task.

In fact, why this could *not* work:

“Pathogenicity” is not an intrinsic, context-free property of a variant. Clinical significance is adjudicated relative to a specific disease, inheritance model, and phenotype match in a particular person or family. The same allele can be pathogenic in a recessive disorder only when biallelic, benign in heterozygous carriers, or relevant only in a tissue, developmental stage, or environment that we never observe. The evidence that drives real classifications (e.g. segregation, de novo status, population frequency, phenotype specificity, functional readouts, etc.) is simply missing here. Asking a sequence-and-gene–only model to decide a question that biology defines conditionally sets it up to fail on many legitimate cases.

The sequence→effect mapping is also mediated by layers the model does not see: isoform usage, long-range regulation and 3D chromatin, cell-type–specific epigenetic state, gene–gene compensation, etc. Even “obvious” cues are unreliable in isolation. Nonsense can be tolerated; missense depends on structural context; splice effects hinge on unseen factors. In addition, ClinVar adds its own noise and bias (ascertainment toward extremes, disease-specific labeling). Compressing all of this to local sequence (and limited protein heuristics) is unlikely to resolve the borderline or mechanism-dependent examples that matter clinically.

## Implementation
### Data and splits

We restrict to GRCh38 and ClinVar assertions with “criteria provided, multiple submitters, no conflicts.” Variants are split **by gene** into train/validation/test to minimize leakage. Labels are collapsed to a binary scheme; VUS are excluded upstream.


### DNA embedding (reference/alternate FASTA windows)
We derive a **local sequence perturbation vector** from GRCh38 using the ClinVar VCF-style fields. For each variant, it opens the reference FASTA with `pyfaidx` and pulls a symmetric window around the locus (default ±512 bp, configurable). Coordinates are handled in **1-based VCF convention**: given `ChromosomeAccession`, `PositionVCF`, `ReferenceAlleleVCF`, and `AlternateAlleleVCF`, it slices `[pos−window … pos+len(ref)−1+window]` from the chromosome, verifies that the REF allele matches the FASTA at the expected offset, and **skips** any record that fails this check. The **alternate window** is synthesized in-place by replacing the REF span with ALT (so indels are handled naturally). All characters are upper-cased and non-IUPAC bases are sanitized to `N` to avoid tokenizer surprises. These **REF** and **ALT** windows are then embedded with the 1000G nucleotide transformer, and a single vector per window is obtained by masked mean pooling over token embeddings.

The final signal is the **normalized effect vector** `dna_eff = normalize(embed(ALT) − embed(REF))`, which emphasizes local changes while cancelling much of the background sequence. Only this effect array is persisted on disk (FP16 in a compressed NPZ), alongside a Feather file that caches the curated windows and minimal provenance (window size, FASTA path, row alignment). At training time, `dna_eff` is concatenated with any available protein embeddings to form the conditioning input supplied to the LLM as virtual tokens.

### Protein embedding
For protein-level features, PathoLens integrates ESM-2 embeddings (Meta AI’s protein language model). Using Ensembl VEP with the ProteinSeqs plugin, we extract both the wild-type and mutated protein sequences for each ClinVar variant. These sequences are embedded with ESM-2, and we compute a delta representation that captures the local effect of the mutation on the protein context. This embedding encodes biochemical and structural information (such as residue conservation, substitution severity, and domain context) that goes beyond simple VEP consequence labels. The resulting protein effect vectors are concatenated with DNA effect to form the conditioning input to our projector, which maps them into virtual tokens that the LLM consumes alongside the variant prompt.

### LLM fine-tuning
Good call—here’s the same blurb with the **aux head** described too:

The base model is **Qwen3-4B-Instruct-2507** loaded in 4-bit NF4. A lightweight **CondProjector** takes the concatenated conditioning vector and runs a minimal MLP (LayerNorm → Linear → GELU → Dropout → Linear) to produce **K** virtual token embeddings (default **K = 8**), which are **prepended** to the chat prompt so the decoder can attend to them at every layer. Fine-tuning uses LoRA adapters on attention and MLP blocks while the base weights stay frozen.

Alongside the tokens, the projector exposes a small **auxiliary classification head**: a single Linear over the flattened projection (`k*d_out → n_classes`). During training we combine the aux-head cross-entropy with the main LM loss (on the label word) using a modest weight. This gives the projector a direct discriminative signal, stabilizes early training, and encourages the projected tokens to be informative. The aux head is **only** a training aid; at inference we ignore it and score purely from the LM by comparing the log-likelihoods of the two label tokens (“Benign” vs “Pathogenic”).

## Results (Test set)
We evaluate PathoLens on ClinVar (GRCh38; “criteria provided, multiple submitters, no conflicts”), using gene-disjoint train/val/test splits. The model consumes only a compact conditioning vector and a short textual scaffold (HGVS, gene symbol, coarse consequence heuristics). Ground-truth labels are never inserted into prompts or scoring; we compute label log-likelihoods via teacher forcing on the two vocabulary targets (“Benign”, “Pathogenic”) and derive probabilities by softmaxing those two log-scores.

Results are reported on the held-out **test split** (N = 33,115 variants). Positive class = **Pathogenic**, negative class = **Benign**.

### Overall performance
#### MLP probe

| Mode      | Split | Thr  |  Acc  | BalAcc |   F1  | Prec  |  Rec  |  Spec |  ROC  |
|-----------|-------|------|-------|--------|-------|-------|-------|-------|-------|
| dna       | val   | 0.5  | 0.7613| 0.6557 | 0.4764| 0.5027| 0.4528| 0.8587| 0.7173|
|           | val   | best | 0.7279| 0.6572 | 0.4788| 0.4428| 0.5213| 0.7931| 0.7173|
| dna       | test  | 0.5  | 0.7713| 0.6532 | 0.4603| 0.4806| 0.4416| 0.8647| 0.7118|
|           | test  | best | 0.7340| 0.6526 | 0.4569| 0.4160| 0.5066| 0.7985| 0.7118|
| prot      | val   | 0.5  | 0.7446| 0.6323 | 0.4388| 0.4638| 0.4164| 0.8481| 0.6735|
|           | val   | best | 0.7837| 0.6433 | 0.4530| 0.5754| 0.3735| 0.9131| 0.6735|
| prot      | test  | 0.5  | 0.7947| 0.6752 | 0.4980| 0.5411| 0.4613| 0.8891| 0.7304|
|           | test  | best | 0.8251| 0.6757 | 0.5076| 0.6707| 0.4083| 0.9432| 0.7304|
| dna+prot  | val   | 0.5  | 0.7631| 0.7515 | 0.5962| 0.5042| 0.7293| 0.7737| 0.8291|
|           | val   | best | 0.8319| 0.7578 | 0.6371| 0.6604| 0.6155| 0.9002| 0.8291|
| dna+prot  | test  | 0.5  | 0.8160| 0.7954 | 0.6455| 0.5617| 0.7585| 0.8323| 0.8579|
|           | test  | best | 0.8685| 0.7845 | 0.6804| 0.7342| 0.6340| 0.9350| 0.8579|

This table show the results of a 2-layer Multi Layer Perceptron (MLP) on the raw concatenated embedding space (without the LLM).
See [mlp_test.py](./src/mlp_test.py). These results demonstrate that there is learnable signal in the embedding space, and that the concatenated vector outperforms each embedding space individually.

An earlier version ([3b65d2a](https://github.com/joelkuiper/PathoLens/commit/3b65d2a9859d138cf86adb37cfa9cc711cc6e093)) also used a node2vec embedding derived from the Gene Ontology (GO) and a Gene Annotation File (GAF), however it was decided to drop this feature due to poor performance.

#### LLM Full dataset
**Classification results (Test)**
| Metric    | Value |
| --------- | ----- |
| Accuracy  | 0.955 |
| Precision | 0.911 |
| Recall    | 0.881 |
| F1        | 0.896 |
| ROC–AUC   | 0.985 |

**Confusion matrix (Test)**
|                     | Pred Benign | Pred Pathogenic |
| ------------------- | ----------- | --------------- |
| **True Benign**     | 24,682      | 617             |
| **True Pathogenic** | 856         | 6,313           |


**Ablation results (Validation, n = 10,000)**
| Mode         | Acc    | F1     | ROC–AUC | PR–AUC |
| ------------ | ------ | ------ | ------- | ------ |
| cond+prompt  | 0.9392 | 0.8956 | 0.9827  | 0.9661 |
| cond\_only   | 0.7716 | 0.5388 | 0.7479  | 0.5885 |
| prompt\_only | 0.9265 | 0.8730 | 0.9763  | 0.9540 |

Δ vs cond+prompt:
- cond_only     ΔAcc=-0.1676  ΔF1=-0.3569  ΔROC-AUC=-0.2348  ΔPR-AUC=-0.3776
- prompt_only   ΔAcc=-0.0127  ΔF1=-0.0226  ΔROC-AUC=-0.0064  ΔPR-AUC=-0.0120

These results show that whilst the model is able to learn the separation between benign and pathogenic, it seems to do this almost exclusively on the prompt (only a small lift in prompt+cond from the ablation probe).

#### LLM missense only
Focusing on `most_severe_consequence=missense_variant` from VEP we obtain the following results.

**Classification report (Test)**
|                | precision | recall | f1-score | support |
|----------------|-----------|--------|----------|---------|
| **Benign**     | 0.877     | 0.876  | 0.877    | 3117    |
| **Pathogenic** | 0.812     | 0.814  | 0.813    | 2053    |
| **accuracy**   |           |        | 0.851    | 5170    |
| **macro avg**  | 0.845     | 0.845  | 0.845    | 5170    |
| **weighted avg** | 0.851   | 0.851  | 0.851    | 5170    |

**Confusion Matrix (Test)**

|                     | Pred Benign | Pred Pathogenic |
|---------------------|-------------|-----------------|
| **True Benign**     | 2730        | 387             |
| **True Pathogenic** | 382         | 1671            |


**Ablation test (Test)**
| Mode           |    N |   Acc  |   F1   | ROC-AUC | PR-AUC |
|----------------|------|--------|--------|---------|--------|
| cond+prompt    | 5170 | 0.8507 | 0.8118 |  0.9224 | 0.8972 |
| cond_only      | 5170 | 0.7062 | 0.4604 |  0.7734 | 0.7223 |
| prompt_only    | 5170 | 0.8087 | 0.7229 |  0.8897 | 0.8519 |
| cond+noise     | 5170 | 0.8135 | 0.7382 |  0.8871 | 0.8523 |
| cond+permute   | 5170 | 0.7768 | 0.7214 |  0.8480 | 0.7819 |

Here we do observe a lift in performance in `cond+prompt` but the prompt still carries considerable signal, further investigation is warranted in these results.

## Set-up and training
### Input

- [ClinVar variant summary](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz)
- [GRCh38 fna](https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.40_GRCh38.p14/GCF_000001405.40_GRCh38.p14_genomic.fna.gz)

To download them you can use:

```bash
mkdir -p data/raw/
wget -i sources.txt -P data/raw/
```

In addition it pulls the [nucleotide-transformer-500m-1000g model](https://huggingface.co/InstaDeepAI/nucleotide-transformer-500m-1000g), [ESM2](https://huggingface.co/facebook/esm2_t12_35M_UR50D) and [Qwen3-4B-Instruct-2507 model](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) from HuggingFace.

### Dependencies
Install the dependencies via [uv](https://docs.astral.sh/uv/) and activate the venv.

``` bash
uv sync
uv pip install '.[cuda]'
source .venv/bin/activate
```

### Pipeline configuration

All pipeline stages are configured via a single TOML file. An example is
available at [`configs/pipeline.example.toml`](./configs/pipeline.example.toml). Copy it
to a new location (e.g. `configs/pipeline.local.toml`) and update the paths to match your
environment:

- `[Paths]` points at the ClinVar TSV, GRCh38 FASTA, and the destination artifacts
  directory.
- `[DNA]` controls nucleotide-transformer windowing/encoding and cache overwrite flags.
- `[Protein]` enables the optional VEP → ESM2 pathway and includes Docker/cache settings.
- `[LLM]` configures the Qwen LoRA fine-tune.
- `[Run]` controls manifest writing and global overrides (e.g. forcing split regeneration).

Relative paths are resolved relative to the config file, and `~` is expanded everywhere.

> ℹ️ The Python dataclasses in [`src/pipeline/config.py`](./src/pipeline/config.py)
> simply mirror this schema for validation. Copy and edit the TOML file to tweak
> behaviour; the dataclass defaults only serve as fallbacks when a key is
> omitted.

### Training & Evaluation

Then build the caches and (optionally) run the LLM fine-tune using the unified pipeline
config:

``` bash
python train.py --config configs/pipeline.local.toml

```

Use `--device` to override the auto-detected accelerator (`cpu`, `cuda:0`, …) and
`--skip-train` to stop after cache + manifest creation. When enabled, a manifest JSON is
written to `Run.manifest` (defaults to `artifacts/pipeline_manifest.json`) describing all
derived artifacts. For a quick interactive look at the cached datasets, drop into
IPython and paste:

```python
from src.pipeline.datasets import load_manifest_datasets

cfg, manifest, datasets = load_manifest_datasets("configs/pipeline.local.toml")
train_ds = datasets["train"]
```

`load_manifest_datasets` uses the manifest location from the TOML by default, and
returns the parsed config alongside the manifest/dataset objects for further
inspection.

To reproduce the MLP ablation probe, load the datasets as above and run:

```python
from src.mlp_test import run_ablation_probes, print_probe_table

results = run_ablation_probes(datasets)
print_probe_table(results)
```

`src/mlp_test.py` also exposes `run_probes_from_config` which bundles the config
load, dataset construction, and probe execution in a single call.

The full fine-tune on a RTX 4090 takes roughly 6 hours; building caches and running
evaluation roughly doubles the wall-clock time.
