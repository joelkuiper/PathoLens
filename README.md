# PathoLens

⚠️ This is a learning experiment; please don’t take it too seriously.

## Task
Given [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/), predict the pathogenicity of a variant. Many tools attempt to predict pathogenicity (e.g., CADD), and it is generally considered a hard problem because signal comes from diverse molecular and biological factors.

Here we reduce the task to a binary label (pathogenic/benign) and fine-tune an LLM to consume *virtual conditioning tokens* derived from a [1000 Genomes nucleotide transformer](https://huggingface.co/InstaDeepAI/nucleotide-transformer-500m-1000g) difference vector, a node2vec embedding of the gene's [Gene Ontology](https://www.geneontology.org/) neighbourhood, and optional protein-level descriptors built from [Ensembl VEP](https://www.ensembl.org/info/docs/tools/vep/index.html) + [ESM2](https://github.com/facebookresearch/esm) embeddings.

## Intuition and basic idea

PathoLens separates “feature extraction” from “decision making.” The feature extractor produces a compact numeric summary of the variant: a nucleotide-transformer difference vector (`dna_alt – dna_ref`) that approximates local sequence perturbation, concatenated with protein embeddings (`prot_mt - prot_wt`) aligned per variant. This joint vector is projected into a small set of *virtual tokens* that are prepended to the natural-language prompt. A decoder-only LLM is then fine-tuned (via LoRA) to read those tokens alongside a short textual context (gene symbol, HGVS, optional coarse consequence heuristics) and emit a single-word label.

Why this could work:

1. **Transformers attend over tokens.** Encoding the biology as a few learned embeddings lets the model attend to it at every layer, without adding a separate classifier head.
2. **The conditioning is informative.** `alt – ref` captures a local perturbation signal; the GO vector encodes broad functional context; protein embeddings inject finer structural detail when available.
3. **Probabilities fall out naturally.** The target is literally a vocabulary word, so we can score `log P("Benign")` vs `log P("Pathogenic")` directly.

This is not a claim that the approach *works*. It’s a test of whether modest, structured biological signal, inserted in a way that matches how transformers operate, can move the needle on a hard classification task.

In fact, why this could *not* work:

“Pathogenicity” is not an intrinsic, context-free property of a variant. Clinical significance is adjudicated relative to a specific disease, inheritance model, and phenotype match in a particular person or family. The same allele can be pathogenic in a recessive disorder only when biallelic, benign in heterozygous carriers, or relevant only in a tissue, developmental stage, or environment that we never observe. The evidence that drives real classifications (e.g. segregation, de novo status, population frequency, phenotype specificity, functional readouts, etc.) is simply missing here. Asking a sequence-and-gene–only model to decide a question that biology defines conditionally sets it up to fail on many legitimate cases.

The sequence→effect mapping is also mediated by layers the model does not see: isoform usage, long-range regulation and 3D chromatin, cell-type–specific epigenetic state, gene–gene compensation, etc. Even “obvious” cues are unreliable in isolation. Nonsense can be tolerated; missense depends on structural context; splice effects hinge on unseen factors. In addition, ClinVar adds its own noise and bias (ascertainment toward extremes, disease-specific labeling). Compressing all of this to local sequence (and limited protein heuristics) is unlikely to resolve the borderline or mechanism-dependent examples that matter clinically.

## Implementation
### Data and splits

We restrict to GRCh38 and ClinVar assertions with “criteria provided, multiple submitters, no conflicts.” Variants are split **by gene** into train/validation/test to minimize leakage. Labels are collapsed to a binary scheme; VUS are excluded upstream.


### GO embedding (Gene Ontology)
We derive **gene-level embeddings** from a heterogeneous GO graph constructed from the Gene Annotation File (GAF). Nodes represent genes and GO terms; edges connect each human gene (taxon 9606) to the GO terms it is annotated with, after removing `NOT` qualifiers and optionally restricting to curated evidence codes. To inject hierarchical context, the graph also includes GO **term–term** links limited to `is_a` relations parsed from the GO graph. We drop the three GO root terms and can prune very high-degree term nodes to reduce trivial shortcuts; any isolates created by pruning are removed. A Node2Vec model is then trained on this graph, and only the gene embeddings are retained and L2-normalized. These vectors summarize each gene’s neighbourhood in GO and provide a coarse prior about biological processes, molecular functions, and cellular components tied to the gene.

### DNA embedding (reference/alternate FASTA windows)
We derive a **local sequence perturbation vector** from GRCh38 using the ClinVar VCF-style fields. For each variant, it opens the reference FASTA with `pyfaidx` and pulls a symmetric window around the locus (default ±512 bp, configurable). Coordinates are handled in **1-based VCF convention**: given `ChromosomeAccession`, `PositionVCF`, `ReferenceAlleleVCF`, and `AlternateAlleleVCF`, it slices `[pos−window … pos+len(ref)−1+window]` from the chromosome, verifies that the REF allele matches the FASTA at the expected offset, and **skips** any record that fails this check. The **alternate window** is synthesized in-place by replacing the REF span with ALT (so indels are handled naturally). All characters are upper-cased and non-IUPAC bases are sanitized to `N` to avoid tokenizer surprises. These **REF** and **ALT** windows are then embedded with the 1000G nucleotide transformer, and a single vector per window is obtained by masked mean pooling over token embeddings.

The final signal is the **normalized effect vector** `dna_eff = normalize(embed(ALT) − embed(REF))`, which emphasizes local changes while cancelling much of the background sequence. Only this effect array is persisted on disk (FP16 in a compressed NPZ), alongside a Feather file that caches the curated windows and minimal provenance (window size, FASTA path, row alignment). At training time, `dna_eff` is concatenated with the gene’s GO embedding (and protein features when available) to form the conditioning input supplied to the LLM as virtual tokens.

### Protein embedding
For protein-level features, PathoLens integrates ESM-2 embeddings (Meta AI’s protein language model). Using Ensembl VEP with the ProteinSeqs plugin, we extract both the wild-type and mutated protein sequences for each ClinVar variant. These sequences are embedded with ESM-2, and we compute a delta representation after mean pooling that captures the local effect of the mutation on the protein context. This embedding encodes biochemical and structural information (such as residue conservation, substitution severity, and domain context) that goes beyond simple VEP consequence labels. The resulting protein effect vectors are concatenated with DNA effect to form the conditioning input to our projector, which maps them into virtual tokens that the LLM consumes alongside the variant prompt.

By default the code attempts to run VEP through Docker (by pulling the image from the Docker registry). It is possible to use `vep` from the `$PATH` instead by setting the flag in the configuration TOML.

When VEP produces multiple transcript consequence annotations for a single variant, we adhere to the following pick order by default: `mane_select, canonical, appris, ccds, rank, tsl, length`.

### LLM fine-tuning
The base model is **Qwen3-4B-Instruct-2507** loaded in 4-bit NF4. A lightweight **CondProjector** takes the concatenated conditioning vector and runs a minimal MLP (Linear → GELU → Dropout → Linear) to produce **K** virtual token embeddings (default **K = 8**), which are prepended to the chat prompt so the decoder can attend to them at every layer. Fine-tuning uses LoRA adapters on attention and MLP blocks while the base weights stay frozen.

Alongside the tokens, the projector exposes a small **auxiliary classification head**: a single Linear over the flattened projection (`k*d_out → n_classes`). During training we combine the aux-head cross-entropy with the main LM loss (on the label word) using a modest weight. This gives the projector a direct discriminative signal, stabilizes early training, and encourages the projected tokens to be informative. The aux head is **only** a training aid; at inference we ignore it and score purely from the LM by comparing the log-likelihoods of the two label tokens (“Benign” vs “Pathogenic”).

## Results (Test set)
We evaluate PathoLens on ClinVar (GRCh38; “criteria provided, multiple submitters, no conflicts”), using gene-disjoint train/val/test splits. The model consumes only a compact conditioning vector (DNA effect ⊕ GO(gene) ⊕ optional protein delta) and a short textual scaffold (HGVS, gene symbol, coarse consequence heuristics). Ground-truth labels are never inserted into prompts or scoring; we compute label log-likelihoods via teacher forcing on the two vocabulary targets (“Benign”, “Pathogenic”) and derive probabilities by softmaxing those two log-scores.

### Overall performance
#### MLP probe (Validation)

| Mode      | Thr  |  Acc  | BalAcc |   F1  | Prec  |  Rec  |  Spec |  ROC  |   PR  |
|-----------|------|------:|------:|------:|------:|------:|------:|------:|------:|
| dna       | 0.5  | 0.6947| 0.6482| 0.4675| 0.4018| 0.5587| 0.7376| 0.7115| 0.5279|
|           | best | 0.7350| 0.6518| 0.4710| 0.4518| 0.4920| 0.8116| 0.7115| 0.5279|
| go        | 0.5  | 0.6794| 0.5695| 0.3489| 0.3401| 0.3582| 0.7807| 0.6635| 0.3593|
|           | best | 0.6143| 0.6618| 0.4836| 0.3561| 0.7530| 0.5705| 0.6635| 0.3593|
| prot      | 0.5  | 0.7388| 0.6275| 0.4317| 0.4512| 0.4138| 0.8413| 0.6801| 0.4966|
|           | best | 0.7722| 0.6384| 0.4453| 0.5353| 0.3812| 0.8956| 0.6801| 0.4966|
| dna+go    | 0.5  | 0.7619| 0.6784| 0.5106| 0.5034| 0.5181| 0.8388| 0.7457| 0.5482|
|           | best | 0.7309| 0.6817| 0.5113| 0.4529| 0.5870| 0.7763| 0.7457| 0.5482|
| dna+prot  | 0.5  | 0.7736| 0.7655| 0.6137| 0.5193| 0.7499| 0.7810| 0.8376| 0.6884|
|           | best | 0.8168| 0.7731| 0.6434| 0.6033| 0.6892| 0.8570| 0.8376| 0.6884|
| go+prot   | 0.5  | 0.7011| 0.6455| 0.4636| 0.4069| 0.5387| 0.7523| 0.7332| 0.5562|
|           | best | 0.6055| 0.6681| 0.4894| 0.3548| 0.7885| 0.5477| 0.7332| 0.5562|
| dna+go+prot| 0.5 | 0.7561| 0.7630| 0.6042| 0.4945| 0.7764| 0.7496| 0.8405| 0.6957|
|           | best | 0.8142| 0.7703| 0.6391| 0.5981| 0.6860| 0.8546| 0.8405| 0.6957|

This table show the results of a 2-layer Multi Layer Perceptron (MLP) on the raw concatenated embedding space (without the LLM).
See [mlp_test.py](./src/mlp_test.py). These results demonstrate that there is learnable signal in the embedding space, and that the concatenated vector outperforms each embedding space individually.


#### LLM Full dataset

**Classification results (Test)**

| Metric    | Value |
| --------- | ----: |
| Accuracy  | 0.948 |
| Precision | 0.903 |
| Recall    | 0.859 |
| F1        | 0.880 |
| ROC–AUC   | 0.983 |
| PR–AUC    | 0.953 |

**Confusion matrix (Test)**

|                     | Pred Benign | Pred Pathogenic |
| ------------------- | ----------: | --------------: |
| **True Benign**     |      24,635 |             664 |
| **True Pathogenic** |       1,012 |           6,157 |

**Ablation results (Test, n = 5,000)**

| Mode             |    Acc |     F1 | ROC–AUC | PR–AUC |
| ---------------- | -----: | -----: | ------: | -----: |
| cond+prompt      | 0.8886 | 0.8360 |  0.9568 | 0.9337 |
| cond_only        | 0.7684 | 0.6114 |  0.8141 | 0.7335 |
| prompt_only      | 0.8476 | 0.7434 |  0.9429 | 0.9140 |
| cond_zero_dna    | 0.8822 | 0.8247 |  0.9575 | 0.9342 |
| cond_zero_prot   | 0.8666 | 0.7897 |  0.9448 | 0.9169 |
| cond_zero_go     | 0.8866 | 0.8302 |  0.9559 | 0.9325 |
| prompt_no_hgvsp  | 0.8810 | 0.8240 |  0.9511 | 0.9250 |
| prompt_no_hgvsc  | 0.8844 | 0.8302 |  0.9555 | 0.9323 |
| prompt_no_gene   | 0.8848 | 0.8462 |  0.9586 | 0.9316 |
| cond+noise       | 0.8626 | 0.7876 |  0.9421 | 0.9128 |

Δ vs **cond+prompt**:

- **cond_only** ΔAcc = −0.1202, ΔF1 = −0.2246, ΔROC–AUC = −0.1427, ΔPR–AUC = −0.2002
- **prompt_only** ΔAcc = −0.0410, ΔF1 = −0.0926, ΔROC–AUC = −0.0139, ΔPR–AUC = −0.0197
- **cond_zero_prot** ΔAcc = −0.0220, ΔF1 = −0.0464, ΔROC–AUC = −0.0120, ΔPR–AUC = −0.0169
- **cond_zero_go** ΔAcc = −0.0020, ΔF1 = −0.0058, ΔROC–AUC = −0.0009, ΔPR–AUC = −0.0012
- **cond_zero_dna** ΔAcc = −0.0064, ΔF1 = −0.0114, ΔROC–AUC = +0.0007, ΔPR–AUC = +0.0005

> Note: On the full dataset the prompt includes short VEP-derived fields (e.g., HGVS, consequence term), so prompt-only remains strong; adding conditioning still provides a clear lift. Protein contributes most on average.

#### LLM — Missense only

Focusing on `most_severe_consequence = missense_variant` from VEP we obtain the following results. We emphasize missense variants because they are biologically and clinically an interesting class:

* **Sheer prevalence** missense substitutions are a common type of coding variant in ClinVar and in human genomes.
* **Ambiguous functional effect**  unlike loss-of-function (nonsense, frameshift, canonical splice) variants, which often have predictable outcomes, missense changes can be benign or highly pathogenic depending on subtle context (conservation, domain, structure, biochemical compatibility).
* **Limited prompt signal** the HGVS protein notation (e.g. `p.Gly12Asp`) alone does not convey whether the substitution is harmful. Models trained only on text struggle here, tending towards random or majority-class behavior.

By concentrating evaluation on missense variants, we test whether the conditioning path (DNA + protein effect vectors projected into virtual tokens) actually provides *non-trivial discriminative signal*. A lift on this subset is strong evidence that the model is not just memorizing textual heuristics but genuinely leveraging the embeddings.

Here’s the corrected **Missense-only** block for your README:

**Classification report (Test)**

|                  | precision | recall | f1-score | support |
| ---------------- | --------: | -----: | -------: | ------: |
| **Benign**       |     0.876 |  0.907 |    0.891 |    3117 |
| **Pathogenic**   |     0.850 |  0.806 |    0.827 |    2053 |
| **accuracy**     |           |        |    0.867 |    5170 |
| **macro avg**    |     0.863 |  0.856 |    0.859 |    5170 |
| **weighted avg** |     0.866 |  0.867 |    0.866 |    5170 |

**Confusion matrix (Test)**

|                     | Pred Benign | Pred Pathogenic |
| ------------------- | ----------: | --------------: |
| **True Benign**     |       2,826 |             291 |
| **True Pathogenic** |         399 |           1,654 |

**Ablation results (Test, n = 5,170)**

| Mode            |    Acc |     F1 | ROC–AUC | PR–AUC |
| --------------- | -----: | -----: | ------: | -----: |
| cond+prompt     | 0.8660 | 0.8270 |  0.9258 | 0.8999 |
| cond_only       | 0.6876 | 0.3796 |  0.7651 | 0.7182 |
| prompt_only     | 0.7890 | 0.6658 |  0.8867 | 0.8520 |
| cond_zero_dna   | 0.8673 | 0.8277 |  0.9279 | 0.9024 |
| cond_zero_prot  | 0.8315 | 0.7691 |  0.8987 | 0.8557 |
| cond_zero_go    | 0.8431 | 0.7791 |  0.9187 | 0.8943 |
| prompt_no_hgvsp | 0.8418 | 0.7774 |  0.9093 | 0.8746 |
| prompt_no_hgvsc | 0.8609 | 0.8162 |  0.9244 | 0.8995 |
| prompt_no_gene  | 0.7530 | 0.6564 |  0.8068 | 0.7592 |
| cond+noise      | 0.7876 | 0.7269 |  0.8554 | 0.8024 |
```
On the missense-only set the model reaches ~86.6% accuracy and F1 ≈ 0.83, showing that conditioning provides clear discriminative signal beyond the prompt strings alone. The ablations confirm that protein embeddings carry the strongest marginal contribution, while GO adds smaller but consistent value; prompt-only performance remains decent, but conditioning lifts both calibration and recall, indicating the model genuinely leverages the embedding features for these harder cases. Interestingly, removing the gene symbol from the prompt (prompt_no_gene) causes one of the largest drops (F1 −0.17), highlighting that the model relies heavily on gene context when reasoning about missense pathogenicity.


## Set-up and training
### Input

- [ClinVar variant summary](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz)
- [GRCh38 fna](https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.40_GRCh38.p14/GCF_000001405.40_GRCh38.p14_genomic.fna.gz)
- [Gene Association File (GAF)](https://current.geneontology.org/annotations/goa_human.gaf.gz)
- [Gene Ontology JSON graph](https://purl.obolibrary.org/obo/go.json)

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
- `[Protein]` controls the VEP → ESM2 pathway and includes Docker/cache settings.
- `[LLM]` configures the Qwen LoRA fine-tune.
- `[Run]` controls manifest writing and global overrides (e.g. forcing split regeneration).

Relative paths are resolved relative to the config file, and `~` is expanded everywhere.

> ℹ️ The Python dataclasses in [`src/pipeline/config.py`](./src/pipeline/config.py)
> simply mirror this schema for validation. Copy and edit the TOML file to tweak
> behaviour; the dataclass defaults only serve as fallbacks when a key is
> omitted.

### Training & Evaluation

Generate the GO node2vec embeddings once (adjust hyperparameters as needed):

``` bash
python -m src.go.go_node2vec \
  --go-json data/raw/go.json \
  --gaf data/raw/goa_human.gaf.gz \
  --out-prefix data/processed/go_n2v \
  --dim 256 \
  --epochs 20 \
  --walk-len 40 \
  --walks-per-node 5 \
  --ctx-size 5 \
  --neg-samples 2 \
  --batch-size 256 \
  --drop-roots \
  --prune-term-degree 200
```

Then build the caches and (optionally) run the LLM fine-tune using the unified pipeline
config:

``` bash
python train.py --config configs/pipeline.local.toml

```

Use `--device` to override the auto-detected accelerator (`cpu`, `cuda:0`, …) and
`--skip-train` to stop after cache + manifest creation. When enabled, a manifest JSON is
written to `Run.manifest` (defaults to `artifacts/pipeline_manifest.json`) describing all
derived artifacts.

The full fine-tune on a RTX 4090 takes roughly 6 hours; building caches and running
evaluation roughly doubles the wall-clock time.

For a quick interactive look at the cached datasets, drop into IPython and paste:

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
