# PathoLens

⚠️ This is a learning experiment; please don’t take it too seriously.

## Task
Given [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/), predict the pathogenicity of a variant. Many tools attempt to predict pathogenicity (e.g., CADD), and it is generally considered a hard problem because signal comes from diverse molecular and biological factors.

Here we reduce the task to a binary label (pathogenic/benign) and fine-tune an LLM to consume *virtual conditioning tokens* derived from two sources: a [1000 Genomes nucleotide transformer](https://huggingface.co/InstaDeepAI/nucleotide-transformer-500m-1000g) and a [node2vec](https://en.wikipedia.org/wiki/Node2vec) embedding over the [Gene Ontology](https://www.geneontology.org/).

## Intuition and basic idea

PathoLens separates “feature extraction” from “decision making.” The feature extractor produces a compact numeric summary of the variant: a nucleotide-transformer difference vector (`dna_alt – dna_ref`) that approximates local sequence perturbation, concatenated with a node2vec embedding of the gene’s GO neighborhood. This joint vector is projected into a small set of *virtual tokens* that are prepended to the natural-language prompt. A decoder-only LLM is then fine-tuned (via LoRA) to read those tokens alongside a short textual context (gene symbol, HGVS, coarse consequence heuristics) and emit a single-word label.

Why this could work:

1. **Transformers attend over tokens.** Encoding the biology as a few learned embeddings lets the model attend to it at every layer, without adding a separate classifier head.
2. **The conditioning is informative.** `alt – ref` captures a local perturbation signal; the GO vector encodes coarse function and pathway context that often covary with impact.
3. **LoRA is enough to retarget attention.** Low-rank adapters let the model learn to route from the textual scaffold to the conditioning without full retraining.
4. **Probabilities fall out naturally.** The target is literally a vocabulary word, so we can score `log P("Benign")` vs `log P("Pathogenic")` directly.

This is not a claim that the approach *works*. It’s a test of whether modest, structured biological signal, inserted in a way that matches how transformers operate, can move the needle on a hard classification task.

In fact, why this could *not* work:

“Pathogenicity” is not an intrinsic, context-free property of a variant. Clinical significance is adjudicated relative to a specific disease, inheritance model, and phenotype match in a particular person or family. The same allele can be pathogenic in a recessive disorder only when biallelic, benign in heterozygous carriers, or relevant only in a tissue, developmental stage, or environment that we never observe. The evidence that drives real classifications (e.g. segregation, de novo status, population frequency, phenotype specificity, functional readouts, etc.) is simply missing here. Asking a sequence-and-gene–only model to decide a question that biology defines conditionally sets it up to fail on many legitimate cases.

The sequence→effect mapping is also mediated by layers the model does not see: isoform usage, long-range regulation and 3D chromatin, cell-type–specific epigenetic state, gene–gene compensation, etc. Even “obvious” cues are unreliable in isolation. Nonsense can be tolerated; missense depends on structural context; splice effects hinge on unseen factors. In addition, ClinVar adds its own noise and bias (ascertainment toward extremes, disease-specific labeling). Compressing all of this to local sequence plus a coarse GO prior is unlikely to resolve the borderline or mechanism-dependent examples that matter clinically.

## Input

- [Gene Association File](https://current.geneontology.org/annotations/goa_human.gaf.gz)
- [Gene Ontology JSON](https://purl.obolibrary.org/obo/go.json)
- [ClinVar variant summary](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz)
- [GRCh38 fna](https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.40_GRCh38.p14/GCF_000001405.40_GRCh38.p14_genomic.fna.gz)

To download them you can use:

```bash
mkdir -p data/raw/
wget -i sources.txt -P data/raw/
```

In addition it pulls the [nucleotide-transformer-500m-1000g model](https://huggingface.co/InstaDeepAI/nucleotide-transformer-500m-1000g) and [Qwen3-4B-Instruct-2507 model](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) from HuggingFace.

## Data and splits

We restrict to GRCh38 and ClinVar assertions with “criteria provided, multiple submitters, no conflicts.” Variants are split **by gene** into train/validation/test to minimize leakage. Labels are collapsed to a binary scheme; VUS are excluded upstream.

## Implementation

### GO embedding (Gene Ontology)
We derive **gene-level embeddings** from a heterogeneous GO graph constructed from the Gene Annotation File (GAF). Nodes represent genes and GO terms; edges connect each human gene (taxon 9606) to the GO terms it is annotated with, after removing `NOT` qualifiers and, optionally, restricting to curated evidence codes. To inject hierarchical context, the graph also includes GO **term–term** links limited to `is_a` and `part_of` relations parsed from the GO graph. To reduce hub effects and trivial shortcuts, the three GO root terms are dropped and very high-degree term nodes can be pruned; any isolates created by pruning are removed. A Node2Vec model is then trained on this graph, and only the gene embeddings are retained and L2-normalized. These vectors summarize each gene’s neighborhood in GO.


### DNA embedding (reference/alternate FASTA windows)
We derive a **local sequence perturbation vector** from GRCh38 using the ClinVar VCF-style fields. For each variant, it opens the reference FASTA with `pyfaidx` and pulls a symmetric window around the locus (default ±512 bp, configurable). Coordinates are handled in **1-based VCF convention**: given `ChromosomeAccession`, `PositionVCF`, `ReferenceAlleleVCF`, and `AlternateAlleleVCF`, it slices `[pos−window … pos+len(ref)−1+window]` from the chromosome, verifies that the REF allele matches the FASTA at the expected offset, and **skips** any record that fails this check. The **alternate window** is synthesized in-place by replacing the REF span with ALT (so indels are handled naturally). All characters are upper-cased and non-IUPAC bases are sanitized to `N` to avoid tokenizer surprises. These **REF** and **ALT** windows are then embedded with the 1000G nucleotide transformer, and a single vector per window is obtained by masked mean pooling over token embeddings.

The final signal is the **normalized effect vector** `dna_eff = normalize(embed(ALT) − embed(REF))`, which emphasizes local changes while cancelling much of the background sequence. Only this effect array is persisted on disk (FP16 in a compressed NPZ), alongside a Feather file that caches the curated windows and minimal provenance (window size, FASTA path, row alignment). At training time, `dna_eff` is concatenated with the gene’s GO embedding (`dna_eff ⊕ GO(gene)`) to form the conditioning input supplied to the LLM as virtual tokens.


### LLM fine-tuning
The base model is **Qwen3-4B-Instruct-2507** loaded in 4-bit NF4. A small projector (Linear → Tanh) maps the conditioning vector to **K** virtual token embeddings (default **K = 4**), which are prepended to the chat prompt. Fine-tuning uses LoRA on attention and MLP blocks.

The target is the **bare label word** (“Benign” or “Pathogenic”). All prompt tokens are masked, including any `<think>…</think>` block the Qwen chat template inserts; only the label span is supervised, and it up-weights that span during loss computation.


## Results (Test set)
We evaluate PathoLens on ClinVar (GRCh38; “criteria provided, multiple submitters, no conflicts”), using gene-disjoint train/val/test splits to minimize leakage across splits. The model consumes only a compact conditioning vector (dna_eff ⊕ GO(gene)) and a short textual scaffold (HGVS, gene symbol, coarse consequence heuristics). Ground-truth labels are never inserted into prompts or scoring; we compute label log-likelihoods via teacher forcing on the two vocabulary targets (“Benign”, “Pathogenic”) and derive probabilities by softmaxing those two log-scores.

Results are reported on the held-out **test split** (N = 33,115 variants). Positive class = **Pathogenic**, negative class = **Benign**.

### Overall performance

| Metric               |      Value |
|----------------------|-----------:|
| Accuracy             | **95.99%** |
| Precision            | **91.39%** |
| Recall               | **89.95%** |
| F1                   | **90.67%** |
| ROC–AUC              | **0.9899** |
| PR–AUC               | **0.9696** |

  ### Confusion matrix

|                        |    Pred: Benign | Pred: Pathogenic |  Row total |
| ---------------------- | --------------: | ---------------: | ---------: |
| **Actual: Benign**     | **25,330** (TN) |     **608** (FP) |     25,938 |
| **Actual: Pathogenic** |    **721** (FN) |   **6,456** (TP) |      7,177 |
| **Column total**       |          26,051 |            7,064 | **33,115** |

**Notes.** The model maintains a low false-positive rate (FP = 608; **FPR ≈ 2.34%**) while missing some positives (FN = 721; **FNR ≈ 10.05%**), indicating a slightly conservative bias toward the Benign label unless evidence is strong. The very high ROC–AUC and PR–AUC suggest robust ranking quality across thresholds.



## Set-up and training
First make sure you have downloaded all the required files.

Install the dependencies via [uv](https://docs.astral.sh/uv/).

``` bash
uv pip install '.[cuda]'
```

Next generate the GO node2vec embeddings:

``` bash
python go_node2vec.py \
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
If it complains `ImportError: 'Node2Vec' requires either the 'pyg-lib' or 'torch-cluster' package` try to install either.
For example:

``` bash
uv pip install pyg-lib -f https://data.pyg.org/whl/torch-2.8.0+cu129.html
```

Then generate the data and run the LLM fine-tune, which also calls the full evaluation.

``` bash
python train.py

```
The actual fine-tune takes about 6 hours on an Nvidia RTX 4090, however preparing the data and evaluating adds considerable overhead.
In total the runtime was about 12 hours, end to end.
