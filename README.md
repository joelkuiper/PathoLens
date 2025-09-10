# PathoLens

⚠️ This is a learning experiment; please don’t take it too seriously.

## Task
Given [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/), predict the pathogenicity of a variant. Many tools attempt this (e.g., CADD), and it is generally considered a hard problem because signal comes from diverse molecular and biological factors.

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

“Pathogenicity” is not an intrinsic, context-free property of a variant. Clinical significance is adjudicated relative to a specific disease, inheritance model, and phenotype match in a particular person or family. The same allele can be pathogenic in a recessive disorder only when biallelic, benign in heterozygous carriers, or relevant only in a tissue, developmental stage, or environment that we never observe. The evidence that drives real classifications (e.g. segregation, de novo status, allelic phase, zygosity, penetrance, population frequency, phenotype specificity, and functional readouts, etc.) is simply missing here. Asking a sequence-and-gene–only model to decide a question that biology defines conditionally sets it up to fail on many legitimate cases.

The sequence→effect mapping is also mediated by layers the model does not see: isoform usage, long-range regulation and 3D chromatin, cell-type–specific epigenetic state, dosage sensitivity and mechanism (loss- vs gain-of-function), domain-level biophysics, and gene–gene compensation. Even “obvious” cues are unreliable in isolation. Nonsense can be tolerated; missense depends on structural context; splice effects hinge on unseen factors. In addition, ClinVar adds its own noise and bias (ascertainment toward extremes, disease-specific labeling). Compressing all of this to local sequence plus a coarse GO prior is unlikely to resolve the borderline or mechanism-dependent examples that matter clinically.

## Data and splits

I restrict to GRCh38 and ClinVar assertions with “criteria provided, multiple submitters, no conflicts.” Variants are split **by gene** into train/validation/test to minimize leakage. Labels are collapsed to a binary scheme; VUS are excluded upstream.

## Implementation

### Node2Vec (GO embedding)
I construct a graph from GO relations and train node2vec; a gene’s GO vector is obtained by pooling its associated term embeddings. The representation is intentionally low-dimensional so the LLM treats it as context rather than a substitute classifier.

### DNA embedding
For each variant I extract 512 bp windows for reference and alternate alleles under GRCh38, embed both with the 1000G nucleotide transformer, and form `dna_eff = embed(alt) – embed(ref)`. Concatenate with the GO vector to get the final conditioning vector (`dna_eff ⊕ GO(gene)`).

### LLM fine-tuning
The base model is **Qwen3-4B-Instruct-2507** loaded in 4-bit NF4. A small projector (Linear → Tanh) maps the conditioning vector to **K** virtual token embeddings (default **K = 4**), which are prepended to the chat prompt. Fine-tuning uses LoRA on attention and MLP blocks.

The target is the **bare label word** (“Benign” or “Pathogenic”). All prompt tokens are masked, including any `<think>…</think>` block the Qwen chat template inserts; only the label span is supervised, and it up-weights that span during loss computation.
