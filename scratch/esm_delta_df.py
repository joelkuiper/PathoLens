# scratch/esm_delta_df.py
import math, re
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

ESM_MODEL_ID = "facebook/esm2_t12_35M_UR50D"  # small & fast

# --- HGVS.p parsing ---
RX_HGVSP = re.compile(r"p\.([A-Za-z]{3})(\d+)([A-Za-z]{3}|\*)", re.I)
AA3_TO_1 = {
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Glu": "E",
    "Gln": "Q",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V",
    "Sec": "U",
    "Ter": "*",
    "Xaa": "X",
}


def parse_hgvsp(pstr: str) -> Optional[Tuple[str, int, str]]:
    m = RX_HGVSP.search(pstr or "")
    if not m:
        return None
    wt3, pos, mut3 = m.groups()
    wt1 = AA3_TO_1.get(wt3.capitalize())
    mut1 = AA3_TO_1.get(mut3.capitalize())
    if wt1 is None or mut1 is None:
        return None
    try:
        pos = int(pos)
    except Exception:
        return None
    return wt1, pos, mut1


# --- Build gene→(entry, seq) from a UniProt DF (no I/O) ---
def build_protein_index_from_uniprot_df(
    df, organism: Optional[str] = "Homo sapiens"
) -> Dict[str, Tuple[str, str]]:
    use = df.copy()
    if organism and "Organism" in use.columns:
        use = use[use["Organism"].astype(str).str.contains(organism, na=False)]
    # first token in "Gene Names" is primary symbol
    use["gene_primary"] = use["Gene Names"].fillna("").str.split().str[0]
    use = use[
        (use["gene_primary"] != "") & use["Sequence"].notna() & (use["Sequence"] != "")
    ]
    use["is_reviewed"] = (
        use["Reviewed"].astype(str).str.lower().eq("reviewed").astype(int)
    )
    use["seq_len"] = use["Sequence"].str.len()
    # Prefer reviewed, then longest sequence
    use = use.sort_values(
        ["gene_primary", "is_reviewed", "seq_len"], ascending=[True, False, False]
    )
    best = use.drop_duplicates(subset=["gene_primary"], keep="first")
    return {
        g: (acc, seq)
        for g, acc, seq in zip(best["gene_primary"], best["Entry"], best["Sequence"])
    }


# --- ESM scorer (stateful, caches model/tokenizer) ---
class ESMDeltaScorer:
    def __init__(
        self,
        uniprot_df,
        organism: Optional[str] = "Homo sapiens",
        device: Optional[str] = None,
    ):
        self.prot_index = build_protein_index_from_uniprot_df(
            uniprot_df, organism=organism
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(ESM_MODEL_ID, do_lower_case=False)
        self.mdl = (
            AutoModelForMaskedLM.from_pretrained(ESM_MODEL_ID).to(self.device).eval()
        )
        self.vocab = self.tok.get_vocab()
        self._cache_tokens: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = (
            {}
        )  # gene -> (input_ids, attn_mask)

    def _aa_id(self, aa: str) -> Optional[int]:
        return self.vocab.get(aa, None)

    def _tokenize_gene(self, gene: str):
        if gene in self._cache_tokens:
            return self._cache_tokens[gene]
        _, seq = self.prot_index[gene]
        spaced = " ".join(list(seq))
        enc = self.tok(spaced, return_tensors="pt")
        input_ids = enc.input_ids.to(self.device)  # (1, L+2)
        attn_mask = enc.attention_mask.to(self.device)  # (1, L+2)
        self._cache_tokens[gene] = (input_ids, attn_mask)
        return input_ids, attn_mask

    @torch.no_grad()
    def delta_logp(
        self, gene: str, wt: str, pos1: int, mut: str
    ) -> Tuple[bool, float, float, float]:
        """ΔlogP = log P(mut|seq,pos) − log P(wt|seq,pos). Returns (ok, delta, P(wt), P(mut))."""
        if gene not in self.prot_index:
            return False, float("nan"), float("nan"), float("nan")
        _, seq = self.prot_index[gene]
        L = len(seq)
        if not (1 <= pos1 <= L):
            return False, float("nan"), float("nan"), float("nan")
        if seq[pos1 - 1] != wt:
            return False, float("nan"), float("nan"), float("nan")

        inp, mask = self._tokenize_gene(gene)
        masked = inp.clone()
        masked[0, pos1] = self.tok.mask_token_id  # BOS at 0 → site at pos1
        out = self.mdl(input_ids=masked, attention_mask=mask)
        log_probs = out.logits[0, pos1].log_softmax(-1)
        id_wt = self._aa_id(wt)
        id_mut = self._aa_id(mut)
        if id_wt is None or id_mut is None:
            return False, float("nan"), float("nan"), float("nan")
        lp_wt, lp_mut = float(log_probs[id_wt]), float(log_probs[id_mut])
        return True, lp_mut - lp_wt, math.exp(lp_wt), math.exp(lp_mut)

    def build_block(self, split) -> np.ndarray:
        """Return (N,1) ΔlogP vector for a split; NaNs → 0."""
        meta = split.meta
        out: List[float] = []
        for i in range(len(split)):
            row = meta.iloc[i]
            gene = (row.get("GeneSymbol") or row.get("Gene") or "").strip()
            pstr = None
            for k in ("HGVS.p", "HGVSp", "ProteinChange", "Name"):
                v = row.get(k)
                if isinstance(v, str) and "p." in v:
                    pstr = v
                    break
            parsed = parse_hgvsp(pstr or "")
            if not gene or not parsed:
                out.append(0.0)
                continue
            wt, pos, mut = parsed
            ok, dlp, _, _ = self.delta_logp(gene, wt, pos, mut)
            out.append(0.0 if (not ok or not np.isfinite(dlp)) else float(dlp))
        return np.asarray(out, dtype=np.float32).reshape(-1, 1)


def mapping_coverage_from_df(
    split, uniprot_df, organism: Optional[str] = "Homo sapiens"
):
    """Quick coverage/mismatch stats against DF index."""
    idx = build_protein_index_from_uniprot_df(uniprot_df, organism=organism)
    meta = split.meta
    tot = ok = mismatch = oob = 0
    for i in range(len(split)):
        row = meta.iloc[i]
        gene = (row.get("GeneSymbol") or row.get("Gene") or "").strip()
        pstr = None
        for k in ("HGVS.p", "HGVSp", "ProteinChange", "Name"):
            v = row.get(k)
            if isinstance(v, str) and "p." in v:
                pstr = v
                break
        parsed = parse_hgvsp(pstr or "")
        if not gene or not parsed or gene not in idx:
            continue
        wt, pos, _ = parsed
        _, seq = idx[gene]
        tot += 1
        if 1 <= pos <= len(seq):
            ok += int(seq[pos - 1] == wt)
            mismatch += int(seq[pos - 1] != wt)
        else:
            oob += 1
    return {"n_with_hgvsp": tot, "ok": ok, "mismatch_wt": mismatch, "pos_oob": oob}
