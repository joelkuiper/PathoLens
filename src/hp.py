import json, re

_HP_IRI_RE = re.compile(r"(?:^|/)(HP)[:_](\d{7})$")  # .../HP_0000712 or HP:0000712


def _to_hp_id(x: str) -> str | None:
    if not isinstance(x, str):
        return None
    if x.startswith("HP:") and len(x) == 11 and x[3:].isdigit():
        return x
    m = _HP_IRI_RE.search(x)
    if m:
        return f"{m.group(1)}:{m.group(2)}"
    return None


def _label_from_node(node: dict) -> str | None:
    # OBO Graph nodes often have 'lbl' or in meta.basicPropertyValues as rdfs:label
    lbl = node.get("lbl") or node.get("label")
    if isinstance(lbl, str) and lbl.strip():
        return lbl.strip()
    meta = node.get("meta") or {}
    bpv = meta.get("basicPropertyValues") or []
    for p in bpv:
        pred = p.get("pred") or ""
        if "label" in pred or pred.endswith("#label"):
            val = p.get("val") or p.get("value") or p.get("@value")
            if isinstance(val, str) and val.strip():
                return val.strip()
    return None


def _obj_to_str(o):
    if isinstance(o, str):
        return o
    if isinstance(o, dict):
        # JSON-LD-ish literal
        return o.get("val") or o.get("value") or o.get("@value") or ""
    return ""


def load_hpo_id2label(path: str) -> dict[str, str]:
    """
    Supports:
      - OBO Graph JSON: {"graphs":[{"nodes":[{"id":"HP:0000118","lbl":"..."}, ...], "edges":[...]}]}
    """
    with open(path, "r") as f:
        data = json.load(f)

    id2label: dict[str, str] = {}

    # OBO Graph JSON
    if isinstance(data, dict) and "graphs" in data:
        for g in data.get("graphs", []):
            for n in g.get("nodes", []) or []:
                hp = _to_hp_id(n.get("id"))
                if not hp:
                    continue
                lab = _label_from_node(n)
                if lab:
                    id2label[hp] = lab

        for g in data.get("graphs", []):
            for e in g.get("edges", []) or []:
                if "label" in (e.get("pred") or "") or str(e.get("pred", "")).endswith(
                    "#label"
                ):
                    hp = _to_hp_id(e.get("sub"))
                    lab = _obj_to_str(e.get("obj"))
                    if hp and lab and lab.strip():
                        id2label.setdefault(hp, lab.strip())
        return id2label

    # Fallback: nothing matched
    return id2label
