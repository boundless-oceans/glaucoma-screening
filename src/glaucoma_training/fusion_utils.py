"""Fixed probability averaging with strict name/order/label checks; no framework."""
import math


def blend_predictions(global_rows, local_rows, allow_subset=False, labeled=False):
    def validate(rows):
        names, probabilities = [], []
        for row in rows:
            name = row["Image_name"]
            value = float(row["Probability"])
            if not name or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Invalid prediction row")
            if labeled and row.get("True_Label") not in ("0", "1"):
                raise ValueError("Validation prediction requires binary labels")
            names.append(name); probabilities.append(value)
        if not names or len(names) != len(set(names)):
            raise ValueError("Empty/duplicate prediction names")
        return names, probabilities
    global_names, global_probs = validate(global_rows)
    names, local_probs = validate(local_rows)
    if allow_subset:
        selected = set(names)
        if [n for n in global_names if n in selected] != names:
            raise ValueError("Local debug rows must be an ordered subset of global rows")
    elif global_names != names:
        raise ValueError("Global/local prediction names or order differ")
    lookup = {r["Image_name"]: (r, p) for r,p in zip(global_rows, global_probs)}
    aligned = [lookup[name][1] for name in names]
    labels = None
    if labeled:
        labels = [int(r["True_Label"]) for r in local_rows]
        if any(lookup[name][0]["True_Label"] != row["True_Label"] for name,row in zip(names, local_rows)):
            raise ValueError("Global/local validation labels disagree")
    return names, aligned, local_probs, [(a+b)/2 for a,b in zip(aligned, local_probs)], labels
