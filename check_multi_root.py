import sys

path = sys.argv[1]
sent_id = None
roots = []

def flush():
    global sent_id, roots
    if roots and len(roots) > 1:
        print("MULTI-ROOT:", sent_id, "roots=", len(roots))
        for ln in roots:
            print("  ", ln)
        sys.exit(0)
    sent_id = None
    roots = []

with open(path, encoding="utf-8") as f:
    for raw in f:
        line = raw.rstrip("\n")
        if not line.strip():
            flush()
            continue
        if line.startswith("# sent_id"):
            sent_id = line.split("=", 1)[1].strip()
            continue
        if line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) != 10:
            continue
        if "-" in cols[0] or "." in cols[0]:
            continue
        if cols[6] == "0":
            roots.append(line)

flush()
print("OK: nenhuma sentença com múltiplas raízes.")

