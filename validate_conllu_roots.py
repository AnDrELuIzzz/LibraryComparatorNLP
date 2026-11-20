import sys

path = sys.argv[1]

sent_id = None
sent_lines = []
root_lines = []
bad_cols = []

def flush():
    global sent_id, sent_lines, root_lines, bad_cols
    if bad_cols:
        print("BAD-COLS:", sent_id, "exemplos=", len(bad_cols))
        for ln in bad_cols[:5]:
            print("  ", ln)
        sys.exit(0)

    if len(root_lines) > 1:
        print("MULTI-ROOT:", sent_id, "roots=", len(root_lines))
        for ln in root_lines[:10]:
            print("  ", ln)
        sys.exit(0)

    sent_id = None
    sent_lines = []
    root_lines = []
    bad_cols = []

with open(path, encoding="utf-8", errors="replace") as f:
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

        # conll18 espera 10 colunas; se vier diferente, já é bug do arquivo
        if len(cols) != 10:
            bad_cols.append(line)
            continue

        tok_id = cols[0]
        if "-" in tok_id or "." in tok_id:
            continue

        head = cols[6].strip()
        # se HEAD não for int, conll18 tende a falhar/interpretar errado; acusar aqui
        try:
            head_i = int(head)
        except Exception:
            bad_cols.append(line)
            continue

        if head_i == 0:
            root_lines.append(line)

flush()
print("OK: sem múltiplas raízes e sem linhas com colunas inválidas.")
