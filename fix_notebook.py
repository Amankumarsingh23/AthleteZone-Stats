import nbformat

path = "notebooks/04_ml_classification.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

source = nb.cells[0].source

if "import seaborn as sns" not in source:
    lines = source.split('\n')
    new_lines = []
    for line in lines:
        new_lines.append(line)
        if "import matplotlib.pyplot as plt" in line:
            new_lines.append("import seaborn as sns")
            
    nb.cells[0].source = '\n'.join(new_lines)
    with open(path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print("Successfully added seaborn import")
else:
    print("seaborn import already exists")
