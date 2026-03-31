import nbformat

path = "notebooks/04_ml_classification.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == "code" and "shap.summary_plot(shap_vals[2]" in cell.source:
        cell.source = cell.source.replace("shap.summary_plot(shap_vals[2]", "shap.summary_plot(shap_vals[:,:,2] if isinstance(shap_vals, np.ndarray) else shap_vals[2]")

with open(path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Successfully updated shap summary_plot argument.")
