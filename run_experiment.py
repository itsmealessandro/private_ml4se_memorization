import json
import os

notebook_path = "main.ipynb"

with open(notebook_path, "r") as f:
    nb = json.load(f)

code_content = ""
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        # Skip magic commands
        lines = source.splitlines()
        clean_lines = [l for l in lines if not l.strip().startswith("!")]
        code_content += "\n".join(clean_lines) + "\n\n"

with open("full_experiment_script.py", "w") as f:
    f.write(code_content)

print("Created full_experiment_script.py. Running it...")
os.system("./.venv/bin/python full_experiment_script.py")
