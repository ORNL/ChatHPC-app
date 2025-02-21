# %%
import os
from subprocess import check_output

from chathpc.app import App as ChatApp

GIT_ROOT = check_output("git rev-parse --show-toplevel", shell=True).decode().strip()
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

app = ChatApp.from_json(
    os.path.join(GIT_ROOT, "tests/files/config.json"),
    {"prompt_template_file": os.path.join(GIT_ROOT, "tests/files/prompt_template.txt")},
)

# %%
app.load_datasets()

# %%
df = app.train_dataset.to_pandas()

df["prompt_filled"] = df.apply(lambda row: app.training_prompt(**row), axis=1)

# %%
df["prompt_filled"].to_markdown("df.md")

# %%
with open("out.md", "w") as fd:
    fd.write("Output2\n")
    fd.write("----\n\n")
    for index, row in df.iterrows():
        fd.write(app.extract_answer(row["prompt_filled"], **row) + "\n")
        fd.write("\n----\n\n")

# %%
