# C-DIRA One-Run Google Colab Design

## Goal

Provide one repository-maintained Python script that a user can launch from a
single Google Colab cell:

```python
%run scripts/run_colab.py
```

The script takes a fresh Colab runtime from Kaggle authentication through
dataset preparation, CUDA training, evaluation, and Markdown/HTML report
generation. The repository must already be cloned or uploaded, and the
notebook working directory must be the repository root.

## Scope

The runner will:

1. verify that it is running from the C-DIRA repository root;
2. verify a compatible Python version and a usable CUDA GPU;
3. install the local project and its runtime dependencies;
4. request `kaggle.json` through the Colab upload dialog when valid Kaggle
   credentials are not already installed;
5. download and extract the State Farm Distracted Driver Detection dataset;
6. validate the extracted `data/raw/train/c0` through `c9` layout;
7. prepare manifests and pseudo-domains using the existing pipeline;
8. train and evaluate C-DIRA and the MobileNetV3 baseline;
9. generate the existing Markdown and self-contained HTML reports; and
10. print artifact paths and Colab download links when possible.

It will not embed credentials, modify the model architecture, add temporal
video modeling, mount Google Drive, or attempt to bypass Kaggle competition
rules.

## Architecture

Add `scripts/run_colab.py` as a thin orchestration layer. It will use only the
Python standard library until the editable project installation completes.
After installation, it will invoke the existing `cdira` CLI through
subprocesses. This keeps dataset, training, evaluation, and reporting behavior
inside the existing tested application rather than duplicating it in notebook
code.

Add `configs/colab.yaml` as an explicit, reviewable derivative of the standard
profile. It will select:

- `training.device: cuda`;
- `training.mixed_precision: true`;
- `data.num_workers: 2`;
- paths rooted in the repository's `data/` and `artifacts/colab`; and
- the existing standard ten-epoch experiment settings.

The runner will not generate or silently mutate tracked YAML during a run.

## Stage and Rerun Behavior

The runner will print a numbered heading and elapsed time for every stage.
Subprocess output will be streamed directly so model download, pseudo-domain
extraction, batch progress, validation progress, and epoch summaries remain
visible.

Stages will be restart-friendly:

- Dependency installation may run again safely.
- Existing credentials will be reused without printing their contents.
- An existing dataset will be validated and reused; it will only be downloaded
  if validation fails because the expected dataset is absent.
- Existing manifests and feature caches are reused by the current pipeline
  where supported.
- Training writes to a fresh timestamped run below `artifacts/colab`, avoiding
  accidental checkpoint overwrite.

The runner is restart-friendly rather than a training-resume implementation:
an interrupted model-training stage starts a new training run.

## Kaggle Authentication and Download

When `/root/.kaggle/kaggle.json` is not present, the runner imports
`google.colab.files`, opens the upload picker, and requires exactly one file
named `kaggle.json`. It writes the file to `/root/.kaggle/kaggle.json` with
mode `0600` and does not log its contents.

The existing `cdira data download --config configs/colab.yaml` command performs
the competition download. Authentication failures and HTTP 401/403 responses
will be translated into a concise explanation covering:

- invalid or missing Kaggle credentials;
- the need to sign into Kaggle and accept the State Farm competition rules;
  and
- rerunning the notebook after fixing access.

No retry loop will repeatedly issue a forbidden request.

## Validation and Error Handling

The script will stop before expensive work when:

- the repository root or required files cannot be found;
- Python is outside the project's supported range;
- CUDA is unavailable;
- the uploaded credential file is missing or malformed;
- Kaggle denies access; or
- the extracted dataset fails the existing layout validator.

Command failures will retain their original output and end with the failed
stage name. The script will never display credential values.

## Notebook Output

At completion, the notebook displays:

- total elapsed time;
- the concrete run directory;
- checkpoint path;
- Markdown report path;
- HTML report path; and
- Colab file-download links for both reports when `google.colab.files` is
  available.

The user can inspect epoch progress in the same cell because the subprocess
does not capture or suppress training output.

## Testing

Unit tests will run the orchestration logic with subprocesses, Colab upload,
CUDA detection, and filesystem side effects replaced by fakes. Tests will
cover:

- repository-root validation;
- credential installation and permissions;
- reuse of valid credentials;
- reuse of a valid existing dataset;
- the expected command sequence for a fresh run;
- immediate failure when CUDA is unavailable;
- actionable Kaggle access errors; and
- final artifact discovery.

The existing full test suite, Ruff, mypy, and `git diff --check` will be run
after implementation. A real Kaggle download and full GPU training are not
part of automated local verification.

