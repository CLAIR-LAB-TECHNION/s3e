# Security policy

## Supported versions

Security fixes are made in the latest release of `s3e` on
[PyPI](https://pypi.org/project/s3e/). Older versions are not patched.

## Reporting a vulnerability

Please do not report security problems in public issues. Instead, email the
maintainer at guy.azran@campus.technion.ac.il, or use "Report a
vulnerability" under the repository's
[Security tab](https://github.com/CLAIR-LAB-TECHNION/s3e/security) if it is
offered there. Include the affected version, a description of the problem,
and steps to reproduce it.

You will get a reply once the report has been assessed; fixes are released as
a new version and noted in the [changelog](CHANGELOG.md).

## Things to keep in mind

- `s3e` loads models through Hugging Face Transformers and vLLM. Keyword
  arguments such as `trust_remote_code=True` are passed through unchanged and
  run code from the model repository; only enable them for models you trust.
- Files written by `CalibrationSet.save`, `PlattCalibrator.save`, and the
  `LLMTranslator` cache are plain JSON, read back with the `json` module; they
  hold no pickled objects.
