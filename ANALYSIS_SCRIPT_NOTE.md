# Analysis-script portability note

`analysis/verify_release.py` is the portable, standard-library verification entry point for this release candidate.

The other analysis scripts are archived provenance copies of the exact scripts used during manuscript preparation. Some retain the original `/mnt/data` workspace paths and are included to preserve the generation history rather than as turn-key public CLIs. Their input/output path assumptions should not be interpreted as scientific dependencies. The released source tables and `verify_release.py` are sufficient to verify the frozen counts and principal numerical claims without rerunning the estimators.
