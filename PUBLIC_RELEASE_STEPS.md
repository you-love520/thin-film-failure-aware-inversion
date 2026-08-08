# Public release steps

The archive is scientifically frozen and prepared as a release candidate. The following steps require the repository owner to complete actions that are not available through the current connected GitHub tool.

1. Create a new public GitHub repository, suggested name: `thin-film-failure-aware-inversion`.
2. Upload the **contents** of this release-candidate folder to the repository root.
3. Select licensing before publication. A practical dual-license option is:
   - source code: MIT License;
   - data tables, figures, and documentation authored for this study: CC BY 4.0.
   The authors should approve the final license choice before files are made public.
4. Run:
   ```bash
   python analysis/verify_release.py
   sha256sum -c provenance/SHA256SUMS.txt
   ```
5. Create a GitHub release (for example `v1.0.0`).
6. Link the GitHub repository to Zenodo and archive the release.
7. Insert the resulting stable repository URL and Zenodo DOI into the manuscript Data Availability Statement and `CITATION.cff`.
8. Re-run the manifest/checksum verification after any public-release metadata change.

Do not upload reviewer reports, internal red-team notes, personal credentials, or unrelated project files.
