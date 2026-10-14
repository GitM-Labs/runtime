# Reconciliation with main on October 13

The branch was rebased from `16d6d44c984d5446b6fc124550c995649cf96106` onto current `main` at `12dd8b4002125447777dcb2ea20f9cd5fa772032`, nine upstream commits later. All five local commits replayed without conflicts. Git range-diff confirmed the changes in each commit were retained. The pre-rebase history is preserved in the local backup branch `ishaan/cross-architecture-pre-reconcile-20261013`.

Upstream changed registry reporting, GPU discovery and planner tests among other runtime areas. The integrated result retains those changes, including the AMD kfd discovery path and GPU warning behavior. The existing assumption that an unknown spatial device must not inherit GPU defaults remains applicable; the local named-resource API does not infer such a mapping. Existing catalogue model geometry and roofline equations are unchanged by the rebase.

## Verification on the refreshed branch

- Full repository suite on local macOS with bundled Python 3.12.14 and project dev/bench dependencies: **2400 passed, 6 skipped, 0 failed** in approximately 81 seconds. This is a local CPU run, not GitHub CI or a new GPU run.
- Fourteen legacy catalogue/Qwen output comparisons still match the original hashes after excluding the additive result-semantics metadata.
- Repository-pinned Ruff **0.12.11** passes across the tree. The only new lint failures were in the supplied Arshia source copies. A narrow `extend-exclude` entry now excludes `docs/cross-architecture/sources/arshia`; its historical files remain byte-for-byte unchanged. Maintained planner code, scripts and tests are still linted.
- Authored diff whitespace checks pass. Retained calculator stdout whitespace and CSV CRLF endings remain unchanged evidence exceptions.
- Importer fixtures generated or modified by the tests were restored to their exact pre-run state after completion.

Five skipped cases require unavailable CUDA/ROCm tooling, headers or device support. The remaining skip is the existing curated-workload lever-reachability case. Skips are recorded as skips, not passes. The GitHub Linux/Python 3.10–3.12 matrix has not run for this branch.

The older 512-case selected-suite result remains historical verification of the prepared changes. It is supplemented by this broader run; its count must not be added to 2400 as independent coverage. Original GPU timing and numerical failures remain unchanged.

## Related DeepSeek work

A merge-tree preview with Medha's PR #176 at `fd22c54d9f8394eaee0c5a6c180a1d1a34b432a3` completed without textual conflicts. The preview wrote Git objects only; it did not merge that PR into this branch or change the working tree.

Her stored-weight/MTP/V3.2 handling and our independent index-cache fields address distinct concerns in `glm_graph.py`. A clean textual merge does not establish semantic compatibility. No combined-branch test run, acceptance of her changes, dependency order or reviewer sign-off is claimed. Recheck the then-current PR before merge. CF-07 remains open on this branch; the related unmerged implementation is relevant evidence for its next review.

## Review and publication

The PR description has been refreshed with the new base, full local suite, pinned lint result and preview boundary. Agent-written changes and automated verification remain distinguished from independent human verification. Local commit authorization does not establish maintainer approval, a shared live register or team adoption. No push, PR, reviewer request, external message or additional GPU spend occurred.
