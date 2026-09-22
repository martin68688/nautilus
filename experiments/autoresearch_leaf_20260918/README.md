# Leaf autoresearch migration — implementation, NOT launched

## Superseding protocol (2026-09-18)

Old A/B and independent AWS OAuth are cancelled. R1 is a historical zero-memory
run restricted to completed evidence at launch+7200 seconds; R2 uses only its
newly distilled memory, R3 uses only R1+R2. New rounds use Terra-low through the
coordinator-authorized apizh-ai.com relay, 7200 seconds each. The historical
91-card package and engineering smoke outputs are excluded. Source audit and
new synthesis are pending. Older builds are archival and MUST NOT be launched.
Build now requires an explicit memory path/hash and rejects the old 91-card hash.
The below A/B description is superseded historical implementation context.

Upstream: ruz048/autoresearch commit 1e9849798006a069d17399f0d4ad0178200ee946.
Adapted main.py's separation: one proposing Agent, Python executes the candidate,
Host decides keep/discard. No MLEvolve searcher is imported. The original Leaf
checkpoint ABI is vendored independently, with completed checkpoints retained.

## Frozen intent

- Leaf, not Hotel. Coordinator is resolving user's final task choice.
- A: no historical memory; B: fixed audited historical pack; both own-run history.
- Luna-low; 7200 seconds each; candidate inclusive ceiling 3600 seconds.
- Latest authorization changes NRP A10 to AWS A10G. Same resource cap for A/B:
  4 CPU, 14Gi RAM, 2Gi shm, one GPU. Architecture freedom retained; CPU models valid.
- No AWS instance creation by this task. Coordinator allocates exact instance.
- Original fit/tune/search/test and class order unchanged, no fitted source weights.

## Implemented, offline only

- `core.py`: rejects nonzero exits/nonfinite metrics; null failure scores;
  immutable candidate attempts; Host evidence hashes; recovered incumbent and
  absolute original deadline; subprocess process-group cleanup.
- `agent.py`: same Agent queries literal terms, reads full original cards, records
  adoption/adaptation/rejection/defer reasons and exact code excerpts. No file tools.
- `provider.py`: trusted-host CPA Chat Completions JSON adapter,
  Luna/low request and usage receipts; private URL asserted, no relay fallback.
- `sandbox.py`: planned separate Docker candidate with no network, credentials,
  Host-label mount, other experiments, Docker socket or privileges. Resource capped.
- `host.py`: old log-loss clipping/renormalization semantics, immutable artifact
  hashes, IDs/classes/simplex/submission validation, privileged Host never imports
  untrusted candidate/checkpoint Python. `replay.py` runs in candidate sandbox.
- `main.py`: integration draft; formal launch fails closed without live gates and
  runtime manifest. It is NOT currently a launch-approved runtime.

## Open gates / work remaining (do not bypass)

1. AWS instance allocated and data verified; independent CPA device authorization
   and real model probe still pending coordinator. Credentials stay on root volume.
2. The 91-card cumulative pack is approved for byte-preserved reuse with separate
   compatibility annotations. Original-source re-execution is NOT claimed. Its source
   metadata includes R1/R2/R3, R4 Luna 0.04, Terra variants, cancelled/copied resume,
   startup/auth failures and subsequent CPA branches. Preserve those distinctions;
   transport tests are not validation of lessons. No new distillation performed.
3. Live Docker negative probes: hidden labels, other-run outputs, credentials,
   process namespaces, writes outside artifacts, network, kill descendant cleanup.
4. Actual requested-model generation and full-card decision probe NOT run.
5. Minimal real train/save/fresh-reload/Host score NOT run.
6. `supervise.py` now wraps the whole arm with the original absolute deadline and
   reaps exact run-labelled Docker containers. Exclusive controller/supervisor locks
   and runtime member hashing are implemented. Live Docker kill/restart tests remain
   required; current local subprocess tests do not prove the Docker daemon cleanup.
7. Audit every adopted parameter against executed events/checkpoint metadata;
   current decisions are explicitly `execution_verified=false`.
8. Freeze shared dependency image and equally available generic pretrained cache;
   no silent narrowing to numerical-only architectures.
9. Before launch establish coordinator-owned persistent monitor, result return
   EBS -> Mac -> fresh PVC backup and exclusivity-checked EC2 stop gates.

Run offline tests:
`python3 -m unittest discover -s experiments/autoresearch_leaf_20260918 -p 'test_*.py' -v`

No generated code or training outcome, no new Host score, no memory benefit claim.
API schema reference: https://developers.openai.com/api/reference/resources/chat
