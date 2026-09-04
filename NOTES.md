# Build notes: the local-LLM step

validation.md flagged one real risk up front: "the small local instruction
model step ... will be slow at any real scale" and told this build to scope
it down to a fixed sample rather than discover the problem mid-build. The
actual problem hit was more fundamental than slowness.

## What was tried

Four independent loading strategies were tried, for two different models,
in this exact build environment (Windows, sandboxed shell):

1. `AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")`
   (standard HF download+load path) -> crashed with a Rust allocator abort
   (`memory allocation of 67078537 bytes failed`) during download.
2. Direct `curl` download of the safetensors file (bypassing the HF Rust
   downloader entirely) + `from_pretrained(local_dir, dtype=torch.bfloat16,
   low_cpu_mem_usage=True)` -> download succeeded (988MB file, verified
   complete), but loading failed with `OSError: The paging file is too
   small for this operation to complete. (os error 1455)` inside
   `safetensors`' `safe_open`.
3. Repeated the same with a much smaller model
   (`HuggingFaceTB/SmolLM2-135M-Instruct`, 269MB safetensors) -> same
   `safe_open` paging-file error. Confirmed independently that a raw
   Python `mmap.mmap()` of the same 988MB file succeeds fine, and that
   `safetensors.torch.load_file()` (a different code path that doesn't do
   the same private/COW mapping) loads the 269MB file in 0.27s with no
   error -- so the failure is specific to how `transformers`' internal
   `_load_pretrained_model` opens the file, not a hard cap on file size or
   memory as such.
4. Manual load bypassing that code path: build the model on `meta` device,
   load real weights via `safetensors.torch.load_file` +
   `model.load_state_dict(sd, assign=True)`, and separately reconstruct
   the (non-persistent, so not in the state dict) `LlamaRotaryEmbedding`
   buffer. This got furthest -- reached "model ready" in ~32s once -- but
   was **not reproducible**: identical re-runs of the identical script
   segfaulted (exit 139) with zero output before even printing "model
   ready," twice in a row.

For comparison: `sentence-transformers/all-MiniLM-L6-v2` (a ~90MB
transformer encoder, no `generate()` / KV-cache involved) loaded and ran a
forward pass with no issue on the first try, every time. A genuinely tiny
test-stub causal LM (`sshleifer/tiny-gpt2`, a few MB, random weights meant
only for CI plumbing) also loaded and called `.generate()` successfully --
confirming the `generate()` code path itself isn't broken, just that
nothing large enough to produce coherent language survives loading
reliably in this environment.

## Conclusion

This looks like a low, environment-specific memory/commit-charge ceiling
that causal-LM checkpoint loading in `transformers` hits (Windows reports
it as a "paging file too small" error; Rust panics elsewhere in the stack
suggest allocator failures right at that same ceiling), independent of
which specific model or loading strategy is used, and independent of true
available system RAM (5.8GB free was reported by the OS at the time).
Modifying system-level Windows pagefile settings to work around this is
outside the scope of what this build should do unilaterally (requires
admin rights and a reboot, and is a persistent environment change, not a
code change) -- so it was not attempted.

## What this means for the build

`src/generate.py::llm_narrative()` is fully implemented and is the
function a reader on a normal machine (or with more headroom than this
sandbox) would call with `use_llm=True`. It was **not** exercised
end-to-end in this environment, and the build does not claim it was.

The shipped demo instead runs `template_narrative()` -- a deterministic,
always-grounded generator built directly from the same SHAP-selected
ReasonCode objects. This is a real scope cut, made explicit here rather
than hidden: the "generation" step in the shipped demo does not involve
an LLM. What is real, and what the EVAL GATE actually measures
(EVAL-FINDINGS.md), is the grounding gate itself -- the mechanism
validation.md identified as the actually-novel part of this idea,
independent of which generator produces the raw text.
