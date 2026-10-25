Packed FP16 convolution tuning
==============================

The GPU compiler offers native packed convolution candidates alongside rocMLIR
when the rocMLIR convolution path is selected, for a deliberately narrow,
validated family of batch-one FP16 NCHW operations.
Recognition uses graph semantics and tensor shapes, independently of model names.
For this path, set ``MIGRAPHX_MLIR_USE_SPECIFIC_OPS=convolution,fused_convolution``
before library initialization, or use the application's existing rocMLIR backend
selection. Other backend policies remain governed by the normal compiler.

Supported operations
--------------------

* A 3x3 convolution, padding one, stride/dilation one, group one, channel bias,
  and SiLU, with input ``[1,512,10,10]`` or ``[1,32,10,10]`` and 32 output channels.
* The 32-channel variant followed by a biased 1x1 convolution with four output
  channels. An output reshape preserving element order is allowed.

Inputs and weights must be standard contiguous FP16 tensors. Bias broadcasting
must address the channel dimension. Dynamic shapes, additional activation
operations, and externally consumed intermediates prevent two-stage fusion.
The initial hardware gate accepts gfx900/gfx90c with wave64 and sufficient
workgroup capacity. Broader hardware and shape support requires validation.

Native candidates require the existing ``compile_options.fast_math`` permission.
They keep convolution, bias and SiLU in FP32; this can avoid overflow and change
rounding relative to explicitly materialized FP16 operations. With ``fast_math``
false, the native candidates and two-convolution region fusion are excluded.
This permission does not enable approximate exponentials: the kernel uses ordinary
``expf`` and division. The two-convolution epilogue rounds SiLU to nearest-even FP16
before FP32 pointwise accumulation. Reduction order can also change outputs.
These candidates therefore require model accuracy validation, rather than a claim
of bit-exact preservation of the original graph.

Split-K partitions work among waves within one workgroup, without atomics or a
global partial-result workspace. Each wave owns disjoint operand tiles. Converged
wave barriers, memory compiler fences and explicit LDS waits order those accesses.
Two full-workgroup barriers separate operand reads, shared-arena reuse and partial
reduction. Wave zero retains its partial in registers; the other waves store only
their own partial planes. Split one omits this cross-wave phase. Only wave zero
produces and consumes the transposed FP16 activation tile, using packed dword
channel pairs and MAD_MIX for the pointwise tail. Necessary image-edge and spatial
tail guards remain; exact launch coverage eliminates channel-tail checks.

Compiler integration
--------------------

``analyze_packed_conv`` is shared by graph matching and JIT compilation.
``gpu::fuse_mlir`` merges only eligible single-consumer secondary heads, retaining
the original operations in the submodule. Dead nodes are eliminated before
checking consumer counts.

``get_packed_conv_tuning_config`` adds native candidates to the existing tuning
framework. For a two-stage region it also offers the ordinary rocMLIR pair and
a packed first stage followed by the ordinary 1x1 implementation. The retained
submodule supplies the unfused fallback. Both ordinary stages are tuned
independently through their existing compiler before benchmarking complete region
alternatives;
there is no Cartesian product. GPU-backed stage selection runs serially after
parallel configuration discovery. Each stage's normal compiler replacement applies
its own allocations, nested splits and prefill metadata at benchmark and runtime.
No new runtime operator is needed.

Quick tuning starts with the measured 32x8/K16/split8 and 16x16/K8/split8
large-head configurations, plus 32x8/K16/split6, 32x8/K36/split8 and
32x8/K8/split4 for the small head.
``exhaustive_tune`` searches M tiles 16/32 (32 for the fused head), N tiles 8/16/32,
K tiles 8/16/18/32/36 and splits 1/2/3/4/6/8. Shared-memory accounting uses the maximum
of operand and reduction storage, plus the optional activation tile. The 64 KiB
limit and exact K divisibility filter candidates before compilation. Guarded K
padding, narrower N4 mapping, extra prefetch buffers, cooperative epilogues and
alternate B padding were measured but did not justify permanent kernel variants.
Launch dimensions are derived from the configuration; every output tile receives
a workgroup.

The problem cache retains structured provider/configuration records. Problem keys
use canonical JSON strings so serialization preserves hash identity. Keys include
the implementation revision, compiler tweak, device architecture/CU count, input
shapes/strides, argument order, JIT optimization/flags, and search scope/limit.
A cached quick-search winner does not prevent an explicit exhaustive search.
Region keys include the selected ordinary-stage configurations; stage keys include
their legacy cache seeds. Updating a stage winner invalidates dependent region
selection. Numeric configuration fields are validated individually so JSON
serialization does not change their validity. Existing tracing
reports configurations, timing, cache hits, and compilation failures.
``MIGRAPHX_TRACE_MLIR`` also reports eligibility rejection reasons.

Build and validation
--------------------

Use the project's normal build prerequisites, including a rocMLIR version matching
``requirements.txt``. Build into a separate prefix before replacing an installation.
The kernel header is embedded by the existing kernel-file CMake machinery; the GPU
library adds ``packed_conv.cpp``. The normal test discovery adds
``test_gpu_packed_conv`` automatically.

Run the packed-convolution tests and existing GPU fusion/JIT tests. The focused
suite checks semantic matching, external intermediate consumers, unsupported
graphs, rejected launch configurations, independently computed double-precision
convolution results, nearest-even FP16 intermediate rounding, overflow/cancellation
and subnormal/near-zero cases, math permission, prefill insertion for flat and tuple
outputs, stage-cache dependencies, JSON configuration round trips, repeated
execution, and output canaries across the admissible configurations.
Unsupported hardware skips execution-specific cases explicitly.

After passing focused tests, compare whole-model latency using interleaved runs,
then evaluate accuracy separately. Use identical model files for each comparison;
separate FP32-derived and pre-exported FP16 files can contain different weights.
An isolated-kernel winner can regress whole-model latency through allocation and
code-placement effects, so deployment profiles must retain whole-model evidence.

Compiled code objects are embedded in saved MXRs. Loading them requires neither
kernel source files nor an injector. Existing host-copy and resident/graph runtimes
continue to execute them through the normal code-object path. Publish a distinct
compiler build identity, or clear its compiled-model cache, when replacing a build;
changing tuning records cannot rewrite an existing MXR.
