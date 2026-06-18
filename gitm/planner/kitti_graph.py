"""Predicted execution graph for PointPillars on KITTI.

Models the 4-stage PointPillars pipeline as roofline nodes on the GPU,
then annotates CPU-bound stages (load, voxelization, NMS) that the roofline
does not cover. The gap between total predicted GPU time and actual measured
frame time is the CPU overhead the stream-concurrency invariant targets.

Usage::

    from gitm.planner.kitti_graph import predict_kitti_graph, render_kitti_graph
    from gitm.planner.roofline import HardwareSpec

    hw = HardwareSpec()  # defaults to A100-SXM4-80GB
    graph = predict_kitti_graph(hw=hw)
    print(render_kitti_graph(graph))

    # Compare against a measured baseline JSON:
    import json
    baseline = json.loads(Path("kitti_baseline_1.json").read_text())
    print(render_kitti_graph(graph, measured=baseline))
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gitm.planner.roofline import HardwareSpec, RooflinePrediction, roofline

# ── PointPillars KITTI frame shape constants ──────────────────────────────────
# Based on OpenPCDet pointpillar.yaml (kitti_models) defaults.
# These are per-frame averages; real variance is captured in stage_spread.

_MEAN_POINTS_PER_FRAME = 15_000       # typical KITTI Velodyne points
_MAX_PILLARS          = 12_000        # DATA_PROCESSOR max_number_of_voxels (train)
_MAX_POINTS_PER_PILLAR = 100          # max_points_per_voxel
_PILLAR_FEATURES       = 10           # NUM_POINT_FEATURES after augmentation
_VFE_OUT_CHANNELS      = 64           # PillarVFE output channels
_BEV_H, _BEV_W         = 248, 216     # BEV feature map size (after stride-2 conv)
_BACKBONE_CHANNELS     = 64           # initial 2D backbone channels
_HEAD_ANCHORS          = 70400        # SSD-style anchor grid (KITTI default)
_NUM_CLASSES           = 3            # Car, Pedestrian, Cyclist
_DTYPE_BYTES           = 4            # float32


@dataclass
class KittiNode:
    """One pipeline stage — GPU-side stages get a roofline prediction."""
    name: str
    device: str          # "gpu" | "cpu" | "pcie"
    prediction: RooflinePrediction | None = None
    note: str = ""
    expected_stream_id: int = 0  # -1 for CPU, 0+ for CUDA streams


@dataclass
class KittiGraph:
    hw: HardwareSpec
    nodes: list[KittiNode] = field(default_factory=list)

    @property
    def total_gpu_pred_ms(self) -> float:
        return sum(
            n.prediction.t_pred_s * 1000
            for n in self.nodes
            if n.prediction is not None
        )

    @property
    def total_pcie_pred_ms(self) -> float:
        return sum(
            n.prediction.t_pred_s * 1000
            for n in self.nodes
            if n.device == "pcie" and n.prediction is not None
        )


def predict_kitti_graph(hw: HardwareSpec | None = None) -> KittiGraph:
    """Return a predicted execution graph for one PointPillars KITTI frame.

    GPU stages use the roofline model (max of compute-bound and memory-bound).
    CPU stages are annotated but not predicted — they depend on the host CPU
    and are measured directly by WorkUnit.

    The stream-concurrency invariant predicts that voxelization (CPU, stream=-1)
    for frame N+1 should overlap backbone inference (GPU, stream=0) for frame N.
    The graph makes this expectation explicit via expected_stream_id.
    """
    hw = hw or HardwareSpec()
    g = KittiGraph(hw=hw)

    # Stage 1: Load .bin from disk (CPU / I/O — not modeled by roofline)
    g.nodes.append(KittiNode(
        name="load_bin",
        device="cpu",
        note="np.fromfile: ~15k points x 4 float32 = ~240 KB I/O. Not roofline-modeled.",
        expected_stream_id=-1,
    ))

    # Stage 2: Voxelization (CPU — scatter 15k points into 12k pillars)
    # Memory: read points (15k x 10 x 4B = 600KB) + write pillar buffer
    # (12k pillars x 100 pts x 10 feat x 4B = ~48 MB). Memory-bound on CPU.
    g.nodes.append(KittiNode(
        name="voxelization",
        device="cpu",
        note="Host-side scatter into voxel grid. CPU memory-bound. ~48 MB writes.",
        expected_stream_id=-1,  # CPU thread — overlaps GPU stream 0 (stream-concurrency invariant)
    ))

    # Stage 3: H2D copy (PCIe — pillar features host -> device)
    # Bytes: max_pillars x max_pts_per_pillar x features x float32
    h2d_bytes = _MAX_PILLARS * _MAX_POINTS_PER_PILLAR * _PILLAR_FEATURES * _DTYPE_BYTES
    # PCIe Gen4 x16 peak: ~32 GB/s; use conservative 20 GB/s effective
    pcie_bw = 20e9
    g.nodes.append(KittiNode(
        name="h2d_copy",
        device="pcie",
        prediction=RooflinePrediction(
            op="h2d_copy",
            flops=0,
            bytes=h2d_bytes,
            t_compute_s=0.0,
            t_memory_s=h2d_bytes / pcie_bw,
            t_pred_s=h2d_bytes / pcie_bw,
            bound="memory",
        ),
        note=f"pillar features: {h2d_bytes / 1e6:.1f} MB @ ~20 GB/s PCIe",
        expected_stream_id=0,
    ))

    # Stage 4: Pillar Feature Encoder / VFE (GPU)
    # Linear: (max_pts_per_pillar x pillar_features) -> vfe_out per pillar
    # FLOPs: 2 x active_pillars x max_pts_per_pillar x pillar_features x vfe_out
    # (~12k active pillars in practice; use max_pillars as upper bound)
    vfe_flops = 2 * _MAX_PILLARS * _MAX_POINTS_PER_PILLAR * _PILLAR_FEATURES * _VFE_OUT_CHANNELS
    vfe_bytes = _DTYPE_BYTES * (
        _MAX_PILLARS * _MAX_POINTS_PER_PILLAR * _PILLAR_FEATURES   # input
        + _PILLAR_FEATURES * _VFE_OUT_CHANNELS                      # weight
        + _MAX_PILLARS * _VFE_OUT_CHANNELS                          # output
    )
    g.nodes.append(KittiNode(
        name="pillar_vfe",
        device="gpu",
        prediction=roofline("pillar_vfe", vfe_flops, vfe_bytes, hw, dtype="fp32"),
        note="PointPillar feature encoder: linear + BN + ReLU per pillar",
        expected_stream_id=0,
    ))

    # Stage 5: BEV scatter (GPU — pillar features -> spatial BEV grid)
    # Scatter write: max_pillars x vfe_out -> bev_h x bev_w x vfe_out
    # Essentially a gather/scatter; compute is trivial, memory dominates.
    bev_scatter_bytes = _DTYPE_BYTES * (
        _MAX_PILLARS * _VFE_OUT_CHANNELS                   # read pillar features
        + _BEV_H * _BEV_W * 2 * _VFE_OUT_CHANNELS         # write BEV (2 strides)
    )
    g.nodes.append(KittiNode(
        name="bev_scatter",
        device="gpu",
        prediction=roofline("bev_scatter", 0, bev_scatter_bytes, hw, dtype="fp32"),
        note="Scatter pillar features -> 2D BEV pseudo-image",
        expected_stream_id=0,
    ))

    # Stage 6: 2D Backbone CNN (GPU)
    # Simplified estimate for PointPillars backbone (3 conv blocks, ~6 GFLOPs total)
    # BEV input: (BEV_H x BEV_W x VFE_OUT) processed through strided convolutions.
    # FLOPs dominated by early layers (large spatial, many channels).
    # Rough estimate: 6 GFLOP for KITTI resolution (published benchmarks).
    backbone_flops = 6e9
    # Activations: ~4 layers with growing channels, total ~300 MB
    backbone_bytes = 300e6
    g.nodes.append(KittiNode(
        name="backbone_2d",
        device="gpu",
        prediction=roofline("backbone_2d", backbone_flops, backbone_bytes, hw, dtype="fp32"),
        note="Strided 2D CNN: 3 blocks, progressively downsampled BEV features",
        expected_stream_id=0,
    ))

    # Stage 7: Detection head (GPU — anchor-based classification + regression)
    # FLOPs: 2 x head_anchors x (channels_in x num_classes + channels_in x 7_box)
    head_flops = 2 * _HEAD_ANCHORS * _BACKBONE_CHANNELS * (_NUM_CLASSES + 7)
    head_bytes = _DTYPE_BYTES * (
        _HEAD_ANCHORS * _BACKBONE_CHANNELS   # input activations
        + _BACKBONE_CHANNELS * (_NUM_CLASSES + 7) * 2  # cls + reg weights (conv1x1)
        + _HEAD_ANCHORS * (_NUM_CLASSES + 7)  # output
    )
    g.nodes.append(KittiNode(
        name="detection_head",
        device="gpu",
        prediction=roofline("detection_head", head_flops, head_bytes, hw, dtype="fp32"),
        note="SSD-style anchor cls + box regression, 1x1 conv",
        expected_stream_id=0,
    ))

    # Stage 8: NMS (CPU — CPU-accelerated iou3d_nms_cuda is a separate CUDA kernel
    # but serialized against the backbone stream via CPU synchronization)
    g.nodes.append(KittiNode(
        name="nms",
        device="cpu",
        note="iou3d_nms_cuda: box decode + IoU matrix + suppression. CPU-sync stall.",
        expected_stream_id=-1,
    ))

    return g


def render_kitti_graph(
    graph: KittiGraph,
    measured: dict | None = None,
) -> str:
    """Human-readable predicted execution graph, optionally compared to measured.

    Args:
        graph:    From predict_kitti_graph().
        measured: Optional baseline JSON dict (from run_baseline()) for comparison.
    """
    hw = graph.hw
    lines = [
        f"PointPillars predicted execution graph ({hw.name})",
        f"  peak FLOPS (fp32): {hw.peak_flops_fp32_per_s/1e12:.1f} TFLOP/s",
        f"  peak HBM BW:       {hw.peak_mem_bw_bytes_per_s/1e9:.0f} GB/s",
        "",
    ]

    hdr = f"{'stage':<20} {'device':<6} {'pred_ms':>9} {'bound':<8} note"
    lines += [hdr, "-" * 80]

    for node in graph.nodes:
        pred_ms = f"{node.prediction.t_pred_s * 1000:.3f}" if node.prediction else "N/A (CPU)"
        bound = node.prediction.bound if node.prediction else "--"
        lines.append(
            f"{node.name:<20} {node.device:<6} {pred_ms:>9} {bound:<8} {node.note[:50]}"
        )

    lines += [
        "-" * 80,
        f"{'total GPU pred':>26}: {graph.total_gpu_pred_ms:.3f} ms",
        f"{'PCIe H2D pred':>26}: {graph.total_pcie_pred_ms:.3f} ms",
    ]

    if measured:
        fps      = measured.get("frames_per_second", 0)
        frame_ms = 1000 / fps if fps > 0 else 0
        gpu_pct  = measured.get("gpu_active_pct", 0)
        data_pct = measured.get("data_stall_pct", 0)
        lines += [
            "",
            "Measured (baseline):",
            f"  frame time:   {frame_ms:.1f} ms  ({fps:.1f} fps)",
            f"  GPU active:   {frame_ms * gpu_pct / 100:.1f} ms  ({gpu_pct:.1f}%)",
            f"  data stall:   {frame_ms * data_pct / 100:.1f} ms  ({data_pct:.1f}%)",
            "",
            f"  Roofline predicts {graph.total_gpu_pred_ms:.1f} ms GPU time.",
            f"  Actual GPU active is {frame_ms * gpu_pct / 100:.1f} ms "
            f"({'FASTER' if frame_ms * gpu_pct / 100 < graph.total_gpu_pred_ms else 'SLOWER'} than roofline).",
            f"  Remaining {frame_ms * (100 - gpu_pct) / 100:.1f} ms ({100 - gpu_pct:.1f}%) is",
            "  CPU overhead (voxelization + NMS) -- the stream-concurrency invariant target.",
        ]

    return "\n".join(lines)
