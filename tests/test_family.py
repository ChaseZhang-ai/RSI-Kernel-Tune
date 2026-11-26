"""rsi_kernel_tune.family: the operator family of a board problem, read from families.toml."""

from __future__ import annotations

import io
import json
import re
from collections import Counter
from pathlib import Path

import pytest

from rsi_kernel_tune import family
from rsi_kernel_tune.family import (OTHER, FamilyError, count_by_family, family_key, families,
                                is_backward, is_family, load_rules, main, parse_rules, stem)

# The 235 problems of the board snapshot of 2026-11-24, by the family the rule table gives
# each. A rule change that moves a problem shows up here as a failed name: that is the point
# of the table. (The board numbers its problems per track, so a bare number repeats; the full
# names do not.)
BOARD = {
    "rmsnorm": [
        "013_fused_residual_rms_norm_backward",
        "038_flux_multi_head_rmsnorm_qk",
        "057_mtp_shifted_embedding_with_dual_rms_norm_fusion",
        "069_rms_norm",
        "001_fused_add_rmsnorm_h2048",
        "002_fused_add_rmsnorm_h4096",
        "003_fused_add_rmsnorm_h7168",
        "021_rmsnorm_h128",
        "022_rmsnorm_h512",
        "023_rmsnorm_h1536",
        "024_rmsnorm_h2048",
        "025_rmsnorm_h4096",
        "026_rmsnorm_h7168",
    ],
    "transformer_block": [
        "002_decoder_layer_full_block",
        "004_fused_residual_rms_mlp",
        "009_decoder_layer_with_residual_connections",
        "019_decoder_layer_fused_attention_mlp",
        "020_decoder_layer_pre_post_norm_residual",
        "053_text_decoder_layer_with_self_attention_and_mlp",
        "054_vision_encoder_layer_with_gated_residuals",
        "056_language_model_decoder_prenorm_attention_ffn_residual_backward",
        "059_decoder_layer_full_block",
        "062_decoder_complete_layer",
        "063_encoder_layer_dual_residual_norm_chain",
        "069_joint_transformer_block_residual_path",
        "070_basic_transformer_block",
        "079_git_vision_encoder_layer",
        "030_nvfp4_fused_decoder_layer_pre_norm_attention",
    ],
    "vision_block": [
        "086_sam_hq_mask_decoder_iou_hypernetwork_fusion",
        "089_vae_attention_block_with_groupnorm",
        "022_video_latent_denoising_unet_block",
        "023_video_latent_vae_encoder_downsampling",
        "033_multi_scale_feature_pyramid",
        "035_convnextv2_block_with_grn",
        "036_convnextv2_layer_with_nhwc_persistence_backward",
        "050_vae_decoder_mid_block_attention_resnet",
        "066_resnet_block_with_time_embedding",
        "074_sam_hq_mask_decoder_iou_hypernetwork_fusion_backward",
        "075_sam_hq_mask_decoder_two_way_transformer",
        "078_fused_final_layer_upsample_with_adaptive_norm",
    ],
    "quant_gemm": [
        "001_fp8_attention_output_projection",
        "002_fp8_attention_qkv_projection",
        "003_fp8_mlp_gate_up_projection",
        "004_fp8_moe_expert_linear",
        "005_fp8_moe_router_projection",
        "006_fp8_vision_attention_output_projection",
        "007_fp8_vision_attention_qkv_projection",
        "008_nvfp4_multimodal_embedding_projection",
        "009_nvfp4_vision_temporal_patch_merge_with_projection",
        "010_fp8_attention_output_projection",
        "012_fp8_shared_expert_mlp",
        "013_fp8_mla_kv_compression_projection",
        "015_fp8_mla_attention_output_projection",
        "016_fp8_multi_latent_attention_qkv_projection",
        "017_fp8_shared_expert_mlp",
        "018_nvfp4_attention_output_projection_with_residual",
        "020_nvfp4_linear_layer",
        "021_fp8_mamba2_in_projection",
        "022_fp8_mamba2_out_projection",
        "024_nvfp4_attention_output_projection",
        "025_nvfp4_attention_qkv_projection",
        "026_nvfp4_mamba2_out_projection",
        "027_nvfp4_moe_router_projection",
        "028_nvfp4_attention_qkv_projection_with_rope",
        "029_nvfp4_fused_mlp_silu_gating",
        "032_nvfp4_moe_expert_linear_with_gating",
    ],
    "moe": [
        "008_expert_output_weighted_index_add_accumulation",
        "009_expert_token_scatter_with_weighted_forward_backward",
        "017_moe_expert_swiglu_with_down_projection_backward",
        "042_moe_expert_load_balancing_and_token_capacity_backward",
        "044_moe_expert_computation",
        "058_moe_expert_token_radix_sort_with_prefix_sum",
        "059_moe_group_score_aggregation_and_masking",
        "076_batched_expert_forward",
        "093_grouped_topk_moe_routing_backward",
        "008_moe_sparse_routing_and_dispatch",
        "010_moe_expert_computation_with_weighted_accumulation",
        "011_moe_sparse_routing_and_dispatch_backward",
        "012_moe_expert_batched_execution_with_capacity_factor",
        "013_expert_weighted_aggregation_with_shared_expert",
        "016_moe_expert_mlp_with_load_balancing",
        "024_moe_expert_parallel_execution",
        "025_moe_expert_parallel_execution_backward",
        "026_moe_expert_parallel_execution_with_weighted_aggregation",
        "029_moe_sparse_routing_and_dispatch",
        "047_moe_training_token_repeat_and_expert_computation",
        "048_moe_expert_inference_batched_dispatch",
        "049_grouped_limited_topk_routing",
        "061_sparse_moe_routing_and_experts",
        "065_sparse_expert_dispatch_and_combine",
        "080_moe_complete_layer_with_shared_expert_backward",
        "081_moe_sparse_expert_dispatch",
        "082_moe_layer_complete_forward_with_residual",
        "011_fp8_moe_gate_routing",
        "033_nvfp4_moe_routing_with_topk_selection",
        "020_moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048",
    ],
    "ssm": [
        "006_hyena_depthwise_conv1d_split_gate",
        "007_hyena_fft_size_padding_rfft",
        "029_mamba_conv1d_with_gating",
        "056_mamba_ssm_dt_projection_softplus_clamp_backward",
        "070_mamba2_fused_intra_chunk_diagonal_computation",
        "079_ImageNet_83.6_ssm_output_projection_gate_multiply_backward",
        "043_mamba_chunk_scan_with_segsum",
        "044_mamba_discretization_and_segsum",
        "051_seqlen-finetuned-reconstructed_hyena_complete_forward_block",
        "052_seqlen-finetuned-reconstructed_hyena_gating_and_output_projection",
        "058_mamba2_selective_scan",
        "060_chunk_gated_delta_rule_linear_attention",
        "023_fp8_mamba2_ssm_discretization",
    ],
    "conv": [
        "002_vae_conv3x3_groupnorm_silu_residual_fused",
        "005_conv_gated_projection_with_causal_conv",
        "040_conv2d_residual_block",
        "041_strided_conv2d_downsampling",
        "015_audio_sinusoidal_position_embedding_with_conv_projection",
        "055_audio_encoder_conv_positional_layer_stack",
    ],
    "activation": [
        "025_video_latent_gelu_activation",
        "053_gaussian_topk_sparse_activation",
        "061_tanh_gated_residual_add_backward",
        "084_silu_activation_backward",
        "085_geglu_activation",
    ],
    "mlp": [
        "020_vision_patch_merger_spatial_shuffle_mlp",
        "022_vision_mlp_gelu_backward",
        "037_flux_feedforward_gelu_approximate",
        "045_fused_linear_gelu_grn_linear",
        "048_fused_gate_up_projection_with_swiglu",
        "074_fused_gated_mlp_silu",
        "005_swiglu_mlp_backward",
        "042_ffn_gelu_projection_fused_backward",
        "068_gelu_approximate_feedforward_backward",
        "073_feedforward_mlp_backward",
    ],
    "gemm": [
        "003_lm_head_projection_with_logit_slicing",
        "004_attention_output_projection_with_reshape_backward",
        "010_attention_value_projection_with_transpose",
        "026_video_patch_embedding_projection",
        "030_attention_output_projection_with_residual",
        "032_attention_weights_matmul_with_value_projection",
        "036_flux_output_norm_projection_chain",
        "043_mla_fused_qkv_rope_split",
        "050_attention_qkv_projection_with_bias_and_reshape",
        "051_attention_qkv_with_qk_norm_single_kernel_backward",
        "054_audio_attention_qkv_projection_with_normalization",
        "055_attention_qkv_projection_with_gqa_reshape",
        "063_attention_output_reshape_and_projection",
        "065_fused_statistics_projection_and_split",
        "072_cross_attention_qkv_projection_with_gqa_repeat_backward",
        "073_fused_encoder_final_norm_to_decoder_cross_attention_kv_projection",
        "077_whisper_decoder_output_projection",
        "081_joint_attention_context_projection",
        "091_fused_gqa_qkv_rope_qk_norm_backward",
        "031_flux_timestep_guidance_projection_embedding",
        "064_multi_head_qkv_projection_with_rope_backward",
        "004_gemm_n128_k2048",
        "005_gemm_n256_k7168",
        "006_gemm_n2048_k4096",
        "007_gemm_n4096_k4096",
        "008_gemm_n4096_k14336",
        "009_gemm_n5120_k2048",
        "010_gemm_n6144_k4096",
        "011_gemm_n28672_k4096",
    ],
    "attention": [
        "001_attention_softmax_dropout_value_matmul_backward",
        "015_grouped_query_attention_with_rope_and_qk_norm",
        "021_vision_cu_seqlens_variable_length_attention",
        "027_video_spatial_attention_with_rope_3d",
        "028_hybrid_attention_mask_preparation",
        "031_repeat_kv_attention_matmul",
        "046_attention_softmax_with_softcapping_and_dropout",
        "047_attention_with_qk_norm_and_rope",
        "049_attention_qk_matmul_with_gqa_repeat_and_scaling",
        "060_fused_attention_qk_matmul_scale_mask_softmax_backward",
        "062_kv_cache_update_with_rope_backward",
        "064_latent_kv_expansion_with_split",
        "066_masked_softmax_with_attention_dropout_backward",
        "067_flash_attention_gqa_ultralong",
        "071_kv_cache_update_with_rope",
        "075_grouped_query_self_attention_with_rope",
        "082_qk_norm_scaled_dot_product_attention",
        "083_attention_score_value_matmul",
        "092_gqa_attention_with_qk_norm",
        "001_fused_vision_multihead_attention_with_norms_backward",
        "003_grouped_query_attention_with_rope_backward",
        "007_multimodal_rotary_embedding_attention",
        "014_audio_encoder_varlen_attention_with_chunking_backward",
        "017_fused_vision_cu_seqlens_attention_with_2d_rope_backward",
        "018_cu_seqlens_variable_length_vision_attention",
        "021_cross_attention_text_video_conditioning_backward",
        "027_grouped_query_attention_with_yarn_rope_and_qk_norm",
        "028_gqa_rotary_attention_core_backward",
        "032_dual_stream_attention_with_conditional_cross_attention",
        "034_vision_language_cross_attention_fusion",
        "037_qk_norm_rope_attention_core_backward",
        "038_audio_relative_position_attention",
        "039_kv_shared_attention_with_rope",
        "041_kv_shared_attention_with_dual_rope",
        "046_conformer_relative_position_attention",
        "072_region_aware_self_attention_with_edit_bias_backward",
        "076_sam_hq_vision_attention_with_relative_position_backward",
        "077_sam_hq_vision_encoder_window_partition_attention",
        "019_nvfp4_grouped_query_attention",
        "031_nvfp4_grouped_query_attention_with_kv_repeat",
        "012_gqa_paged_decode_h32_kv4_d128_ps1",
        "013_gqa_paged_decode_h32_kv8_d128_ps1",
        "014_gqa_paged_prefill_causal_h32_kv4_d128_ps1",
        "015_gqa_paged_prefill_causal_h32_kv8_d128_ps1",
        "016_gqa_ragged_prefill_causal_h32_kv4_d128",
        "017_gqa_ragged_prefill_causal_h32_kv8_d128",
        "018_mla_paged_decode_h16_ckv512_kpe64_ps1",
        "019_mla_paged_prefill_causal_h16_ckv512_kpe64_ps1",
    ],
    "rope": [
        "011_rotary_position_embedding",
        "012_fused_cos_sin_embedding_generation",
        "014_rotary_embedding_with_attention_scaling_backward",
        "016_rope_inverse_frequency_computation",
        "018_fused_rope_with_qk_norm_and_kv_cache_update",
        "019_vision_3d_rotary_embedding_with_spatial_merge_indexing_backward",
        "023_multimodal_rope_position_computation_with_grid_based_indexing",
        "024_vision_rotary_position_embedding_generation_backward",
        "034_flux_multi_axis_rope_frequency_computation",
        "068_fused_rope_application_backward",
        "088_rotary_position_embedding_application",
        "090_batched_2d_rope_position_encoding_backward",
        "006_multimodal_rope_position_calculation",
        "014_fp8_yarn_rope_embedding",
    ],
    "norm": [
        "033_post_norm_residual",
        "035_flux_ada_layer_norm_zero_modulation_extraction",
        "039_adaptive_instance_normalization_2d_backward",
        "078_group_norm_fusion",
        "080_adaptive_layernorm_continuous_with_modulation",
        "087_embedding_with_initial_layernorm_backward",
    ],
    "other": [
        "052_altup_hidden_state_collapse_with_magnitude_normalization",
        "094_time_decay_exponential_stabilization",
        "030_flux_concatenated_sequence_processing_with_split",
        "040_altup_predict_correction_cycle_backward",
        "045_audio_encoder_to_language_model_multimodal_fusion",
        "057_residual_coupling_flow_block",
        "067_patch_embed_to_joint_attention_input",
        "071_edit_consistency_loss_with_perceptual_weighting",
    ],
}
NAMES = [n for names in BOARD.values() for n in names]
BACKWARD_ON_BOARD = 46


# --- the board -----------------------------------------------------------------------------


def test_the_board_snapshot_is_pinned_problem_by_problem():
    assert len(NAMES) == len(set(NAMES)) == 235
    for key, names in BOARD.items():
        assert names, key
        for name in names:
            assert family_key(name) == key, name


def test_every_family_is_declared_once_and_other_comes_last():
    keys = families()
    assert len(keys) == len(set(keys)) and keys[-1] == OTHER
    assert set(keys) == set(BOARD), "a family the rules give is not one the table declares"
    assert all(is_family(k) for k in keys)
    assert not is_family("rms_norm") and not is_family("GEMM")


def test_no_family_is_a_handful_and_the_sizes_are_pinned():
    """What the rule table is for: problems to share what was measured on others. A family of
    one or two has nobody to learn from, so a rule that empties one shows up here."""
    counts = count_by_family(NAMES)
    assert sum(counts.values()) == 235
    assert counts == {
        "rmsnorm": 13, "transformer_block": 15, "vision_block": 12, "quant_gemm": 26,
        "moe": 30, "ssm": 13, "conv": 6, "activation": 5, "mlp": 10, "gemm": 29,
        "attention": 48, "rope": 14, "norm": 6, "other": 8}
    assert max(counts.values()) / 235 < 0.25, "one family holds a quarter of the board"
    assert counts[OTHER] / 235 < 0.05, "a name in twenty falls through every rule"
    assert min(v for k, v in counts.items() if k != OTHER) >= 5, "a family too small to learn from"
    assert counts["attention"] >= 8 and counts["gemm"] >= 8      # the FlashInfer cores
    assert counts["rmsnorm"] == 13                               # 3 fused-add + 6 plain + 4 others


def test_a_backward_kernel_is_its_forwards_family_and_says_that_it_is_backward():
    assert not is_backward("071_kv_cache_update_with_rope")
    assert is_backward("062_kv_cache_update_with_rope_backward")
    assert family_key("071_kv_cache_update_with_rope") == family_key("062_kv_cache_update_with_rope_backward")
    assert sum(is_backward(n) for n in NAMES) == BACKWARD_ON_BOARD
    assert not any(is_backward(n) for n in BOARD["attention"] if "paged" in n)
    # the word inside a name is not the pass: "forward_backward" is the backward of the pair
    assert is_backward("009_expert_token_scatter_with_weighted_forward_backward")


# --- what the rules see --------------------------------------------------------------------


def test_rules_see_the_name_in_lower_case_without_numbering_or_backward():
    assert stem("069_rms_norm") == "rms_norm"
    assert stem("060_Fused_Attention_QK_Matmul_Scale_Mask_Softmax_Backward") == \
        "fused_attention_qk_matmul_scale_mask_softmax"
    assert stem("tasks/069_rms_norm") == "rms_norm" and stem("tasks/069_rms_norm/") == "rms_norm"
    assert stem("tasks\\069_rms_norm") == "rms_norm"
    # a number that is part of the name stays; hyphens and dots stay
    assert stem("079_ImageNet_83.6_ssm_output_projection_gate_multiply_backward") == \
        "imagenet_83.6_ssm_output_projection_gate_multiply"
    assert stem("051_seqlen-finetuned-reconstructed_hyena_complete_forward_block") == \
        "seqlen-finetuned-reconstructed_hyena_complete_forward_block"
    assert stem("123_") == "123_" and stem("rms_norm") == "rms_norm"


def test_a_task_directory_names_the_problem():
    assert family_key("tasks/069_rms_norm") == "rmsnorm"
    assert family_key(Path("tasks") / "012_gqa_paged_decode_h32_kv4_d128_ps1") == "attention"


def test_a_name_no_rule_matches_is_other():
    assert family_key("999_something_no_rule_names") == OTHER
    assert family_key("") == OTHER


@pytest.mark.parametrize("name,key", [
    # the first rule wins: a projection of attention tensors is a GEMM, an attention core
    # that carries a projection's word in a piece of its name is attention
    ("050_attention_qkv_projection_with_bias_and_reshape", "gemm"),
    ("049_attention_qk_matmul_with_gqa_repeat_and_scaling", "attention"),
    ("083_attention_score_value_matmul", "attention"),
    # a quantized projection is a quantized GEMM whoever eats it -- before `moe`
    ("005_fp8_moe_router_projection", "quant_gemm"),
    ("004_fp8_moe_expert_linear", "quant_gemm"),
    ("011_fp8_moe_gate_routing", "moe"),
    # ... a quantized attention core, RoPE or SSM step is its operator's
    ("019_nvfp4_grouped_query_attention", "attention"),
    ("014_fp8_yarn_rope_embedding", "rope"),
    ("023_fp8_mamba2_ssm_discretization", "ssm"),
    ("030_nvfp4_fused_decoder_layer_pre_norm_attention", "transformer_block"),
    # attention and a position embedding in one name: whichever the name says first
    ("014_rotary_embedding_with_attention_scaling_backward", "rope"),
    ("007_multimodal_rotary_embedding_attention", "attention"),
    ("047_attention_with_qk_norm_and_rope", "attention"),
    ("015_grouped_query_attention_with_rope_and_qk_norm", "attention"),
    ("018_fused_rope_with_qk_norm_and_kv_cache_update", "rope"),
    ("071_kv_cache_update_with_rope", "attention"),
    # a block stays a block over its parts
    ("080_moe_complete_layer_with_shared_expert_backward", "moe"),
    ("054_vision_encoder_layer_with_gated_residuals", "transformer_block"),
    ("077_whisper_decoder_output_projection", "gemm"),
    # "gelu_activation" is an activation, "mlp_gelu" an MLP
    ("085_geglu_activation", "activation"),
    ("022_vision_mlp_gelu_backward", "mlp"),
    # Mamba's and Hyena's convolutions are state-space steps; a VAE's is a convolution
    ("029_mamba_conv1d_with_gating", "ssm"),
    ("002_vae_conv3x3_groupnorm_silu_residual_fused", "conv"),
    ("046_conformer_relative_position_attention", "attention"),   # "conformer" is not "conv"
    # the few names whose words mislead every rule
    ("067_patch_embed_to_joint_attention_input", OTHER),
    ("052_altup_hidden_state_collapse_with_magnitude_normalization_backward", OTHER),
])
def test_the_rule_order_decides_the_names_that_could_go_two_ways(name, key):
    assert family_key(name) == key, name


def test_a_new_problem_of_a_known_kind_takes_its_kind():
    assert family_key("100_fp8_ffn_down_projection") == "quant_gemm"
    assert family_key("100_int4_gate_up_projection") == "quant_gemm"
    assert family_key("100_moe_shared_expert_down_projection") == "moe"   # not quantized: with MoE
    assert family_key("100_layer_norm_qwen3") == "norm"
    assert family_key("100_mamba2_conv1d_update") == "ssm"
    assert family_key("100_grouped_query_attention_flash_decoding") == "attention"


# --- the rule table -------------------------------------------------------------------------


def _table(tmp_path, monkeypatch, text: str):
    p = tmp_path / "families.toml"
    p.write_text(text, encoding="utf-8")
    monkeypatch.setattr(family, "families_file", lambda: p)
    return p


def test_a_rule_table_names_every_problem_before_any_of_it_is_used(tmp_path, monkeypatch):
    _table(tmp_path, monkeypatch, '[[rule]]\nfamily = "nope"\nmatch = "x"\n'
            '[[rule]]\nfamily = "gemm"\n'
            '[[rule]]\nfamily = "gemm"\nname = "a"\nmatch = "b"\n'
            '[[rule]]\nfamily = "gemm"\nmatch = "("\n'
            '[[rule]]\nfamily = "gemm"\nname = 7\n'
            '[[family]]\nkey = "Bad-Key"\n'
            '[[family]]\nkey = "gemm"\n'
            '[[family]]\nkey = "gemm"\n')
    with pytest.raises(FamilyError) as e:
        load_rules()
    problems = "\n".join(e.value.problems)
    assert len(e.value.problems) == 7
    for why in ("is not declared", "not a family key", "declares 'gemm' twice",
                "exactly one of `match`", "does not compile", "must be a list of names"):
        assert why in problems, why


def test_a_table_that_does_not_parse_or_exist_is_named(tmp_path, monkeypatch):
    _table(tmp_path, monkeypatch, "[[rule\n")
    with pytest.raises(FamilyError, match="does not parse"):
        load_rules()
    monkeypatch.setattr(family, "families_file", lambda: tmp_path / "absent.toml")
    with pytest.raises(FamilyError, match="cannot read it"):
        families()


def test_a_table_without_families_still_answers_other(tmp_path, monkeypatch):
    _table(tmp_path, monkeypatch, "")
    assert load_rules() == ((OTHER,), ())
    assert family_key("069_rms_norm") == OTHER
    _table(tmp_path, monkeypatch, '[[family]]\nkey = "gemm"\n'
            '[[rule]]\nfamily = "other"\nmatch = "^skip$"\n'
            '[[rule]]\nfamily = "gemm"\nmatch = "proj"\n'
            '[[rule]]\nfamily = "gemm"\nname = ["a_b", "c_d"]\n')
    keys, rules = load_rules()
    assert keys == ("gemm", OTHER) and len(rules) == 3
    assert family_key("001_a_b_backward") == family_key("007_c_d") == "gemm"
    assert family_key("001_skip_me") == OTHER and family_key("001_out_proj") == "gemm"
    # `name` is the whole name: a name that merely contains one of them does not match
    assert family_key("001_a_b_c") == OTHER


def test_a_table_is_read_again_when_it_changes(tmp_path, monkeypatch):
    p = _table(tmp_path, monkeypatch, '[[rule]]\nfamily = "other"\nmatch = "."\n')
    assert family_key("069_rms_norm") == OTHER
    p.write_text('[[family]]\nkey = "rmsnorm"\n[[rule]]\nfamily = "rmsnorm"\nmatch = "rms"\n'
                 + "# a longer table, so that size and time both move\n" * 20, encoding="utf-8")
    assert family_key("069_rms_norm") == "rmsnorm" and families() == ("rmsnorm", OTHER)
    assert parse_rules('[[family]]\nkey = "x"\n[[rule]]\nfamily = "x"\nmatch = "y"\n',
                       where="w.toml") == \
        (("x", OTHER), (family.Rule(family="x", match=re.compile("y"), name=()),))


def test_the_repository_table_declares_what_it_uses():
    keys, rules = load_rules()
    used = {r.family for r in rules}
    assert used <= set(keys)
    assert keys[:-1] and OTHER in used, "`other` needs the rule that files the names that mislead"
    assert all(r.match is not None or r.name for r in rules)


# --- the command line ----------------------------------------------------------------------


def test_the_command_line_names_the_family_of_each_problem(capsys):
    assert main(["069_rms_norm", "060_fused_attention_qk_matmul_scale_mask_softmax_backward",
                 "tasks/001_fp8_attention_output_projection"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "069_rms_norm  rmsnorm",
        "060_fused_attention_qk_matmul_scale_mask_softmax_backward  attention  (backward)",
        "tasks/001_fp8_attention_output_projection  quant_gemm"]


def test_the_command_line_prints_json(capsys):
    assert main(["069_rms_norm", "062_kv_cache_update_with_rope_backward", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == [
        {"name": "069_rms_norm", "family": "rmsnorm", "backward": False},
        {"name": "062_kv_cache_update_with_rope_backward", "family": "attention", "backward": True}]


def test_the_command_line_counts_a_snapshot_or_a_list_of_names(monkeypatch, capsys):
    snapshot = json.dumps({"data": {"kernel_leaderboards": [{"name": n, "kernel_id": i}
                                                            for i, n in enumerate(NAMES)]}})
    monkeypatch.setattr("sys.stdin", io.StringIO(snapshot))
    assert main(["--counts"]) == 0
    rows = [ln.split() for ln in capsys.readouterr().out.splitlines()]
    assert rows[-1] == ["total", "235"]
    assert dict(rows[:-1]) == {k: str(v) for k, v in count_by_family(NAMES).items()}
    assert dict(rows)["attention"] == "48"
    monkeypatch.setattr("sys.stdin", io.StringIO("\n069_rms_norm\n\n004_gemm_n128_k2048\n"))
    assert main(["--counts", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "total": 2, "counts": {**dict.fromkeys(families()[:-1], 0), "rmsnorm": 1, "gemm": 1, OTHER: 0}}


def test_the_command_line_refuses_to_guess_and_exits_2(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit) as e:
        main([])
    assert e.value.code == 2 and "--counts" in capsys.readouterr().err
    _table(tmp_path, monkeypatch, "[[rule\n")
    assert main(["069_rms_norm"]) == 2
    err = capsys.readouterr().err
    assert "[family]" in err and "does not parse" in err


def test_the_command_line_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "--counts" in out and "--json" in out and "first rule" in out


def test_counts_by_family_is_robust_to_an_empty_list():
    assert sum(count_by_family([]).values()) == 0
    assert Counter(family_key(n) for n in NAMES) == Counter({k: len(v) for k, v in BOARD.items()})
