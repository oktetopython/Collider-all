#ifndef CHECKPOINT_HPP
#define CHECKPOINT_HPP
#include <string>
#include <vector>
struct CheckpointState {
    unsigned long long current_giant_step_offset; std::string target_pubkey_hex;
    unsigned long long baby_steps_count; unsigned long long giant_steps_total_range;
    unsigned long long steps_per_kernel_launch; int w_param; int htsz_param_mb;
    CheckpointState() : current_giant_step_offset(0), baby_steps_count(0), giant_steps_total_range(0), steps_per_kernel_launch(0), w_param(0), htsz_param_mb(0) {}
};
bool save_checkpoint(const CheckpointState& state, const std::string& filename);
bool load_checkpoint(CheckpointState& state, const std::string& filename);
#endif // CHECKPOINT_HPP
