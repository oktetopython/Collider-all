#include "checkpoint.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <cstring>
#include <cerrno>
bool save_checkpoint(const CheckpointState& state, const std::string& filename) {
    std::ofstream ofs(filename); if (!ofs) { std::cerr << "Error: Could not open checkpoint file '" << filename << "' for writing. Errno: " << errno << " (" << strerror(errno) << ")" << std::endl << std::flush; return false; }
    ofs << "current_giant_step_offset=" << state.current_giant_step_offset << std::endl;
    ofs << "target_pubkey_hex=" << state.target_pubkey_hex << std::endl;
    ofs << "baby_steps_count=" << state.baby_steps_count << std::endl;
    ofs << "giant_steps_total_range=" << state.giant_steps_total_range << std::endl;
    ofs << "steps_per_kernel_launch=" << state.steps_per_kernel_launch << std::endl;
    ofs << "w_param=" << state.w_param << std::endl;
    ofs << "htsz_param_mb=" << state.htsz_param_mb << std::endl;
    if (ofs.fail()) { std::cerr << "Error: Failed to write to checkpoint file: " << filename << ". Errno: " << errno << " (" << strerror(errno) << ")" << std::endl << std::flush; ofs.close(); return false; }
    ofs.close(); if (ofs.fail()) { std::cerr << "Error: Failed to close checkpoint file properly: " << filename << ". Errno: " << errno << " (" << strerror(errno) << ")" << std::endl << std::flush; return false; }
    std::cout << "[Checkpoint] State saved to " << filename << " (offset " << state.current_giant_step_offset << ")" << std::endl << std::flush;
    return true;
}
bool load_checkpoint(CheckpointState& state, const std::string& filename) {
    std::ifstream ifs(filename); if (!ifs) { return false; }
    std::string line; CheckpointState temp_state; bool loaded_offset = false;
    while (std::getline(ifs, line)) {
        size_t delimiter_pos = line.find('='); if (delimiter_pos == std::string::npos || delimiter_pos == 0 || delimiter_pos == line.length() - 1) continue;
        std::string key = line.substr(0, delimiter_pos); std::string value_str = line.substr(delimiter_pos + 1);
        try {
            if (key == "current_giant_step_offset") { temp_state.current_giant_step_offset = std::stoull(value_str); loaded_offset = true; }
            else if (key == "target_pubkey_hex") { temp_state.target_pubkey_hex = value_str; }
            else if (key == "baby_steps_count") { temp_state.baby_steps_count = std::stoull(value_str); }
            else if (key == "giant_steps_total_range") { temp_state.giant_steps_total_range = std::stoull(value_str); }
            else if (key == "steps_per_kernel_launch") { temp_state.steps_per_kernel_launch = std::stoull(value_str); }
            else if (key == "w_param") { temp_state.w_param = std::stoi(value_str); }
            else if (key == "htsz_param_mb") { temp_state.htsz_param_mb = std::stoi(value_str); }
        } catch (const std::exception& e) { std::cerr << "Error parsing checkpoint line: '" << line << "' (" << e.what() << ")" << std::endl; ifs.close(); return false; }
    }
    ifs.close();
    if (loaded_offset && !ifs.bad()) { state = temp_state; std::cout << "[Checkpoint] Loaded from " << filename << ". Resume offset " << state.current_giant_step_offset << std::endl << std::flush; return true;
    } else if (!loaded_offset && ifs.eof() && !ifs.bad()) { return false;
    } else if (ifs.bad()) { std::cerr << "Err: I/O error reading chkpt " << filename << std::endl << std::flush; return false; }
    return false;
}
