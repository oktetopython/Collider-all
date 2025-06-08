#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <random>
#include <cstring>
#include <cstdio>
#include <cctype>
#include <cmath> // Required for std::log2, std::round
#include "utils.hpp"
#include "checkpoint.hpp"

#ifdef WITH_NVML
#include <nvml.h>
#endif

enum LogLevel { LOG_DEBUG, LOG_INFO, LOG_WARN, LOG_ERROR }; LogLevel current_log_level = LOG_INFO;
void log_message(LogLevel level, const std::string& message) { /* ... same ... */
    if (level < current_log_level) return;
    auto now = std::chrono::system_clock::now(); auto now_c = std::chrono::system_clock::to_time_t(now);
    std::stringstream time_ss; time_ss << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S");
    std::cout << time_ss.str() << " ";
    switch (level) {
        case LOG_DEBUG: std::cout << "[DEBUG] "; break; case LOG_INFO:  std::cout << "[INFO]  "; break;
        case LOG_WARN:  std::cout << "[WARN]  "; break; case LOG_ERROR: std::cout << "[ERROR] "; break;
    } std::cout << message << std::endl;
}
struct ProfileParams { int w_param=0; int htsz_param_mb=0; int conceptual_threads_per_block=0; int conceptual_blocks_factor=0; bool loaded=false;};
struct GpuProfile { std::string name; int matches_sm_major=0; ProfileParams params;};
std::vector<GpuProfile> available_profiles;
enum cudaError_t { cudaSuccess = 0, cudaErrorMemoryAllocation, cudaErrorInvalidValue, cudaErrorSetDeviceFailed, cudaErrorNotYetImplemented };
const char* cudaGetErrorString(cudaError_t err) { switch(err){case cudaSuccess:return "cudaSuccess";default:return "Unknown";}}
struct cudaDeviceProp { char name[256]; size_t totalGlobalMem; int major; int minor; int multiProcessorCount; };
std::vector<cudaDeviceProp> mock_gpu_props; int mock_gpu_count = 0;

#ifndef BUILD_WITH_CUDA
int current_mock_device_id_stub = -1; // Used by cudaMemGetInfo stub
#endif

std::default_random_engine random_generator(std::chrono::system_clock::now().time_since_epoch().count());
std::uniform_int_distribution<int> temp_dist(40, 80);
void initialize_mock_gpus(int count) { /* ... same ... */
    mock_gpu_count = count; mock_gpu_props.clear();
    int cc_majors[] = {7,7,8,8,9}; int cc_minors[] = {0,5,0,6,0}; int sm_counts[]={20,28,68,82,128};
    int cc_cycle_len = sizeof(cc_majors)/sizeof(int);
    for(int i=0;i<count;++i){cudaDeviceProp p;snprintf(p.name,sizeof(p.name),"MockGPU-%d-SM%d.%d",i,cc_majors[i%cc_cycle_len],cc_minors[i%cc_cycle_len]);p.totalGlobalMem=(1024ULL*1024*1024)*(8+i%2);p.major=cc_majors[i%cc_cycle_len];p.minor=cc_minors[i%cc_cycle_len];p.multiProcessorCount=sm_counts[i%cc_cycle_len];mock_gpu_props.push_back(p);}
    if(count>0) log_message(LOG_DEBUG, "Initialized " + std::to_string(count) + " mock GPUs.");
}
cudaError_t cudaGetDeviceCount(int* count) { *count = mock_gpu_count; return cudaSuccess; }
cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device) { if(device<0||device>=mock_gpu_count)return cudaErrorInvalidValue;*prop=mock_gpu_props[device];return cudaSuccess;}
cudaError_t cudaSetDevice(int device) {
    if(device<0||device>=mock_gpu_count) return cudaErrorSetDeviceFailed;
    log_message(LOG_DEBUG,"[CUDA STUB] Set active device to "+std::to_string(device));
    #ifndef BUILD_WITH_CUDA
    current_mock_device_id_stub = device;
    #endif
    return cudaSuccess;
}

#ifndef BUILD_WITH_CUDA
cudaError_t cudaMemGetInfo(size_t* free_mem, size_t* total_mem) {
    if (!free_mem || !total_mem) return cudaErrorInvalidValue;
    if (current_mock_device_id_stub >= 0 && current_mock_device_id_stub < mock_gpu_count) {
        *total_mem = mock_gpu_props[current_mock_device_id_stub].totalGlobalMem;
        // Simulate some memory usage, leave 90% free for testing purposes
        *free_mem = static_cast<size_t>(static_cast<double>(*total_mem) * 0.90);
    } else {
        // Fallback if no device set or invalid device ID
        *total_mem = 8ULL * 1024 * 1024 * 1024; // Default to 8GB total
        *free_mem = static_cast<size_t>(static_cast<double>(*total_mem) * 0.90);  // Default to 90% of 8GB free
        log_message(LOG_DEBUG, "[CUDA STUB] cudaMemGetInfo: current_mock_device_id_stub ("+ std::to_string(current_mock_device_id_stub) +") is invalid or no device set, using default fallback values.");
    }
    log_message(LOG_DEBUG, "[CUDA STUB] cudaMemGetInfo for effective device " + std::to_string(current_mock_device_id_stub) +
                           ": Free=" + std::to_string(*free_mem / (1024*1024)) + "MB, Total=" + std::to_string(*total_mem / (1024*1024)) + "MB");
    return cudaSuccess;
}
#endif

cudaError_t cudaMalloc(void** devPtr, size_t size) { *devPtr=malloc(size);log_message(LOG_DEBUG,"[CUDA STUB] cudaMalloc "+std::to_string(size)+" bytes. Ptr: "+(*devPtr?"non-null":"NULL"));return(*devPtr!=nullptr||size==0)?cudaSuccess:cudaErrorMemoryAllocation;}
enum cudaMemcpyKind {cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost};
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) { memcpy(dst,src,count);return cudaSuccess;}
cudaError_t cudaFree(void* devPtr) { if(devPtr)free(devPtr);log_message(LOG_DEBUG,"[CUDA STUB] cudaFree called.");return cudaSuccess;}
cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
void launch_bsgs_kernel_stub(int,const unsigned char*,unsigned long long,unsigned long long,void*,size_t,bool*,unsigned char*,int&);
struct BabyStepEntry { unsigned char point_representation[32]; unsigned long long k_value; };
struct AppConfig {
    std::string target_pubkey_hex; std::vector<int> gpu_ids;
    unsigned long long baby_steps_count = 1<<10; unsigned long long giant_steps_total_range = 1ULL<<20;
    unsigned long long steps_per_kernel_launch = 1<<12; int simulated_gpu_count = 1;
    int w_param = 64; int htsz_param_mb = 2048;
    std::string checkpoint_file = "bsgs_checkpoint.dat"; unsigned long long checkpoint_interval = 1 << 15;
    std::string log_level_str = "INFO"; std::string profiles_config_file = "configs/gpu_profiles.json";
    std::string force_gpu_profile_name = "";
};

std::string trim_whitespace(const std::string& str) { /* ... same ... */
    const std::string whitespace = " \t\n\r\f\v"; size_t first = str.find_first_not_of(whitespace);
    if (std::string::npos == first) return ""; size_t last = str.find_last_not_of(whitespace);
    return str.substr(first, (last - first + 1));
}
std::string find_json_value_manual(const std::string& json_object_str, const std::string& key, LogLevel L=LOG_DEBUG) { /* ... (same as previous, reduced verbosity) ... */
    std::string key_to_find = "\"" + key + "\"";
    size_t key_pos = json_object_str.find(key_to_find); if (key_pos == std::string::npos) return "";
    size_t colon_pos = json_object_str.find(':', key_pos + key_to_find.length()); if (colon_pos == std::string::npos) return "";
    size_t value_start = colon_pos + 1;
    while (value_start < json_object_str.length() && isspace(static_cast<unsigned char>(json_object_str[value_start]))) { value_start++; }
    if (value_start >= json_object_str.length()) return ""; char first_char = json_object_str[value_start];
    size_t value_end = value_start; std::string found_val;
    if (first_char == '"') { value_start++; value_end = json_object_str.find('"', value_start); if (value_end == std::string::npos) return ""; found_val = json_object_str.substr(value_start, value_end - value_start);  }
    else if (isdigit(static_cast<unsigned char>(first_char)) || first_char == '-' || first_char == '.') { value_end = value_start; while (value_end < json_object_str.length() && (isdigit(static_cast<unsigned char>(json_object_str[value_end])) || json_object_str[value_end] == '.' || json_object_str[value_end] == '-' || json_object_str[value_end] == '+')) { value_end++; } found_val = trim_whitespace(json_object_str.substr(value_start, value_end - value_start)); }
    else { return ""; }
    return found_val;
}

void load_gpu_profiles(const std::string& json_filename) { /* ... (same as previous with refined parsing logic) ... */
    available_profiles.clear(); log_message(LOG_DEBUG, "Attempting to load GPU profiles from: " + json_filename);
    std::ifstream ifs(json_filename); if (!ifs) { log_message(LOG_WARN, "GPU profiles file not found: " + json_filename); return; }
    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    size_t profiles_array_start = content.find("\"profiles\":"); if (profiles_array_start == std::string::npos) { log_message(LOG_WARN, "JSON: 'profiles' array key not found."); return; }
    profiles_array_start = content.find('[', profiles_array_start); if (profiles_array_start == std::string::npos) { log_message(LOG_WARN, "JSON: '[' not found for 'profiles' array."); return; }
    size_t profiles_array_end = content.find(']', profiles_array_start); if (profiles_array_end == std::string::npos) { log_message(LOG_WARN, "JSON: ']' not found for 'profiles' array."); return; }
    size_t current_obj_pos = profiles_array_start;
    while(true) {
        current_obj_pos = content.find('{', current_obj_pos); if (current_obj_pos == std::string::npos || current_obj_pos > profiles_array_end) break;
        size_t obj_end_pos = content.find('}', current_obj_pos); if (obj_end_pos == std::string::npos || obj_end_pos > profiles_array_end) { log_message(LOG_WARN, "JSON: Mismatched '}' in profile object."); break; }
        std::string profile_str = content.substr(current_obj_pos, obj_end_pos - current_obj_pos + 1);
        GpuProfile prof; bool profile_parse_ok = true;
        prof.name = trim_whitespace(find_json_value_manual(profile_str, "name", LOG_DEBUG)); if(prof.name.empty()){ log_message(LOG_WARN, "Profile name is empty after parsing/trimming."); profile_parse_ok = false; }
        std::string sm_major_str = find_json_value_manual(profile_str, "matches_sm_major", LOG_DEBUG);
        if (!sm_major_str.empty()) { try { prof.matches_sm_major = std::stoi(sm_major_str); } catch(const std::exception&e){log_message(LOG_WARN, "    StoI error for sm_major '" + sm_major_str + "': " + e.what()); profile_parse_ok = false;}}
        else { log_message(LOG_WARN, "    sm_major string is empty or not found for profile '" + prof.name + "'."); profile_parse_ok = false;}
        size_t params_block_start = profile_str.find("\"params\":");
        if (params_block_start != std::string::npos && profile_parse_ok) {
            params_block_start = profile_str.find('{', params_block_start); size_t params_block_end = profile_str.find('}', params_block_start);
            if (params_block_start != std::string::npos && params_block_end != std::string::npos) {
                std::string params_obj_str = profile_str.substr(params_block_start, params_block_end - params_block_start + 1); bool params_ok = true;
                std::string w_str = find_json_value_manual(params_obj_str, "w_param", LOG_DEBUG); if (!w_str.empty()) { try {prof.params.w_param = std::stoi(w_str);} catch(const std::exception&e){params_ok=false;}} else {params_ok=false;}
                std::string htsz_str = find_json_value_manual(params_obj_str, "htsz_param_mb", LOG_DEBUG); if (!htsz_str.empty()) { try {prof.params.htsz_param_mb = std::stoi(htsz_str);} catch(const std::exception&e){params_ok=false;}} else {params_ok=false;}
                std::string tpb_str = find_json_value_manual(params_obj_str, "conceptual_threads_per_block", LOG_DEBUG); if (!tpb_str.empty()) { try {prof.params.conceptual_threads_per_block = std::stoi(tpb_str);} catch(const std::exception&e){params_ok=false;}} else {params_ok=false;}
                std::string bf_str = find_json_value_manual(params_obj_str, "conceptual_blocks_factor", LOG_DEBUG); if (!bf_str.empty()) { try {prof.params.conceptual_blocks_factor = std::stoi(bf_str);} catch(const std::exception&e){params_ok=false;}} else {params_ok=false;}
                if (params_ok) prof.params.loaded = true; else {prof.params.loaded = false; profile_parse_ok=false;}
            } else { prof.params.loaded = false; profile_parse_ok=false;}
        } else { prof.params.loaded = false; profile_parse_ok=false;}
        if (profile_parse_ok) { available_profiles.push_back(prof); log_message(LOG_INFO, "Successfully Parsed and Stored profile: '" + prof.name + "' (SM: " + std::to_string(prof.matches_sm_major) + ", Loaded: " + (prof.params.loaded?"T":"F") + ")");}
        else { log_message(LOG_WARN, "Skipping profile '" + prof.name + "' due to parsing errors or missing critical fields.");}
        current_obj_pos = obj_end_pos + 1;
    }
    log_message(LOG_INFO, "Load_gpu_profiles: " + std::to_string(available_profiles.size()) + " valid profiles loaded in total.");
}

bool parse_arguments(int argc, char* argv[], AppConfig& config) { /* ... (same as previous, space-separated) ... */
    if (argc < 2) { std::cerr << "Usage: " << argv[0] << " <target_pubkey_hex_64_chars> [options]" << std::endl; return false; }
    config.target_pubkey_hex = argv[1];
    if (config.target_pubkey_hex.length() != 64) { std::cerr << "Error: Pubkey hex must be 64 chars." << std::endl; return false; }
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i]; std::string val = ""; bool next_arg_exists = (i + 1 < argc);
        if(next_arg_exists) val = argv[i+1]; else val = "";
        bool value_consumed = true;
        if (arg == "--gpu-list") { if(!next_arg_exists||val.empty()){std::cerr<<"Missing val for "<<arg<<std::endl;return false;} val.erase(std::remove(val.begin(),val.end(),' '),val.end()); std::stringstream ss(val); std::string s; config.gpu_ids.clear(); while(std::getline(ss,s,',')){if(s.empty())continue;try{config.gpu_ids.push_back(std::stoi(s));}catch(const std::exception&e){std::cerr<<"Err --gpu-list val: '"<<s<<"' ("<<e.what()<<")"<<std::endl;return false;}}}
        else if (arg == "--simulated-gpus") { if(!next_arg_exists||val.empty()){std::cerr<<"Missing val for "<<arg<<std::endl;return false;} try{config.simulated_gpu_count=std::stoi(val);if(config.simulated_gpu_count<0)throw std::out_of_range("neg");}catch(const std::exception&e){std::cerr<<"Err --sim-gpus val: '"<<val<<"' ("<<e.what()<<")"<<std::endl;return false;}}
        else if (arg == "--w") { if(!next_arg_exists||val.empty()){std::cerr<<"Missing val for "<<arg<<std::endl;return false;} try{config.w_param=std::stoi(val);}catch(const std::exception&e){std::cerr<<"Err --w val: '"<<val<<"' ("<<e.what()<<")"<<std::endl;return false;}}
        else if (arg == "--htsz") { if(!next_arg_exists||val.empty()){std::cerr<<"Missing val for "<<arg<<std::endl;return false;} try{config.htsz_param_mb=std::stoi(val);}catch(const std::exception&e){std::cerr<<"Err --htsz val: '"<<val<<"' ("<<e.what()<<")"<<std::endl;return false;}}
        else if (arg == "--checkpoint-file") { if(!next_arg_exists||val.empty()){std::cerr<<"Missing val for "<<arg<<std::endl;return false;} config.checkpoint_file=val;}
        else if (arg == "--checkpoint-interval") { if(!next_arg_exists||val.empty()){std::cerr<<"Missing val for "<<arg<<std::endl;return false;} try{config.checkpoint_interval=std::stoull(val);}catch(const std::exception&e){std::cerr<<"Error parsing --checkpoint-interval. Input string was: '"<<val<<"'. Exception: "<<e.what()<<std::endl;return false;}}
        else if (arg == "--log-level") { if(!next_arg_exists||val.empty()){std::cerr<<"Missing val for "<<arg<<std::endl;return false;} config.log_level_str=val;}
        else if (arg == "--profiles-file") { if(!next_arg_exists||val.empty()){std::cerr<<"Missing val for "<<arg<<std::endl;return false;} config.profiles_config_file=val;}
        else if (arg == "--gpu-profile") { if(!next_arg_exists||val.empty()){std::cerr<<"Missing val for "<<arg<<std::endl;return false;} config.force_gpu_profile_name=trim_whitespace(val); /* Trim here */ }
        else if (arg == "--baby-steps-count") { if(!next_arg_exists||val.empty()){std::cerr<<"Missing val for "<<arg<<std::endl;return false;} try{config.baby_steps_count=std::stoull(val);}catch(const std::exception&e){std::cerr<<"Err --baby val: '"<<val<<"' ("<<e.what()<<")"<<std::endl;return false;}}
        else if (arg == "--giant-steps-total-range") { if(!next_arg_exists||val.empty()){std::cerr<<"Missing val for "<<arg<<std::endl;return false;} try{config.giant_steps_total_range=std::stoull(val);}catch(const std::exception&e){std::cerr<<"Err --gs-range val: '"<<val<<"' ("<<e.what()<<")"<<std::endl;return false;}}
        else if (arg == "--steps-per-kernel-launch") { if(!next_arg_exists||val.empty()){std::cerr<<"Missing val for "<<arg<<std::endl;return false;} try{config.steps_per_kernel_launch=std::stoull(val);}catch(const std::exception&e){std::cerr<<"Err --steps-kern val: '"<<val<<"' ("<<e.what()<<")"<<std::endl;return false;}}
        else { std::cerr << "Unknown arg: " << arg << std::endl; return false; value_consumed = false;}
        if(value_consumed) i++; else { /* Error or flag only */ }
    }
    if (config.log_level_str == "DEBUG") current_log_level = LOG_DEBUG; else if (config.log_level_str == "WARN") current_log_level = LOG_WARN; else if (config.log_level_str == "ERROR") current_log_level = LOG_ERROR; else current_log_level = LOG_INFO;
    initialize_mock_gpus(config.simulated_gpu_count); load_gpu_profiles(config.profiles_config_file);
    return true;
}

// Function to print string as hex for debugging
#ifdef WITH_NVML
bool nvml_initialized_successfully = false;
#endif

std::string string_to_hex_log(const std::string& input) { /* ... same ... */
    std::stringstream ss; ss << std::hex << std::setfill('0');
    for (char c : input) { ss << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(c));}
    return ss.str();
}

int main(int argc, char* argv[]) { /* ... (same as previous, with corrected checkpoint_loaded logic and more logging) ... */
    auto app_start_time = std::chrono::high_resolution_clock::now(); AppConfig config;
    if (!parse_arguments(argc, argv, config)) { return 1; }

    #ifdef WITH_NVML
        log_message(LOG_INFO, "Attempting to initialize NVML for temperature monitoring...");
        nvmlReturn_t nvml_result = nvmlInit_v2();
        if (nvml_result == NVML_SUCCESS) {
            nvml_initialized_successfully = true;
            log_message(LOG_INFO, "NVML initialized successfully.");
        } else {
            log_message(LOG_WARN, "Failed to initialize NVML. Error: " + std::string(nvmlErrorString(nvml_result)) + ". GPU temperature monitoring will be disabled.");
            // nvml_initialized_successfully remains false
        }
    #endif

    log_message(LOG_INFO, "Collider-BSGS Host Application Started."); log_message(LOG_INFO, "Log level set to: " + config.log_level_str);
    unsigned char target_pubkey[32]; if (!hex_to_bytes(config.target_pubkey_hex, target_pubkey, sizeof(target_pubkey))) { log_message(LOG_ERROR, "Failed to parse public key: " + config.target_pubkey_hex); return 1; }
    log_message(LOG_INFO, "Target Public Key (first 4 bytes): " + config.target_pubkey_hex.substr(0,8) + "...");
    CheckpointState cp_state; unsigned long long resume_offset = 0; bool checkpoint_loaded_flag = load_checkpoint(cp_state, config.checkpoint_file);
    if (checkpoint_loaded_flag) { /* ... restore ... */ } else { log_message(LOG_INFO, "No chkpt, starting fresh."); /* ... init cp_state ... */ }
    if(resume_offset >= config.giant_steps_total_range && checkpoint_loaded_flag) {log_message(LOG_INFO, "Search already completed per chkpt."); return 0;}
    int num_gpus = 0; cudaGetDeviceCount(&num_gpus); if(num_gpus==0&&config.simulated_gpu_count>0) num_gpus=mock_gpu_count; std::vector<int> gpus_to_use;
    if(config.gpu_ids.empty()){for(int i=0;i<num_gpus;++i)gpus_to_use.push_back(i);}else{for(int id:config.gpu_ids){if(id<0||id>=num_gpus){log_message(LOG_ERROR,"Invalid GPU ID "+std::to_string(id));return 1;}gpus_to_use.push_back(id);}}
    if(gpus_to_use.empty()){if(num_gpus > 0 && config.simulated_gpu_count > 0) {gpus_to_use.push_back(0);} else {log_message(LOG_INFO,"No GPUs. Exiting."); return 0;}}
    log_message(LOG_INFO, "Selected " + std::to_string(gpus_to_use.size()) + " GPU(s).");
    log_message(LOG_DEBUG, "Total profiles successfully parsed by load_gpu_profiles: " + std::to_string(available_profiles.size()));

    struct GpuTask {
        int gpu_id;
        unsigned long long original_start_offset;
        unsigned long long original_num_steps;
        unsigned long long actual_start_offset;
        unsigned long long actual_num_steps;
        cudaDeviceProp props;
        ProfileParams applied_params;
        bool profile_was_applied;
        #ifdef WITH_NVML
        nvmlDevice_t nvml_device_handle;
        bool nvml_handle_valid;
        #endif
    };

    std::vector<GpuTask> gpu_tasks;
    unsigned long long cumulative_original_offset = 0;
    unsigned long long global_resume_offset = resume_offset;

    log_message(LOG_INFO, "Calculating GPU task assignments and applying profiles...");
    for (size_t gpu_idx = 0; gpu_idx < gpus_to_use.size(); ++gpu_idx) {
        int current_gpu_id = gpus_to_use[gpu_idx];
        GpuTask current_task;
        current_task.gpu_id = current_gpu_id;
        cudaGetDeviceProperties(&current_task.props, current_task.gpu_id);

        #ifdef WITH_NVML
        current_task.nvml_handle_valid = false; // Initialize
        if (nvml_initialized_successfully) {
            nvmlReturn_t nvml_handle_result = nvmlDeviceGetHandleByIndex_v2(current_task.gpu_id, &current_task.nvml_device_handle);
            if (nvml_handle_result == NVML_SUCCESS) {
                current_task.nvml_handle_valid = true;
            } else {
                log_message(LOG_WARN, "GPU ID " + std::to_string(current_task.gpu_id) +
                                       ": Failed to get NVML device handle. Error: " + std::string(nvmlErrorString(nvml_handle_result)));
            }
        }
        #endif

        // Calculate Original Range
        current_task.original_start_offset = cumulative_original_offset;
        unsigned long long steps_for_this_gpu = 0;
        if (gpus_to_use.size() > 0) { // Should always be true here
            steps_for_this_gpu = (config.giant_steps_total_range / gpus_to_use.size()) +
                                 (gpu_idx < (config.giant_steps_total_range % gpus_to_use.size()) ? 1 : 0);
        }
        current_task.original_num_steps = steps_for_this_gpu;

        // Calculate Actual Range (after resume_offset)
        current_task.actual_start_offset = current_task.original_start_offset;
        current_task.actual_num_steps = current_task.original_num_steps;

        if (current_task.actual_num_steps > 0) {
            if (global_resume_offset >= current_task.original_start_offset + current_task.original_num_steps) {
                // This GPU's entire original range is already covered by resume_offset
                current_task.actual_num_steps = 0;
                // actual_start_offset should indicate the point from where it *would* have started if not completed
                current_task.actual_start_offset = current_task.original_start_offset + current_task.original_num_steps;
            } else if (global_resume_offset > current_task.original_start_offset) {
                // Part of this GPU's range is covered by resume_offset
                unsigned long long already_covered_steps = global_resume_offset - current_task.original_start_offset;
                current_task.actual_start_offset = global_resume_offset; // Start from the resume_offset
                current_task.actual_num_steps = current_task.original_num_steps - already_covered_steps;
            }
        }

        // Ensure actual steps do not exceed giant_steps_total_range
        if (current_task.actual_start_offset >= config.giant_steps_total_range) {
            current_task.actual_num_steps = 0;
        } else {
            if (current_task.actual_start_offset + current_task.actual_num_steps > config.giant_steps_total_range) {
                current_task.actual_num_steps = config.giant_steps_total_range - current_task.actual_start_offset;
            }
        }
        // Final safety net for zero total range or if actual_start_offset somehow ended up at or beyond the total range
        if (config.giant_steps_total_range == 0) {
            current_task.actual_num_steps = 0;
        }
        if (current_task.actual_num_steps > 0 && current_task.actual_start_offset >= config.giant_steps_total_range) {
             current_task.actual_num_steps = 0;
        }


        // Log Calculated Ranges
        std::string original_range_end_str = (current_task.original_num_steps == 0) ? "N/A" : std::to_string(current_task.original_start_offset + current_task.original_num_steps - 1);
        std::string actual_range_end_str = (current_task.actual_num_steps == 0) ? "N/A" : std::to_string(current_task.actual_start_offset + current_task.actual_num_steps - 1);
        if (current_task.actual_num_steps == 0 && current_task.actual_start_offset > 0 && current_task.original_num_steps > 0) {
             // If actual_num_steps is 0, but it had original steps, it means it was skipped or completed
             // actual_start_offset might be at the end of its original range or at global_resume_offset
             if (global_resume_offset >= current_task.original_start_offset + current_task.original_num_steps) {
                // Entirely skipped/completed
                 if (current_task.original_start_offset + current_task.original_num_steps > 0) { // Avoid underflow if original_num_steps was 0
                    actual_range_end_str = std::to_string(current_task.original_start_offset + current_task.original_num_steps -1) + " (completed)";
                 } else {
                    actual_range_end_str = "N/A (completed)";
                 }
             } else if (current_task.actual_start_offset >= config.giant_steps_total_range) {
                 // Starts beyond or at the end of total range
                 actual_range_end_str = "N/A (beyond total range)";
             }
        }


        log_message(LOG_INFO, "GPU ID " + std::to_string(current_task.gpu_id) + " (" + current_task.props.name + "): " +
                               "Assigned Original Range: " + std::to_string(current_task.original_start_offset) + " - " + original_range_end_str + " (Steps: " + std::to_string(current_task.original_num_steps) + "). " +
                               "Effective Range to Process: " + std::to_string(current_task.actual_start_offset) + " - " + actual_range_end_str + " (Steps: " + std::to_string(current_task.actual_num_steps) + ").");

        // Apply GPU Profile
        log_message(LOG_DEBUG, "  Applying profile for GPU ID " + std::to_string(current_task.gpu_id) + "...");
        current_task.applied_params.w_param = config.w_param;
        current_task.applied_params.htsz_param_mb = config.htsz_param_mb;
        current_task.applied_params.conceptual_threads_per_block = 256; // Default
        current_task.applied_params.conceptual_blocks_factor = 64;    // Default
        current_task.applied_params.loaded = false; // Indicates it's not from a profile initially
        current_task.profile_was_applied = false;

        log_message(LOG_DEBUG, std::string("  Attempting to apply profile for GPU ") + std::to_string(current_task.gpu_id) + ". Forced profile name: '" + config.force_gpu_profile_name + "'");
        if(!config.force_gpu_profile_name.empty()){
            log_message(LOG_DEBUG, "    Attempting FORCED profile match.");
            for(const auto& p : available_profiles){
                bool name_match = (p.name == config.force_gpu_profile_name);
                bool params_are_loaded = p.params.loaded;
                log_message(LOG_DEBUG, std::string("      Checking forced: Profile.name:'") + p.name +
                                     "') vs ForcedName:'" + config.force_gpu_profile_name +
                                     "'. NameMatch=" + (name_match?"T":"F") + ". ParamsLoaded=" + (params_are_loaded?"T":"F"));
                if(name_match && params_are_loaded){
                    current_task.applied_params = p.params;
                    current_task.profile_was_applied = true;
                    log_message(LOG_INFO,"  GPU ID "+std::to_string(current_task.gpu_id)+": Applied FORCED profile '"+p.name+"'");
                    break;
                }
            }
            if(!current_task.profile_was_applied) log_message(LOG_WARN, "    Forced profile '" + config.force_gpu_profile_name + "' for GPU ID " + std::to_string(current_task.gpu_id) + " not found or its params not loaded.");
        } else {
            log_message(LOG_DEBUG, "    Attempting AUTO-match based on SM " + std::to_string(current_task.props.major) + " for GPU ID " + std::to_string(current_task.gpu_id));
            for(const auto& p : available_profiles){
                bool sm_match = (p.matches_sm_major == current_task.props.major);
                bool params_are_loaded = p.params.loaded;
                log_message(LOG_DEBUG, std::string("      Checking auto-match: Profile:'") + p.name + "' (SM_match:"+std::to_string(p.matches_sm_major)+") vs GPU SM: "+std::to_string(current_task.props.major) + ". SM_Match=" + (sm_match?"T":"F") + ". Params_Loaded=" + (params_are_loaded?"T":"F"));
                if(sm_match && params_are_loaded){
                    current_task.applied_params = p.params;
                    current_task.profile_was_applied = true;
                    log_message(LOG_INFO,"  GPU ID "+std::to_string(current_task.gpu_id)+": Applied AUTO-MATCHED profile '"+p.name+"'");
                    break;
                }
            }
        }

        if(current_task.profile_was_applied){
            log_message(LOG_INFO,"    Using Profile Params for GPU " + std::to_string(current_task.gpu_id) +
                                 ": w=" + std::to_string(current_task.applied_params.w_param) +
                                 ", htsz_mb=" + std::to_string(current_task.applied_params.htsz_param_mb) +
                                 ", TPB=" + std::to_string(current_task.applied_params.conceptual_threads_per_block) +
                                 ", BlocksFactor=" + std::to_string(current_task.applied_params.conceptual_blocks_factor));
        } else {
            log_message(LOG_INFO,"  GPU ID "+std::to_string(current_task.gpu_id)+
                                 ": No specific profile applied. Using default/cmd-line params: w=" +
                                 std::to_string(current_task.applied_params.w_param) +
                                 ", htsz_mb=" + std::to_string(current_task.applied_params.htsz_param_mb) +
                                 ", TPB=" + std::to_string(current_task.applied_params.conceptual_threads_per_block) + // Log these too
                                 ", BlocksFactor=" + std::to_string(current_task.applied_params.conceptual_blocks_factor));
        }

        gpu_tasks.push_back(current_task);
        cumulative_original_offset += current_task.original_num_steps;
    }

    // Conceptual Baby Step Generation
    log_message(LOG_INFO, "Starting Baby Step generation (conceptual)...");
    std::vector<BabyStepEntry> h_baby_steps_table;
    unsigned long long num_baby_steps_to_generate = config.baby_steps_count;
    // Conceptual loop - can be filled with actual generation logic later
    for (unsigned long long i = 0; i < num_baby_steps_to_generate; ++i) {
        // BabyStepEntry entry;
        // entry.k_value = i;
        // memset(entry.point_representation, 0, 32); // Dummy data
        // entry.point_representation[0] = static_cast<unsigned char>(i); // Example dummy data
        // h_baby_steps_table.push_back(entry);
    }
    if (h_baby_steps_table.empty() && num_baby_steps_to_generate > 0) { // Add at least one dummy entry if table is empty but should have items.
        BabyStepEntry dummy_entry; dummy_entry.k_value = 0; memset(dummy_entry.point_representation,0,32);
        h_baby_steps_table.push_back(dummy_entry);
        log_message(LOG_DEBUG, "Populated h_baby_steps_table with one dummy entry as it was empty.");
    }
    log_message(LOG_INFO, "Baby Step generation (conceptual). Table size: " + std::to_string(h_baby_steps_table.size()) + ", Expected: " + std::to_string(num_baby_steps_to_generate));

    bool key_found_overall = false;
    unsigned char found_key_data[32]; // Assuming private key is 32 bytes
    memset(found_key_data, 0, sizeof(found_key_data));
    unsigned long long total_giant_steps_processed_overall = global_resume_offset; // Start from where we resumed

    // Ensure cp_state.current_giant_step_offset is also initialized from global_resume_offset if a checkpoint was loaded.
    // If no checkpoint was loaded, resume_offset (and thus global_resume_offset) would be 0.
    // cp_state.current_giant_step_offset is already correctly set from load_checkpoint or initialized to 0.
    // So, total_giant_steps_processed_overall should align with it.
    if (checkpoint_loaded_flag) {
        total_giant_steps_processed_overall = cp_state.current_giant_step_offset;
         log_message(LOG_INFO, "Resuming. Total giant steps processed so far (from checkpoint): " + std::to_string(total_giant_steps_processed_overall));
    }


    for (GpuTask& task : gpu_tasks) {
        if (key_found_overall) {
            log_message(LOG_INFO, "Key already found. Skipping further GPU tasks.");
            break;
        }
        if (task.actual_num_steps == 0) {
            log_message(LOG_INFO, "GPU ID " + std::to_string(task.gpu_id) + " (" + task.props.name + "): No actual steps to process. Skipping.");
            continue;
        }

        log_message(LOG_INFO, "Starting processing for GPU ID " + std::to_string(task.gpu_id) + " (" + task.props.name + ") for " +
                               std::to_string(task.actual_num_steps) + " giant steps, starting from its assigned offset " + std::to_string(task.actual_start_offset) +
                               " (Global offset " + std::to_string(task.actual_start_offset) + ").");

        cudaError_t cuda_err;
        cuda_err = cudaSetDevice(task.gpu_id);
        if (cuda_err != cudaSuccess) {
            log_message(LOG_ERROR, "Failed to set device to GPU ID " + std::to_string(task.gpu_id) + ". Error: " + cudaGetErrorString(cuda_err) + ". Skipping this GPU.");
            continue;
        }

        // VRAM-based Parameter Adjustment
        size_t free_mem_on_gpu = 0;
        size_t total_mem_on_gpu = 0;
        bool skip_this_gpu_due_to_vram = false;
        unsigned int initial_w_param_for_task = task.applied_params.w_param;

        if (task.applied_params.w_param == 0) {
            if (config.baby_steps_count > 0) {
                double N_val = static_cast<double>(config.baby_steps_count);
                if (N_val > 0) task.applied_params.w_param = static_cast<unsigned int>(std::round(std::log2(N_val)));
                if (task.applied_params.w_param == 0) task.applied_params.w_param = 10; // Ensure w_param is not 0 if log2 was 0
                log_message(LOG_DEBUG, "GPU " + std::to_string(task.gpu_id) + ": Initial task.applied_params.w_param was 0, inferred as " + std::to_string(task.applied_params.w_param) + " from config.baby_steps_count.");
            } else {
                 task.applied_params.w_param = 10; // Default if both profile and global config.baby_steps_count are zero
                 log_message(LOG_DEBUG, "GPU " + std::to_string(task.gpu_id) + ": Initial task.applied_params.w_param and config.baby_steps_count are zero. Defaulting w_param to " + std::to_string(task.applied_params.w_param) + ".");
            }
            initial_w_param_for_task = task.applied_params.w_param; // Update initial_w_param_for_task to the inferred value
        }

        cudaError_t mem_info_err;
        #ifdef BUILD_WITH_CUDA
            mem_info_err = cudaMemGetInfo(&free_mem_on_gpu, &total_mem_on_gpu);
        #else
            mem_info_err = cudaMemGetInfo(&free_mem_on_gpu, &total_mem_on_gpu);
        #endif

        if (mem_info_err == cudaSuccess) {
            log_message(LOG_DEBUG, "GPU " + std::to_string(task.gpu_id) + ": Params before VRAM check: w_param=" + std::to_string(task.applied_params.w_param) +
                                   ", htsz_param_mb=" + std::to_string(task.applied_params.htsz_param_mb) +
                                   ". Available VRAM: " + std::to_string(free_mem_on_gpu / (1024*1024)) + "MB.");

            unsigned int current_w_being_checked = task.applied_params.w_param;
            unsigned long long estimated_baby_step_entries = (1ULL << current_w_being_checked);
            size_t baby_table_size_bytes = estimated_baby_step_entries * sizeof(BabyStepEntry);
            // Estimate other overheads: target pubkey, found_flag, found_private_key, kernel code, CUDA context, etc.
            size_t other_gpu_overhead = (100ULL * 1024 * 1024); // Generous 100MB for other things
            size_t required_mem_for_w = baby_table_size_bytes + other_gpu_overhead;

            // Try to leave about 15% VRAM free (use 85% of free_mem_on_gpu)
            while (required_mem_for_w > free_mem_on_gpu * 0.85 && current_w_being_checked > 10) {
                log_message(LOG_WARN, "GPU " + std::to_string(task.gpu_id) + ": w_param=" + std::to_string(current_w_being_checked) +
                                       " (needs approx. " + std::to_string(required_mem_for_w/(1024*1024)) + "MB total) " +
                                       "exceeds 85% of available VRAM (" + std::to_string(static_cast<size_t>(free_mem_on_gpu*0.85)/(1024*1024)) + "MB). Reducing w_param.");
                current_w_being_checked--;
                estimated_baby_step_entries = (1ULL << current_w_being_checked);
                baby_table_size_bytes = estimated_baby_step_entries * sizeof(BabyStepEntry);
                required_mem_for_w = baby_table_size_bytes + other_gpu_overhead;
            }

            if (current_w_being_checked != initial_w_param_for_task) {
                log_message(LOG_INFO, "GPU " + std::to_string(task.gpu_id) + ": w_param auto-adjusted from " + std::to_string(initial_w_param_for_task) +
                                       " to " + std::to_string(current_w_being_checked) + " due to VRAM constraints. New estimated table size: " +
                                       std::to_string(baby_table_size_bytes/(1024*1024)) + "MB.");
                task.applied_params.w_param = current_w_being_checked;

                // Also adjust htsz_param_mb to reflect the new w_param
                unsigned int new_htsz_mb = static_cast<unsigned int>(baby_table_size_bytes / (1024*1024));
                if (new_htsz_mb == 0 && baby_table_size_bytes > 0) new_htsz_mb = 1; // Min 1MB if table is not empty
                else if (baby_table_size_bytes == 0) new_htsz_mb = 0; // Should not happen if w_param >= 10

                // Adjust htsz_param_mb if it was larger than the new table size, or if w itself changed (implying table size changed)
                if (task.applied_params.htsz_param_mb > new_htsz_mb || initial_w_param_for_task != current_w_being_checked) {
                     log_message(LOG_INFO, "GPU " + std::to_string(task.gpu_id) + ": htsz_param_mb also adjusted from " +
                                           std::to_string(task.applied_params.htsz_param_mb) + "MB to " + std::to_string(new_htsz_mb) + "MB.");
                    task.applied_params.htsz_param_mb = new_htsz_mb;
                }

            } else { // w_param was not changed by VRAM check, but it might have been inferred from 0 if initial_w_param_for_task was based on a profile value of 0.
                 task.applied_params.w_param = current_w_being_checked; // Ensure the (possibly inferred) w_param is set
            }

            if (current_w_being_checked <= 10 && required_mem_for_w > free_mem_on_gpu * 0.85) {
                log_message(LOG_ERROR, "GPU " + std::to_string(task.gpu_id) +
                                       ": Cannot meet VRAM requirements even with minimum w_param=" + std::to_string(current_w_being_checked) +
                                       ". Required: " + std::to_string(required_mem_for_w/(1024*1024)) + "MB. Marking GPU task to be skipped.");
                skip_this_gpu_due_to_vram = true;
            }
        } else {
            log_message(LOG_WARN, "GPU " + std::to_string(task.gpu_id) +
                                   ": Could not get memory info (Error: " + ((mem_info_err == cudaSuccess) ? "None" : cudaGetErrorString(mem_info_err)) +
                                   "). Using parameters without VRAM-based adjustment.");
            // If w_param was inferred from 0, ensure it's set
            if (initial_w_param_for_task != task.applied_params.w_param && task.applied_params.w_param == config.w_param) { // if task.applied_params.w_param is still the global default
                task.applied_params.w_param = initial_w_param_for_task; // make sure it uses the inferred one
                 log_message(LOG_DEBUG, "GPU " + std::to_string(task.gpu_id) + ": Using inferred w_param=" + std::to_string(task.applied_params.w_param) + " as VRAM check failed.");
            }
        }

        if (skip_this_gpu_due_to_vram) {
            log_message(LOG_ERROR, "GPU " + std::to_string(task.gpu_id) + " will be skipped due to VRAM constraints or inability to determine VRAM.");
            continue;
        }

        unsigned char* d_target_pubkey = nullptr;
        BabyStepEntry* d_baby_steps_table = nullptr;
        bool* d_found_flag = nullptr;
        unsigned char* d_found_private_key = nullptr;
        bool task_gpu_mem_success = true;

        unsigned long long num_baby_steps_for_gpu_device = (1ULL << task.applied_params.w_param);
        size_t baby_table_bytes_for_gpu = num_baby_steps_for_gpu_device * sizeof(BabyStepEntry);

        cuda_err = cudaMalloc(reinterpret_cast<void**>(&d_target_pubkey), 32);
        if (cuda_err != cudaSuccess) { log_message(LOG_ERROR, "GPU " + std::to_string(task.gpu_id) + ": cudaMalloc for d_target_pubkey failed: " + cudaGetErrorString(cuda_err)); task_gpu_mem_success = false; }

        if(task_gpu_mem_success && baby_table_bytes_for_gpu > 0){ // Only alloc if size is non-zero
            cuda_err = cudaMalloc(reinterpret_cast<void**>(&d_baby_steps_table), baby_table_bytes_for_gpu);
            if (cuda_err != cudaSuccess) { log_message(LOG_ERROR, "GPU " + std::to_string(task.gpu_id) + ": cudaMalloc for d_baby_steps_table ("+std::to_string(baby_table_bytes_for_gpu / (1024*1024))+"MB) failed: " + cudaGetErrorString(cuda_err)); task_gpu_mem_success = false; }
        } else if (baby_table_bytes_for_gpu == 0 && num_baby_steps_to_generate > 0) { // num_baby_steps_to_generate is from global config
            log_message(LOG_WARN, "GPU " + std::to_string(task.gpu_id) + ": Baby table size for device is 0 bytes due to w_param=" + std::to_string(task.applied_params.w_param) + ". Not allocating d_baby_steps_table.");
            // This might be fine if w_param is intentionally set very low, but kernel needs to handle 0 table size.
        }


        if(task_gpu_mem_success){
            cuda_err = cudaMalloc(reinterpret_cast<void**>(&d_found_flag), sizeof(bool));
            if (cuda_err != cudaSuccess) { log_message(LOG_ERROR, "GPU " + std::to_string(task.gpu_id) + ": cudaMalloc for d_found_flag failed: " + cudaGetErrorString(cuda_err)); task_gpu_mem_success = false; }
        }
        if(task_gpu_mem_success){
            cuda_err = cudaMalloc(reinterpret_cast<void**>(&d_found_private_key), 32); // Assuming 32-byte private key
            if (cuda_err != cudaSuccess) { log_message(LOG_ERROR, "GPU " + std::to_string(task.gpu_id) + ": cudaMalloc for d_found_private_key failed: " + cudaGetErrorString(cuda_err)); task_gpu_mem_success = false; }
        }

        if (!task_gpu_mem_success) {
            if (d_target_pubkey) cudaFree(d_target_pubkey);
            if (d_baby_steps_table) cudaFree(d_baby_steps_table);
            if (d_found_flag) cudaFree(d_found_flag);
            // d_found_private_key already covered by the check
            log_message(LOG_ERROR, "Skipping GPU ID " + std::to_string(task.gpu_id) + " due to memory allocation failures.");
            continue;
        }

        bool task_data_transfer_success = true;
        cuda_err = cudaMemcpy(d_target_pubkey, target_pubkey, 32, cudaMemcpyHostToDevice);
        if (cuda_err != cudaSuccess) { log_message(LOG_ERROR, "GPU " + std::to_string(task.gpu_id) + ": cudaMemcpy for d_target_pubkey failed: " + cudaGetErrorString(cuda_err)); task_data_transfer_success = false; }

        if(task_data_transfer_success && d_baby_steps_table != nullptr && !h_baby_steps_table.empty()){ // Check d_baby_steps_table to ensure it was allocated
             unsigned long long entries_to_copy = std::min(static_cast<unsigned long long>(h_baby_steps_table.size()), num_baby_steps_for_gpu_device);
             if (entries_to_copy > 0) {
                cuda_err = cudaMemcpy(d_baby_steps_table, h_baby_steps_table.data(), entries_to_copy * sizeof(BabyStepEntry), cudaMemcpyHostToDevice);
                if (cuda_err != cudaSuccess) { log_message(LOG_ERROR, "GPU " + std::to_string(task.gpu_id) + ": cudaMemcpy for d_baby_steps_table ("+std::to_string(entries_to_copy)+" entries) failed: " + cudaGetErrorString(cuda_err)); task_data_transfer_success = false; }
                else { log_message(LOG_DEBUG, "GPU " + std::to_string(task.gpu_id) + ": Copied " + std::to_string(entries_to_copy) + " entries to d_baby_steps_table.");}
             } else {
                log_message(LOG_DEBUG, "GPU " + std::to_string(task.gpu_id) + ": Zero entries to copy to d_baby_steps_table.");
             }
        } else if (d_baby_steps_table == nullptr && baby_table_bytes_for_gpu > 0) { // If allocation was intended but failed
             log_message(LOG_ERROR, "GPU " + std::to_string(task.gpu_id) + ": d_baby_steps_table is null, cannot transfer baby steps. This indicates prior malloc failure.");
             task_data_transfer_success = false; // Should have been caught by task_gpu_mem_success already
        } else if (h_baby_steps_table.empty() && num_baby_steps_for_gpu_device > 0) { // If host table is empty but GPU expects entries
            log_message(LOG_WARN, "GPU " + std::to_string(task.gpu_id) + ": Host baby steps table is empty, but GPU w_param implies " + std::to_string(num_baby_steps_for_gpu_device) + " entries. Nothing to copy.");
        }


        if(task_data_transfer_success){
            bool h_found_flag_initial = false;
            cuda_err = cudaMemcpy(d_found_flag, &h_found_flag_initial, sizeof(bool), cudaMemcpyHostToDevice);
            if (cuda_err != cudaSuccess) { log_message(LOG_ERROR, "GPU " + std::to_string(task.gpu_id) + ": cudaMemcpy for d_found_flag init failed: " + cudaGetErrorString(cuda_err)); task_data_transfer_success = false; }
        }

        if (!task_data_transfer_success) {
            cudaFree(d_target_pubkey); if (d_baby_steps_table) cudaFree(d_baby_steps_table); cudaFree(d_found_flag); cudaFree(d_found_private_key);
            log_message(LOG_ERROR, "Skipping GPU ID " + std::to_string(task.gpu_id) + " due to data transfer failures.");
            continue;
        }

        unsigned long long steps_processed_on_this_gpu_task = 0;
        unsigned long long current_kernel_global_start_offset = task.actual_start_offset; // This is the starting point for kernels for this task
        int kernel_launch_counter_for_this_task = 0; // Will be incremented by stub *after* first potential use for temp log
        unsigned long long last_checkpoint_save_step = cp_state.current_giant_step_offset;

        auto task_start_time = std::chrono::high_resolution_clock::now();
        // steps_processed_on_this_gpu_task is already initialized to 0 by its declaration scope

        while (steps_processed_on_this_gpu_task < task.actual_num_steps) {
            if (key_found_overall) break;

            unsigned long long steps_for_this_launch = std::min(static_cast<unsigned long long>(config.steps_per_kernel_launch), task.actual_num_steps - steps_processed_on_this_gpu_task);

            log_message(LOG_DEBUG, "GPU " + std::to_string(task.gpu_id) + ": Launching kernel for " + std::to_string(steps_for_this_launch) +
                                   " steps, starting at its assigned offset " + std::to_string(current_kernel_global_start_offset) +
                                   " (Processed on this GPU so far: " + std::to_string(steps_processed_on_this_gpu_task) + ").");

            #ifdef BUILD_WITH_CUDA
                // extern "C" void launch_bsgs_optimized_kernel_wrapper(...); // This should be in a header or at file scope
                // For now, using stub as per subtask instructions, actual kernel wrapper needs params from task.applied_params
                 log_message(LOG_DEBUG, "[CUDA KERNEL LAUNCH Placeholder] GPU " + std::to_string(task.gpu_id) + ". Actual kernel would use: " +
                                      "w=" + std::to_string(task.applied_params.w_param) +
                                      ", TPB=" + std::to_string(task.applied_params.conceptual_threads_per_block) +
                                      ", BlocksFactor=" + std::to_string(task.applied_params.conceptual_blocks_factor) +
                                      ", htsz_mb (for shared_mem)=" + std::to_string(task.applied_params.htsz_param_mb) +
                                      ", Kernel BS Table Size=" + std::to_string(num_baby_steps_for_gpu_device) );
                 launch_bsgs_kernel_stub(task.gpu_id, d_target_pubkey, current_kernel_global_start_offset, steps_for_this_launch,
                                        d_baby_steps_table, num_baby_steps_for_gpu_device, // Pass num_baby_steps_for_gpu_device
                                        d_found_flag, d_found_private_key, kernel_launch_counter_for_this_task);
            #else
                launch_bsgs_kernel_stub(task.gpu_id, d_target_pubkey, current_kernel_global_start_offset, steps_for_this_launch,
                                        d_baby_steps_table, num_baby_steps_for_gpu_device, // Pass num_baby_steps_for_gpu_device
                                        d_found_flag, d_found_private_key, kernel_launch_counter_for_this_task);
            #endif

            cuda_err = cudaDeviceSynchronize();
            if (cuda_err != cudaSuccess) {
                log_message(LOG_ERROR, "GPU " + std::to_string(task.gpu_id) + ": cudaDeviceSynchronize after kernel launch failed: " + cudaGetErrorString(cuda_err) + ". Skipping rest of this GPU's task.");
                break;
            }

            bool h_found_flag_gpu = false;
            cuda_err = cudaMemcpy(&h_found_flag_gpu, d_found_flag, sizeof(bool), cudaMemcpyDeviceToHost);
            if (cuda_err != cudaSuccess) {
                log_message(LOG_ERROR, "GPU " + std::to_string(task.gpu_id) + ": cudaMemcpy for d_found_flag readback failed: " + cudaGetErrorString(cuda_err) + ". Assuming key not found for this launch.");
                h_found_flag_gpu = false; // Proceed cautiously
            }

            if (h_found_flag_gpu) {
                log_message(LOG_INFO, "Key found by GPU ID " + std::to_string(task.gpu_id) + "!");
                cuda_err = cudaMemcpy(found_key_data, d_found_private_key, 32, cudaMemcpyDeviceToHost);
                if (cuda_err != cudaSuccess) {
                     log_message(LOG_ERROR, "GPU " + std::to_string(task.gpu_id) + ": cudaMemcpy for d_found_private_key readback failed: " + cudaGetErrorString(cuda_err) + ". Cannot retrieve found key!");
                     // key_found_overall might not be set if we can't get the key
                } else {
                    key_found_overall = true;
                    // Optionally, write to Found.txt here or after main loop
                    std::ofstream found_file("Found.txt", std::ios::app);
                    if (found_file.is_open()) {
                        found_file << "Found Key (hex): " << bytes_to_hex(found_key_data, 32) << std::endl;
                        found_file << "Associated Public Key: " << config.target_pubkey_hex << std::endl;
                        found_file << "Found by GPU: " << task.gpu_id << " (" << task.props.name << ")" << std::endl;
                        found_file << "At giant step offset (approx): " << current_kernel_global_start_offset + (steps_for_this_launch / 2) << std::endl; // Approximate location
                        found_file.close();
                    }
                }
                break;
            }

            steps_processed_on_this_gpu_task += steps_for_this_launch;
            current_kernel_global_start_offset += steps_for_this_launch; // This is the start for the *next* kernel

            // Checkpoint logic
            // total_giant_steps_processed_overall should be the highest point reached by any kernel start + its processed steps
            // The point up to which we are *sure* everything is done.
            // current_kernel_global_start_offset is now the beginning of the *next* unsearched block for this task.
            // So, task.actual_start_offset + steps_processed_on_this_gpu_task is the end of the completed block for this task.
            unsigned long long current_global_progress_for_task = task.actual_start_offset + steps_processed_on_this_gpu_task;
            total_giant_steps_processed_overall = std::max(total_giant_steps_processed_overall, current_global_progress_for_task);


            bool is_last_batch_for_task = (steps_processed_on_this_gpu_task == task.actual_num_steps); // True if this launch will complete the task
            bool should_checkpoint_this_iteration = ( ( (total_giant_steps_processed_overall - last_checkpoint_save_step >= config.checkpoint_interval) && config.checkpoint_interval > 0 ) || is_last_batch_for_task || key_found_overall);

            #ifdef WITH_NVML
            // Log temperature on first launch (kernel_launch_counter_for_this_task is 0 before first launch, 1 after) or if checkpointing
            if (task.nvml_handle_valid && (kernel_launch_counter_for_this_task == 0 || should_checkpoint_this_iteration) ) {
                unsigned int temp_c;
                nvmlReturn_t temp_result = nvmlDeviceGetTemperature(task.nvml_device_handle, NVML_TEMPERATURE_GPU, &temp_c);
                if (temp_result == NVML_SUCCESS) {
                    log_message(LOG_INFO, "GPU ID " + std::to_string(task.gpu_id) + " Temperature: " + std::to_string(temp_c) + " C");
                } else {
                    log_message(LOG_WARN, "GPU ID " + std::to_string(task.gpu_id) +
                                           ": Failed to get temperature via NVML. Error: " + std::string(nvmlErrorString(temp_result)));
                    // task.nvml_handle_valid = false; // Optional: stop trying if it fails once
                }
            }
            #endif

            if (should_checkpoint_this_iteration) {
                 if (total_giant_steps_processed_overall > cp_state.current_giant_step_offset) { // Only save if progress was made
                    cp_state.current_giant_step_offset = total_giant_steps_processed_overall;
                    if (save_checkpoint(cp_state, config.checkpoint_file)) {
                        log_message(LOG_INFO, "Checkpoint saved. Progress: " + std::to_string(cp_state.current_giant_step_offset) + " / " + std::to_string(config.giant_steps_total_range));
                        last_checkpoint_save_step = cp_state.current_giant_step_offset;
                    } else {
                        log_message(LOG_WARN, "Failed to save checkpoint at step " + std::to_string(cp_state.current_giant_step_offset));
                    }
                 }
            }
        } // End of inner while loop (kernel launches for the current task)

        cudaFree(d_target_pubkey);
        if(d_baby_steps_table) cudaFree(d_baby_steps_table);
        cudaFree(d_found_flag);
        cudaFree(d_found_private_key);

        // Per-GPU Task Completion Summary & Speed Log
        if (steps_processed_on_this_gpu_task > 0) {
            auto task_end_time_point = std::chrono::high_resolution_clock::now(); // Renamed to avoid conflict
            auto task_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(task_end_time_point - task_start_time);
            double task_duration_sec = task_duration_ms.count() / 1000.0;

            log_message(LOG_INFO, "GPU ID " + std::to_string(task.gpu_id) + " (" + task.props.name +
                                   "): Processed " + std::to_string(steps_processed_on_this_gpu_task) +
                                   " giant steps from offset " + std::to_string(task.actual_start_offset) +
                                   " to " + std::to_string(task.actual_start_offset + steps_processed_on_this_gpu_task - 1) + ".");
            if (task_duration_sec > 0.001) { // Avoid division by zero if duration is too small
                double steps_per_sec = static_cast<double>(steps_processed_on_this_gpu_task) / task_duration_sec;
                log_message(LOG_INFO, "GPU ID " + std::to_string(task.gpu_id) + " (" + task.props.name +
                                       "): Achieved average " + std::to_string(steps_per_sec) + " steps/sec for its segment (" +
                                       std::to_string(steps_processed_on_this_gpu_task) + " steps in " + std::to_string(task_duration_sec) + "s).");
            } else {
                log_message(LOG_INFO, "GPU ID " + std::to_string(task.gpu_id) + " (" + task.props.name +
                                       "): Processed " + std::to_string(steps_processed_on_this_gpu_task) + " steps too quickly to calculate meaningful speed.");
            }
        } else if (task.actual_num_steps > 0) { // It was supposed to do work but didn't (e.g. key found before it started, or error)
             log_message(LOG_INFO, "GPU ID " + std::to_string(task.gpu_id) + " (" + task.props.name +
                                   "): Did not process any steps. Assigned steps: " + std::to_string(task.actual_num_steps) +
                                   (key_found_overall ? " (key previously found or error)." : "."));
        }

        // Refined final status message for the GPU task
        // This replaces the original simpler log messages for task completion.
        // bool _h_found_flag_gpu_for_log_check = false; // Placeholder, as h_found_flag_gpu is out of scope here.
                                                  // The logic below will rely on key_found_overall and step counts.
        if (key_found_overall) {
            if (steps_processed_on_this_gpu_task > 0 && steps_processed_on_this_gpu_task < task.actual_num_steps) {
                 // This implies the key was likely found by *this* task, or another task finished simultaneously
                 log_message(LOG_INFO, "GPU ID " + std::to_string(task.gpu_id) + " (" + task.props.name + "): Processing stopped part-way as key was found.");
            } else if (steps_processed_on_this_gpu_task == 0 && task.actual_num_steps > 0) {
                 log_message(LOG_INFO, "GPU ID " + std::to_string(task.gpu_id) + " (" + task.props.name + "): Processing skipped as key was already found.");
            }
            // If key_found_overall is true, but this task completed its full range, the "Processed X steps" log above is sufficient.
        } else if (steps_processed_on_this_gpu_task == task.actual_num_steps && task.actual_num_steps > 0) {
             log_message(LOG_INFO, "GPU ID " + std::to_string(task.gpu_id) + " (" + task.props.name + "): Finished processing its assigned " + std::to_string(task.actual_num_steps) + " steps.");
        }
        // Other cases (e.g. task.actual_num_steps == 0) are logged when task is initially skipped or if VRAM issues occurred.
        // Errors during memory alloc/transfer/sync are logged at point of error.
    } // End of outer for loop (GPU tasks)

    auto app_end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(app_end_time - app_start_time);

    // Overall Speed Summary
    unsigned long long net_steps_processed_this_run = total_giant_steps_processed_overall - global_resume_offset;
    if (duration.count() > 0 && net_steps_processed_this_run > 0) {
        double total_duration_sec = duration.count() / 1000.0;
        if (total_duration_sec > 0.001) { // Avoid division by zero for very short runs
            double overall_steps_per_sec = static_cast<double>(net_steps_processed_this_run) / total_duration_sec;
            log_message(LOG_INFO, "Overall performance: " + std::to_string(overall_steps_per_sec) + " steps/sec (" +
                                   std::to_string(net_steps_processed_this_run) + " net steps processed in " + std::to_string(total_duration_sec) + "s).");
        } else {
             log_message(LOG_INFO, "Overall " + std::to_string(net_steps_processed_this_run) + " net steps processed. Duration too short for meaningful speed calculation.");
        }
    } else if (net_steps_processed_this_run > 0) {
        log_message(LOG_INFO, "Overall " + std::to_string(net_steps_processed_this_run) + " net steps processed. Duration was zero or too short for speed calculation.");
    }

    if (key_found_overall) {
        log_message(LOG_INFO, "Key successfully found!");
        log_message(LOG_INFO, "Found Private Key (hex): " + bytes_to_hex(found_key_data, 32));
    } else {
        if (total_giant_steps_processed_overall >= config.giant_steps_total_range) {
             log_message(LOG_INFO, "Key not found after searching the entire configured range (" + std::to_string(config.giant_steps_total_range) + " steps).");
        } else {
             log_message(LOG_INFO, "Key not found. Processing stopped before completing the entire range. Steps processed: " + std::to_string(total_giant_steps_processed_overall) + "/" + std::to_string(config.giant_steps_total_range));
        }
    }
    // Final checkpoint save, especially if the loop finished without a final save trigger
    if (total_giant_steps_processed_overall > cp_state.current_giant_step_offset) {
         cp_state.current_giant_step_offset = total_giant_steps_processed_overall;
         if (save_checkpoint(cp_state, config.checkpoint_file)) {
            log_message(LOG_INFO, "Final checkpoint saved. Progress: " + std::to_string(cp_state.current_giant_step_offset) + " / " + std::to_string(config.giant_steps_total_range));
         }
    }

    log_message(LOG_INFO, "Total application run time: " + std::to_string(duration.count() / 1000.0) + " seconds.");

    #ifdef WITH_NVML
    if (nvml_initialized_successfully) {
        nvmlReturn_t nvml_shutdown_result = nvmlShutdown();
        if (nvml_shutdown_result != NVML_SUCCESS) {
            log_message(LOG_WARN, "Failed to shutdown NVML. Error: " + std::string(nvmlErrorString(nvml_shutdown_result)));
        } else {
            log_message(LOG_INFO, "NVML shut down successfully.");
        }
    }
    #endif
    log_message(LOG_INFO, "BSGS process completed.");
    return 0;
}
void launch_bsgs_kernel_stub(int,const unsigned char*,unsigned long long,unsigned long long,void*,size_t,bool*,unsigned char*,int&launch_counter){launch_counter++;}
