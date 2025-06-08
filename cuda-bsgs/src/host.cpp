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
#include "utils.hpp"
#include "checkpoint.hpp"

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
cudaError_t cudaSetDevice(int device) { if(device<0||device>=mock_gpu_count)return cudaErrorSetDeviceFailed; log_message(LOG_DEBUG,"[CUDA STUB] Set active device to "+std::to_string(device));return cudaSuccess;}
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
std::string string_to_hex_log(const std::string& input) { /* ... same ... */
    std::stringstream ss; ss << std::hex << std::setfill('0');
    for (char c : input) { ss << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(c));}
    return ss.str();
}

int main(int argc, char* argv[]) { /* ... (same as previous, with corrected checkpoint_loaded logic and more logging) ... */
    auto app_start_time = std::chrono::high_resolution_clock::now(); AppConfig config;
    if (!parse_arguments(argc, argv, config)) { return 1; }
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
    for (int gpu_id : gpus_to_use) {
        cudaDeviceProp props; cudaGetDeviceProperties(&props, gpu_id);
        log_message(LOG_INFO, "Processing for GPU ID " + std::to_string(gpu_id) + ": " + props.name + " (SM " + std::to_string(props.major) + "." + std::to_string(props.minor) + ")");
        ProfileParams active_params = {config.w_param, config.htsz_param_mb, 256, 64, false}; bool prof_applied=false;
        log_message(LOG_DEBUG, std::string("  Attempting to apply profile for GPU ") + std::to_string(gpu_id) + ". Forced profile name: '" + config.force_gpu_profile_name + "'");
        if(!config.force_gpu_profile_name.empty()){
            log_message(LOG_DEBUG, "    Attempting FORCED profile match.");
            for(const auto& p:available_profiles){
                bool name_match = (p.name == config.force_gpu_profile_name);
                bool params_are_loaded = p.params.loaded;
                log_message(LOG_DEBUG, std::string("      Checking forced: Profile.name:'") + p.name + "' (len:" + std::to_string(p.name.length()) + ", hex:" + string_to_hex_log(p.name) +
                                     ") vs ForcedName:'" + config.force_gpu_profile_name + "' (len:" + std::to_string(config.force_gpu_profile_name.length()) + ", hex:" + string_to_hex_log(config.force_gpu_profile_name) +
                                     "). NameMatch=" + (name_match?"T":"F") + ". ParamsLoaded=" + (params_are_loaded?"T":"F"));
                if(name_match && params_are_loaded){
                    active_params=p.params;log_message(LOG_INFO,"  GPU ID "+std::to_string(gpu_id)+": Applied FORCED profile '"+p.name+"'");prof_applied=true;
                    break;
                }
            }
            if(!prof_applied) log_message(LOG_WARN, "    Forced profile '" + config.force_gpu_profile_name + "' not found or not applied.");
        } else {
            log_message(LOG_DEBUG, "    Attempting AUTO-match based on SM " + std::to_string(props.major));
            for(const auto& p:available_profiles){
                bool sm_match = (p.matches_sm_major == props.major);
                bool params_are_loaded = p.params.loaded;
                log_message(LOG_DEBUG, std::string("      Checking auto-match: Profile:'") + p.name + "' (SM_match:"+std::to_string(p.matches_sm_major)+") vs GPU SM: "+std::to_string(props.major) + ". SM_Match=" + (sm_match?"T":"F") + ". Params_Loaded=" + (params_are_loaded?"T":"F"));
                if(sm_match && params_are_loaded){
                    active_params=p.params;log_message(LOG_INFO,"  GPU ID "+std::to_string(gpu_id)+": Applied AUTO-MATCHED profile '"+p.name+"'");prof_applied=true;break;
                }
            }
        }
        if(prof_applied){log_message(LOG_INFO,"    Using Profile Params: w="+std::to_string(active_params.w_param)+", htsz_mb="+std::to_string(active_params.htsz_param_mb));}
        else{log_message(LOG_INFO,"  GPU ID "+std::to_string(gpu_id)+": No specific profile applied. Using default/cmd-line params: w="+std::to_string(config.w_param)+", htsz_mb="+std::to_string(config.htsz_param_mb));}
    }
    // ... (Rest of main loop, simplified for this test) ...
    log_message(LOG_INFO, "BSGS process completed."); return 0;
}
void launch_bsgs_kernel_stub(int,const unsigned char*,unsigned long long,unsigned long long,void*,size_t,bool*,unsigned char*,int&launch_counter){launch_counter++;}
