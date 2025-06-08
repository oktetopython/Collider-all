#ifndef UTILS_HPP
#define UTILS_HPP

#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <iomanip>

inline bool hex_to_bytes(const std::string& hex_key, unsigned char* out_key, size_t key_len) {
    if (hex_key.length() != key_len * 2) {
        std::cerr << "Error: Public key hex string has incorrect length. Expected " << key_len * 2 << ", got " << hex_key.length() << std::endl;
        return false;
    }
    for (size_t i = 0; i < key_len; ++i) {
        try {
            std::string byteString = hex_key.substr(i * 2, 2);
            out_key[i] = static_cast<unsigned char>(std::stoul(byteString, nullptr, 16));
        } catch (const std::invalid_argument& ia) {
            std::cerr << "Error: Invalid hex character in public key: '" << hex_key.substr(i*2,2) << "'. " << ia.what() << std::endl;
            return false;
        } catch (const std::out_of_range& oor) {
            std::cerr << "Error: Hex value out of range in public key: '" << hex_key.substr(i*2,2) << "'. " << oor.what() << std::endl;
            return false;
        }
    }
    return true;
}

inline std::string get_utility_message() {
    return "Utility message from utils.hpp!";
}

#endif // UTILS_HPP
