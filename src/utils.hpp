//
// Created by robert on 1/25/25.
//

#pragma once
#include <random>
inline float generateRandomFloat(float lower, float upper) {
    // Create a random number generator
    std::random_device rd;   // Seed for random number generator
    std::mt19937 gen(rd());  // Mersenne Twister engine

    // Create a distribution in the given range [lower, upper]
    std::uniform_real_distribution<float> dist(lower, upper);

    // Generate and return a random number
    return dist(gen);
}