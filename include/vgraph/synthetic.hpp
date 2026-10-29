#pragma once
#include "vgraph/database.hpp"
namespace vgraph {
enum class EmbeddingModel { Uniform, AuthorClustered };
struct SyntheticConfig {
    std::size_t people = 1000;
    std::size_t posts = 100000;
    std::size_t dimension = 128;
    std::size_t knows_per_person = 20;
    std::uint32_t seed = 42;
    EmbeddingModel model = EmbeddingModel::Uniform;
    float noise = 0.3f; // Standard deviation for AuthorClustered only.
};
// Directed, unique, non-self knows edges. Degree capped at people-1.
// Authors are uniform; embeddings are uniform [0,1) or author center + Gaussian noise.
Database generate_synthetic(const SyntheticConfig& config = {});
}
