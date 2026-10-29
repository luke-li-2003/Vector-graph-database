#include "vgraph/synthetic.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_set>
namespace vgraph {
Database generate_synthetic(const SyntheticConfig& c) {
    const auto max_ids = std::numeric_limits<std::uint32_t>::max();
    if (c.people >= max_ids || c.posts >= max_ids) throw std::invalid_argument("synthetic count exceeds ID capacity");
    if (!c.people && c.posts) throw std::invalid_argument("posts require at least one person");
    if (!std::isfinite(c.noise) || c.noise < 0) throw std::invalid_argument("noise must be finite and nonnegative");
    if (c.model != EmbeddingModel::Uniform && c.model != EmbeddingModel::AuthorClustered)
        throw std::invalid_argument("unknown embedding model");
    Database db(c.dimension);
    // Independent streams keep graph, author assignment, and embedding choices separate.
    std::mt19937 graph_rng(c.seed), author_rng(c.seed ^ 0x9e3779b9u), vector_rng(c.seed ^ 0x85ebca6bu);
    for (std::size_t i = 0; i < c.people; ++i) {
        db.add_person("Person " + std::to_string(i));
    }
    const auto degree = c.people ? std::min(c.knows_per_person, c.people - 1) : 0;
    for (std::size_t i = 0; i < c.people && degree; ++i) {
        // Floyd sampling selects degree unique indices from [0, people-2] without
        // rejection loops that become slow near full degree. Skip the source ID.
        std::unordered_set<std::uint32_t> selected;
        selected.reserve(degree);
        for (std::size_t j = c.people - 1 - degree; j < c.people - 1; ++j) {
            auto t = std::uniform_int_distribution<std::uint32_t>(0, static_cast<std::uint32_t>(j))(graph_rng);
            if (!selected.insert(t).second) selected.insert(static_cast<std::uint32_t>(j));
        }
        std::vector<std::uint32_t> neighbors(selected.begin(), selected.end());
        std::sort(neighbors.begin(), neighbors.end());
        for (auto n : neighbors) db.add_knows(PersonId{static_cast<std::uint32_t>(i)}, PersonId{n >= i ? n + 1 : n});
    }
    std::uniform_real_distribution<float> uniform(0, 1);
    std::normal_distribution<float> normal(0, 1);
    std::vector<float> centers;
    if (c.model == EmbeddingModel::AuthorClustered && c.posts) {
        if (c.people > centers.max_size() / c.dimension) throw std::length_error("author centers too large");
        centers.resize(c.people * c.dimension);
        for (auto& x : centers) x = normal(vector_rng);
    }
    std::vector<float> embedding(c.dimension);
    for (std::size_t i = 0; i < c.posts; ++i) {
        auto author = std::uniform_int_distribution<std::uint32_t>(0, static_cast<std::uint32_t>(c.people - 1))(author_rng);
        for (std::size_t d = 0; d < c.dimension; ++d)
            embedding[d] = c.model == EmbeddingModel::Uniform ? uniform(vector_rng)
                : centers[static_cast<std::size_t>(author) * c.dimension + d] + c.noise * normal(vector_rng);
        db.add_post(PersonId{author}, "Post " + std::to_string(i), embedding);
    }
    return db;
}
}
