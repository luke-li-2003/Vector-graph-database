#pragma once
#include "vgraph/database.hpp"
namespace vgraph {
struct HybridResult {
    std::vector<SearchHit> hits;
    std::size_t eligible_authors = 0;
    std::size_t eligible_posts = 0;
    double graph_ms = 0;
    double vector_ms = 0;
};
// Evaluate graph first, materialize endpoint bitmap, then filter during search.
// The backend may navigate through ineligible posts, but may not return them.
HybridResult hybrid_search(const Database&, const VectorSearch&, PersonId anchor,
                           std::size_t hops, const std::vector<float>& query, std::size_t k);
}
