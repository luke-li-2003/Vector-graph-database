#pragma once
#include "vgraph/database.hpp"
namespace vgraph {
struct HnswConfig;
struct HybridResult {
    std::vector<SearchHit> hits;
    std::size_t eligible_authors = 0;
    std::size_t eligible_posts = 0;
    std::size_t witnesses = 0;
    double graph_ms = 0;
    double merge_ms = 0;
    double vector_ms = 0;
};
// Evaluate graph first, materialize endpoint bitmap, then filter during search.
// The backend may navigate through ineligible posts, but may not return them.
HybridResult hybrid_search(const Database&, const VectorSearch&, PersonId anchor,
                           std::size_t hops, const std::vector<float>& query, std::size_t k);
// Evaluate author filter, overlay witness clique, then HNSW search within the union.
// Requires author metadata built with Database::build_author_hnsw().
HybridResult merged_hnsw_search(const Database&, PersonId anchor, std::size_t hops,
                               const std::vector<float>& query, std::size_t k,
                               const HnswConfig&);
}
