#include "vgraph/hybrid.hpp"
#include <algorithm>
#include <chrono>
namespace vgraph {
HybridResult hybrid_search(const Database& db, const VectorSearch& backend, PersonId anchor,
                           std::size_t hops, const std::vector<float>& query, std::size_t k) {
    db.vectors().validate(query);
    auto start=std::chrono::steady_clock::now();
    auto authors=db.reachable(anchor,hops);
    auto posts=db.posts_by(authors);
    std::vector<bool> eligible(db.post_count(),false);
    for (auto p:posts) eligible[p.value]=true;
    auto graph_end=std::chrono::steady_clock::now();
    HybridResult result;
    result.eligible_authors=authors.size(); result.eligible_posts=posts.size();
    if (k && !posts.empty()) {
        PostFilter filter=[&eligible](PostId p){return p.value<eligible.size() && eligible[p.value];};
        result.hits=backend.search(db.vectors(),query,std::min(k,posts.size()),filter);
    }
    auto end=std::chrono::steady_clock::now();
    result.graph_ms=std::chrono::duration<double,std::milli>(graph_end-start).count();
    result.vector_ms=std::chrono::duration<double,std::milli>(end-graph_end).count();
    return result;
}
}
