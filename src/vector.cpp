#include "vgraph/vector.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
namespace vgraph {
VectorStore::VectorStore(std::size_t dimension) : dimension_(dimension) {
    if (!dimension || dimension > 65536)
        throw std::invalid_argument("dimension must be 1..65536");
}
void VectorStore::validate(const std::vector<float>& v) const {
    if (v.size() != dimension_)
        throw std::invalid_argument("embedding dimension mismatch");
    for (float x : v)
        if (!std::isfinite(x)) throw std::invalid_argument("non-finite embedding");
}
const float* VectorStore::row(PostId id) const {
    if (id.value >= size()) throw std::out_of_range("unknown post vector");
    return values_.data() + static_cast<std::size_t>(id.value) * dimension_;
}
void VectorStore::reserve_one() {
    if (dimension_ > values_.max_size() - values_.size())
        throw std::length_error("vector store full");
    // Geometric growth avoids quadratic insertion while keeping append non-throwing.
    const auto needed = values_.size() + dimension_;
    if (needed > values_.capacity()) {
        const auto grown = values_.capacity() <= values_.max_size()/2 ? values_.capacity()*2 : values_.max_size();
        values_.reserve(std::max(needed, grown));
    }
}
void VectorStore::append(const std::vector<float>& v) {
    values_.insert(values_.end(), v.begin(), v.end());
}

std::vector<SearchHit> ExactL2Search::search(const VectorStore& store,
                                             const std::vector<float>& q,
                                             std::size_t k,
                                             const PostFilter& filter) const {
    store.validate(q);
    auto better = [](const SearchHit& a, const SearchHit& b) {
        return a.squared_l2 < b.squared_l2 || (a.squared_l2 == b.squared_l2 && a.post < b.post);
    };
    std::priority_queue<SearchHit, std::vector<SearchHit>, decltype(better)> heap(better);
    if (!k) return {};
    for (std::size_t i = 0; i < store.size(); ++i) {
        PostId id{static_cast<std::uint32_t>(i)};
        if (filter && !filter(id)) continue;
        const auto* row = store.row(id);
        double distance = 0;
        for (std::size_t d = 0; d < q.size(); ++d) {
            double delta = static_cast<double>(q[d]) - row[d];
            distance += delta * delta;
        }
        SearchHit hit{id, distance};
        if (heap.size() < k) heap.push(hit);
        else if (better(hit, heap.top())) { heap.pop(); heap.push(hit); }
    }
    std::vector<SearchHit> hits;
    while (!heap.empty()) { hits.push_back(heap.top()); heap.pop(); }
    std::reverse(hits.begin(), hits.end());
    return hits;
}
}
