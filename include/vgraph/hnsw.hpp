#pragma once
#include "vgraph/vector.hpp"
#include <memory>
namespace vgraph {
struct HnswConfig {
    std::size_t m = 16;
    std::size_t ef_construction = 200;
    std::size_t ef_search = 100;
    std::uint32_t seed = 42;
};
// Immutable index snapshot. Source store must remain alive at the same address,
// without assignment or mutation. Appending posts is detected and requires rebuild.
class HnswSearch final : public VectorSearch {
public:
    explicit HnswSearch(const VectorStore&, HnswConfig = {});
    ~HnswSearch() override;
    HnswSearch(const HnswSearch&) = delete;
    HnswSearch& operator=(const HnswSearch&) = delete;
    std::vector<SearchHit> search(const VectorStore&, const std::vector<float>&,
                                 std::size_t k, const PostFilter& filter = {}) const override;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
