#pragma once
#include <cstdint>
#include <cstddef>
#include <functional>
#include <vector>

namespace vgraph {
struct PostId {
    std::uint32_t value;
    friend bool operator==(PostId a, PostId b) { return a.value == b.value; }
    friend bool operator<(PostId a, PostId b) { return a.value < b.value; }
};
// Row-major storage; row number is the stable PostId. No index topology here.
class VectorStore {
public:
    explicit VectorStore(std::size_t dimension);
    std::size_t dimension() const { return dimension_; }
    std::size_t size() const { return values_.size() / dimension_; }
    const float* row(PostId id) const;
    void validate(const std::vector<float>& vector) const;
private:
    friend class Database;
    void reserve_one();
    void append(const std::vector<float>& vector);
    std::size_t dimension_;
    std::vector<float> values_;
};
struct SearchHit { PostId post; double squared_l2; };
using PostFilter = std::function<bool(PostId)>;
// Exact and ANN implementations consume the same read-only storage and filter.
class VectorSearch {
public:
    virtual ~VectorSearch() = default;
    virtual std::vector<SearchHit> search(const VectorStore&, const std::vector<float>& query,
                                         std::size_t k, const PostFilter& filter = {}) const = 0;
};
class ExactL2Search final : public VectorSearch {
public:
    std::vector<SearchHit> search(const VectorStore&, const std::vector<float>&,
                                 std::size_t k, const PostFilter& filter = {}) const override;
};
} // namespace vgraph
