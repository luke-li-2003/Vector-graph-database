#pragma once
#include "vgraph/database.hpp"
#include <memory>
namespace vgraph {
struct HnswConfig {
    std::size_t m = 16;
    std::size_t ef_construction = 200;
    std::size_t ef_search = 100;
    std::uint32_t seed = 42;
};
// Optional construction checkpoints: (inserted posts, total posts), beginning at
// zero and ending at total. Called synchronously; exceptions abort construction.
using HnswBuildProgress = std::function<void(std::size_t, std::size_t)>;
// Immutable index snapshot. Source store must remain alive at the same address,
// without assignment or mutation. Appending posts is detected and requires rebuild.
class HnswSearch final : public VectorSearch {
public:
    explicit HnswSearch(const VectorStore&, HnswConfig = {}, const HnswBuildProgress& = {});
    ~HnswSearch() override;
    HnswSearch(const HnswSearch&) = delete;
    HnswSearch& operator=(const HnswSearch&) = delete;
    std::vector<SearchHit> search(const VectorStore&, const std::vector<float>&,
                                 std::size_t k, const PostFilter& filter = {}) const override;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Compact immutable topology. Post IDs refer to the database's vector store;
// no source pointers, vector copies, or hnswlib construction locks are retained.
class AuthorHnsw final {
public:
    ~AuthorHnsw();
    AuthorHnsw(const AuthorHnsw&) = delete;
    AuthorHnsw& operator=(const AuthorHnsw&) = delete;
    const std::vector<PostId>& posts() const;
    int max_level() const; // -1 for an empty author
    std::vector<PostId> top_layer() const;
private:
    friend class Database;
    friend class MergedHnswSearch;
    AuthorHnsw(const VectorStore&, const std::vector<PostId>&, HnswConfig);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct HnswWitness {
    PersonId author;
    PostId post;
    int level;
};

// Query-local union of selected author indexes. One random top-layer witness
// per nonempty author is connected to every other witness at all shared levels,
// including level 0. Edges are an overlay; author metadata remains unchanged.
// Build metadata first with Database::build_author_hnsw(). Only ef_search and
// seed are used here (seed controls reproducible witness selection).
// Database must stay alive, at the same address and unmodified except knows edges.
class MergedHnswSearch final {
public:
    MergedHnswSearch(const Database&, const std::vector<PersonId>& authors, HnswConfig = {});
    ~MergedHnswSearch();
    MergedHnswSearch(const MergedHnswSearch&) = delete;
    MergedHnswSearch& operator=(const MergedHnswSearch&) = delete;
    std::size_t size() const;
    const std::vector<HnswWitness>& witnesses() const;
    // Inspect local plus overlay edges. Throws for unselected posts or absent layers.
    std::vector<PostId> neighbors(PostId, int level = 0) const;
    std::vector<SearchHit> search(const std::vector<float>& query, std::size_t k) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
