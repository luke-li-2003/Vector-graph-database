#include "vgraph/hnsw.hpp"
// Disable upstream SIMD prefetch paths: they read one neighbor past a full
// adjacency list in v0.8.0. Our double-precision distance is scalar already.
#define NO_MANUAL_VECTORIZATION
#include <hnswlib/hnswlib.h>
#include <algorithm>
#include <cstring>
#include <stdexcept>
namespace vgraph {
namespace {
// HNSW's packed vector rows may be unaligned. memcpy reads are portable.
// Double accumulation matches ExactL2Search and avoids float-distance overflow.
class DoubleL2Space final : public hnswlib::SpaceInterface<double> {
    std::size_t dimension_;
    static double distance(const void* a, const void* b, const void* param) {
        const auto dim = *static_cast<const std::size_t*>(param);
        double sum = 0;
        for (std::size_t d=0; d<dim; ++d) {
            float x, y;
            std::memcpy(&x, static_cast<const char*>(a)+d*sizeof(float), sizeof(float));
            std::memcpy(&y, static_cast<const char*>(b)+d*sizeof(float), sizeof(float));
            double delta=static_cast<double>(x)-y; sum+=delta*delta;
        }
        return sum;
    }
public:
    explicit DoubleL2Space(std::size_t dim) : dimension_(dim) {}
    std::size_t get_data_size() override { return dimension_*sizeof(float); }
    hnswlib::DISTFUNC<double> get_dist_func() override { return distance; }
    void* get_dist_func_param() override { return &dimension_; }
};
class Filter final : public hnswlib::BaseFilterFunctor {
    const PostFilter& filter_;
public:
    explicit Filter(const PostFilter& f) : filter_(f) {}
    bool operator()(hnswlib::labeltype id) override { return filter_(PostId{static_cast<std::uint32_t>(id)}); }
};
}
struct HnswSearch::Impl {
    const VectorStore* source;
    std::size_t count;
    DoubleL2Space space;
    std::unique_ptr<hnswlib::HierarchicalNSW<double>> index;
    Impl(const VectorStore& store, HnswConfig c) : source(&store), count(store.size()), space(store.dimension()) {
        if (c.m<2 || c.m>10000 || !c.ef_construction || !c.ef_search)
            throw std::invalid_argument("HNSW requires M in 2..10000 and positive ef values");
        index=std::make_unique<hnswlib::HierarchicalNSW<double>>(&space, std::max<std::size_t>(1,count), c.m, c.ef_construction, c.seed);
        for (std::size_t i=0;i<count;++i) index->addPoint(store.row(PostId{static_cast<std::uint32_t>(i)}),i);
        index->setEf(c.ef_search);
    }
};
HnswSearch::HnswSearch(const VectorStore& store, HnswConfig c) : impl_(std::make_unique<Impl>(store,c)) {}
HnswSearch::~HnswSearch() = default;
std::vector<SearchHit> HnswSearch::search(const VectorStore& store, const std::vector<float>& query,
                                         std::size_t k, const PostFilter& filter) const {
    if (&store!=impl_->source || store.size()!=impl_->count)
        throw std::invalid_argument("HNSW source changed; rebuild the index");
    store.validate(query);
    if (!k || !store.size()) return {};
    Filter adapter(filter);
    auto queue=impl_->index->searchKnn(query.data(), std::min(k,store.size()), filter ? &adapter : nullptr);
    std::vector<SearchHit> hits;
    while (!queue.empty()) {
        hits.push_back({PostId{static_cast<std::uint32_t>(queue.top().second)},queue.top().first}); queue.pop();
    }
    std::sort(hits.begin(),hits.end(),[](const auto& a,const auto& b){
        return a.squared_l2<b.squared_l2 || (a.squared_l2==b.squared_l2 && a.post<b.post);
    });
    return hits;
}
}
