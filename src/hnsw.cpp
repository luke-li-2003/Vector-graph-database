#include "vgraph/hnsw.hpp"
// Disable upstream SIMD prefetch paths: they read one neighbor past a full
// adjacency list in v0.8.0. Our double-precision distance is scalar already.
#define NO_MANUAL_VECTORIZATION
#include <hnswlib/hnswlib.h>
#include <algorithm>
#include <cstring>
#include <queue>
#include <random>
#include <stdexcept>
#include <unordered_set>
namespace vgraph {
namespace {
void validate_config(const HnswConfig& c) {
    if (c.m<2 || c.m>10000 || !c.ef_construction || !c.ef_search)
        throw std::invalid_argument("HNSW requires M in 2..10000 and positive ef values");
}
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
    Impl(const VectorStore& store, HnswConfig c, const HnswBuildProgress& progress)
        : source(&store), count(store.size()), space(store.dimension()) {
        validate_config(c);
        if (progress) progress(0,count);
        index=std::make_unique<hnswlib::HierarchicalNSW<double>>(&space, std::max<std::size_t>(1,count), c.m, c.ef_construction, c.seed);
        for (std::size_t i=0;i<count;++i) {
            index->addPoint(store.row(PostId{static_cast<std::uint32_t>(i)}),i);
            if (progress && ((i+1)%1000==0 || i+1==count)) progress(i+1,count);
        }
        index->setEf(c.ef_search);
    }
};
HnswSearch::HnswSearch(const VectorStore& store, HnswConfig c, const HnswBuildProgress& progress)
    : impl_(std::make_unique<Impl>(store,c,progress)) {}
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

struct AuthorHnsw::Impl {
    std::vector<PostId> posts;
    std::vector<int> levels;
    std::vector<std::size_t> first_layer, offsets;
    std::vector<hnswlib::tableint> links, top;
    int max_level = -1;

    Impl(const VectorStore& store, const std::vector<PostId>& ids, HnswConfig c) : posts(ids) {
        validate_config(c);
        if (posts.empty()) return;
        DoubleL2Space space(store.dimension());
        // Construct one shard at a time, then discard hnswlib's per-index lock
        // tables and vector copies. Keeping them per author would cost MiBs each.
        hnswlib::HierarchicalNSW<double> index(&space, posts.size(), c.m, c.ef_construction, c.seed);
        for (auto post:posts) index.addPoint(store.row(post), post.value);
        max_level=index.maxlevel_;
        levels.reserve(posts.size()); first_layer.reserve(posts.size());
        for (std::size_t i=0; i<posts.size(); ++i) {
            auto local=static_cast<hnswlib::tableint>(i);
            int height=index.element_levels_[i];
            levels.push_back(height); first_layer.push_back(offsets.size());
            if (height==max_level) top.push_back(local);
            for (int level=0; level<=height; ++level) {
                offsets.push_back(links.size());
                auto list=index.get_linklist_at_level(local,level);
                auto count=index.getListCount(list);
                auto neighbors=reinterpret_cast<const hnswlib::tableint*>(list+1);
                links.insert(links.end(),neighbors,neighbors+count);
            }
        }
        offsets.push_back(links.size());
    }
};
AuthorHnsw::AuthorHnsw(const VectorStore& store, const std::vector<PostId>& posts, HnswConfig c)
    : impl_(std::make_unique<Impl>(store,posts,c)) {}
AuthorHnsw::~AuthorHnsw() = default;
const std::vector<PostId>& AuthorHnsw::posts() const { return impl_->posts; }
int AuthorHnsw::max_level() const { return impl_->max_level; }
std::vector<PostId> AuthorHnsw::top_layer() const {
    std::vector<PostId> result;
    for (auto local:impl_->top) result.push_back(impl_->posts[local]);
    return result;
}

void Database::build_author_hnsw() { build_author_hnsw(HnswConfig{}); }
void Database::build_author_hnsw(const HnswConfig& c) {
    validate_config(c);
    std::vector<std::shared_ptr<const AuthorHnsw>> indexes;
    indexes.reserve(people_.size());
    for (std::size_t i=0; i<people_.size(); ++i) {
        auto config=c;
        // Independent author streams avoid identical height patterns in every shard.
        std::seed_seq seed{c.seed,static_cast<std::uint32_t>(i)};
        std::mt19937 rng(seed); config.seed=rng();
        indexes.emplace_back(new AuthorHnsw(vectors_,authored_[i],config));
    }
    // Publish only after every shard has been built successfully.
    for (std::size_t i=0; i<people_.size(); ++i) people_[i].hnsw=std::move(indexes[i]);
}

struct MergedHnswSearch::Impl {
    struct Node { std::size_t shard; hnswlib::tableint local; };
    struct Shard {
        std::shared_ptr<const AuthorHnsw> index;
        hnswlib::tableint witness;
    };
    struct Candidate { double distance; Node node; PostId post; };
    struct Better {
        bool operator()(const Candidate& a, const Candidate& b) const {
            return a.distance<b.distance || (a.distance==b.distance && a.post<b.post);
        }
    };
    struct Worse {
        bool operator()(const Candidate& a, const Candidate& b) const { return Better{}(b,a); }
    };
    const Database* source;
    std::size_t source_count, count = 0, ef;
    std::vector<Shard> shards;
    std::vector<HnswWitness> witnesses;
    Node entry{0,0};
    int max_level = -1;

    Impl(const Database& db, std::vector<PersonId> authors, HnswConfig c)
        : source(&db), source_count(db.post_count()), ef(c.ef_search) {
        validate_config(c);
        std::sort(authors.begin(),authors.end());
        authors.erase(std::unique(authors.begin(),authors.end()),authors.end());
        std::mt19937 rng(c.seed);
        for (auto author:authors) {
            auto index=db.person(author).hnsw;
            if (!index || index->posts().size()!=db.authored(author).size())
                throw std::invalid_argument("author HNSW missing or stale; call build_author_hnsw");
            const auto& topology=*index->impl_;
            if (topology.posts.empty()) continue;
            std::uniform_int_distribution<std::size_t> pick(0,topology.top.size()-1);
            auto witness=topology.top[pick(rng)];
            if (topology.max_level>max_level) {
                max_level=topology.max_level;
                entry={shards.size(),witness};
            }
            witnesses.push_back({author,topology.posts[witness],topology.max_level});
            count+=topology.posts.size();
            shards.push_back({std::move(index),witness});
        }
    }
    void validate_source() const {
        if (source->post_count()!=source_count)
            throw std::invalid_argument("merged HNSW source changed; rebuild the merge");
    }
    const AuthorHnsw::Impl& topology(Node node) const { return *shards[node.shard].index->impl_; }
    PostId post(Node node) const { return topology(node).posts[node.local]; }
    template<class F> void for_neighbors(Node node, int level, F visit) const {
        const auto& graph=topology(node);
        auto layer=graph.first_layer[node.local]+static_cast<std::size_t>(level);
        for (auto i=graph.offsets[layer]; i<graph.offsets[layer+1]; ++i)
            visit(Node{node.shard,graph.links[i]});
        if (node.local==shards[node.shard].witness) {
            // This enumerates an undirected complete graph without allocating
            // O(authors^2) edges or pruning them to the local HNSW degree bound.
            for (std::size_t s=0; s<shards.size(); ++s)
                if (s!=node.shard && witnesses[s].level>=level)
                    visit(Node{s,shards[s].witness});
        }
    }
    Candidate candidate(Node node, const std::vector<float>& query) const {
        auto id=post(node);
        const auto* row=source->vectors().row(id);
        double distance=0;
        for (std::size_t d=0; d<query.size(); ++d) {
            double delta=static_cast<double>(query[d])-row[d]; distance+=delta*delta;
        }
        return {distance,node,id};
    }
    std::vector<SearchHit> search(const std::vector<float>& query, std::size_t k) const {
        validate_source(); source->vectors().validate(query);
        if (!k || !count) return {};
        k=std::min(k,count);
        auto breadth=std::min(count,std::max(k,ef));
        auto current=candidate(entry,query);
        // Standard HNSW greedy descent through the merged upper layers.
        for (int level=max_level; level>0; --level) {
            bool changed=true;
            while (changed) {
                changed=false;
                for_neighbors(current.node,level,[&](Node node) {
                    auto next=candidate(node,query);
                    if (next.distance<current.distance) { current=next; changed=true; }
                });
            }
        }
        // Standard ef-bounded best-first exploration at layer 0, starting from
        // the single descended entry. No per-author search or exact fallback.
        std::priority_queue<Candidate,std::vector<Candidate>,Worse> pending;
        std::priority_queue<Candidate,std::vector<Candidate>,Better> best;
        std::unordered_set<std::uint32_t> visited;
        pending.push(current); best.push(current); visited.insert(current.post.value);
        while (!pending.empty()) {
            current=pending.top(); pending.pop();
            if (best.size()==breadth && current.distance>best.top().distance) break;
            for_neighbors(current.node,0,[&](Node node) {
                if (!visited.insert(post(node).value).second) return;
                auto next=candidate(node,query);
                if (best.size()<breadth || Better{}(next,best.top())) {
                    pending.push(next); best.push(next);
                    if (best.size()>breadth) best.pop();
                }
            });
        }
        while (best.size()>k) best.pop();
        std::vector<SearchHit> hits;
        while (!best.empty()) {
            hits.push_back({best.top().post,best.top().distance}); best.pop();
        }
        std::reverse(hits.begin(),hits.end());
        return hits;
    }
};
MergedHnswSearch::MergedHnswSearch(const Database& db, const std::vector<PersonId>& authors, HnswConfig c)
    : impl_(std::make_unique<Impl>(db,authors,c)) {}
MergedHnswSearch::~MergedHnswSearch() = default;
std::size_t MergedHnswSearch::size() const { return impl_->count; }
const std::vector<HnswWitness>& MergedHnswSearch::witnesses() const { return impl_->witnesses; }
std::vector<PostId> MergedHnswSearch::neighbors(PostId post, int level) const {
    impl_->validate_source();
    for (std::size_t s=0; s<impl_->shards.size(); ++s) {
        const auto& graph=*impl_->shards[s].index->impl_;
        auto it=std::lower_bound(graph.posts.begin(),graph.posts.end(),post);
        if (it==graph.posts.end() || !(*it==post)) continue;
        auto local=static_cast<hnswlib::tableint>(it-graph.posts.begin());
        if (level<0 || level>graph.levels[local]) throw std::out_of_range("post absent at HNSW layer");
        std::vector<PostId> result;
        impl_->for_neighbors({s,local},level,[&](Impl::Node node){result.push_back(impl_->post(node));});
        std::sort(result.begin(),result.end());
        return result;
    }
    throw std::out_of_range("post not in merged HNSW");
}
std::vector<SearchHit> MergedHnswSearch::search(const std::vector<float>& query, std::size_t k) const {
    return impl_->search(query,k);
}
}
