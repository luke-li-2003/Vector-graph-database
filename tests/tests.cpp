#include "vgraph/database.hpp"
#include "vgraph/synthetic.hpp"
#include "vgraph/hnsw.hpp"
#include "vgraph/hybrid.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <type_traits>
using namespace vgraph;
#define CHECK(x) do { if (!(x)) throw std::runtime_error("check failed at line " + std::to_string(__LINE__) + ": " #x); } while (false)
template<class F> void rejects(F f) { bool threw = false; try { f(); } catch (const std::exception&) { threw = true; } CHECK(threw); }
void same_hits(const std::vector<SearchHit>& actual, const std::vector<SearchHit>& expected,
               const std::string& context = {}) {
    CHECK(actual.size()==expected.size());
    for (std::size_t i=0; i<actual.size(); ++i) {
        if (!(actual[i].post==expected[i].post) || actual[i].squared_l2!=expected[i].squared_l2)
            throw std::runtime_error(context+" rank="+std::to_string(i)+" actual_post="+
                                     std::to_string(actual[i].post.value)+" expected_post="+
                                     std::to_string(expected[i].post.value));
    }
}
void author_hnsw_tests() {
    Database db(2);
    auto anchor=db.add_person("anchor"), a=db.add_person("many"), b=db.add_person("pair");
    auto singleton=db.add_person("singleton"), empty=db.add_person("empty"), outside=db.add_person("outside");
    db.add_knows(anchor,a); db.add_knows(anchor,b); db.add_knows(anchor,empty);
    db.add_knows(b,singleton); db.add_knows(singleton,anchor);
    auto excluded=db.add_post(outside,"unselected nearest",{0,0});
    // Interleaved global post IDs exercise shard-local to global ID translation.
    for (int i=0; i<80; ++i) {
        db.add_post(a,"a",{float(i+1),float(i%3)});
        if (i<2) db.add_post(b,"b",{-float(i+1),0});
    }
    auto lone=db.add_post(singleton,"one",{0,0.25f});
    rejects([&]{MergedHnswSearch missing(db,{a});});
    HnswConfig config; config.m=2; config.ef_search=1000;
    db.build_author_hnsw(config);
    for (std::uint32_t i=0; i<db.person_count(); ++i) {
        auto author=PersonId{i}; auto index=db.person(author).hnsw;
        CHECK(index && index->posts()==db.authored(author));
        CHECK(index->posts().empty()==(index->max_level()==-1));
    }
    CHECK(db.person(empty).hnsw->top_layer().empty());
    // Deduplicate authors and choose only top-layer posts as witnesses.
    MergedHnswSearch merged(db,{singleton,b,a,a,empty},config);
    CHECK(merged.size()==83 && merged.witnesses().size()==3);
    int highest=-1,lowest=100;
    for (const auto& witness:merged.witnesses()) {
        const auto& index=db.person(witness.author).hnsw;
        auto top=index->top_layer();
        CHECK(std::find(top.begin(),top.end(),witness.post)!=top.end());
        CHECK(witness.level==index->max_level());
        highest=std::max(highest,witness.level); lowest=std::min(lowest,witness.level);
        // Every pair is connected bidirectionally at every shared level.
        for (const auto& other:merged.witnesses()) if (!(witness.post==other.post)) {
            for (int level=0; level<=std::min(witness.level,other.level); ++level) {
                auto neighbors=merged.neighbors(witness.post,level);
                CHECK(std::find(neighbors.begin(),neighbors.end(),other.post)!=neighbors.end());
            }
        }
    }
    CHECK(highest>lowest); // A real upper layer plus level-0-only singleton.
    rejects([&]{merged.neighbors(excluded);});
    rejects([&]{merged.neighbors(lone,db.person(singleton).hnsw->max_level()+1);});
    rejects([&]{merged.neighbors(lone,-1);});
    for (auto author:{a,b,singleton}) for (auto post:db.authored(author)) {
        auto neighbors=merged.neighbors(post);
        CHECK(std::adjacent_find(neighbors.begin(),neighbors.end())==neighbors.end());
        CHECK(std::find(neighbors.begin(),neighbors.end(),post)==neighbors.end());
        for (auto neighbor:neighbors) {
            auto owner=db.post(neighbor).author;
            CHECK(owner==a || owner==b || owner==singleton);
        }
    }
    ExactL2Search exact;
    for (std::size_t k:{0u,1u,3u,100u})
        same_hits(merged.search({0,0},k),exact.search(db.vectors(),{0,0},k,db.author_filter({a,b,singleton})),"author subset");
    for (std::size_t hops:{0u,1u,2u,20u}) for (std::size_t k:{0u,1u,10u,100u}) {
        auto result=merged_hnsw_search(db,anchor,hops,{0,0},k,config);
        auto truth=hybrid_search(db,exact,anchor,hops,{0,0},k);
        CHECK(result.eligible_authors==truth.eligible_authors && result.eligible_posts==truth.eligible_posts);
        same_hits(result.hits,truth.hits,"author hybrid hops="+std::to_string(hops));
    }
    same_hits(MergedHnswSearch(db,{singleton},config).search({0,0},99),{{lone,0.0625}});
    CHECK(MergedHnswSearch(db,{},config).search({0,0},10).empty());
    CHECK(MergedHnswSearch(db,{empty},config).search({0,0},10).empty());
    rejects([&]{merged.search({0},0);});
    rejects([&]{merged.search({NAN,0},1);});
    rejects([&]{MergedHnswSearch(db,{},config).search({0},0);});
    rejects([&]{MergedHnswSearch invalid(db,{PersonId{999}},config);});
    rejects([&]{merged_hnsw_search(db,PersonId{999},1,{0,0},1,config);});

    // Query overlays never add cross-author edges to reusable metadata.
    MergedHnswSearch single(db,{a},config);
    for (auto post:db.authored(a)) for (auto neighbor:single.neighbors(post))
        CHECK(db.post(neighbor).author==a);
    MergedHnswSearch reordered(db,{empty,a,b,singleton},config);
    for (std::size_t i=0; i<merged.witnesses().size(); ++i)
        CHECK(merged.witnesses()[i].post==reordered.witnesses()[i].post);

    // Sampling can select different top-layer vertices, rather than always the entry point.
    Database flat(1); auto f=flat.add_person("flat");
    for (int i=0; i<4; ++i) flat.add_post(f,"",{float(i)});
    HnswConfig flat_config; flat_config.m=100;
    flat.build_author_hnsw(flat_config);
    CHECK(flat.person(f).hnsw->top_layer().size()>1);
    std::vector<PostId> sampled;
    for (std::uint32_t seed=0; seed<32; ++seed) {
        flat_config.seed=seed; MergedHnswSearch pick(flat,{f},flat_config);
        sampled.push_back(pick.witnesses()[0].post);
    }
    std::sort(sampled.begin(),sampled.end());
    CHECK(std::unique(sampled.begin(),sampled.end())-sampled.begin()>1);

    // The witness clique is complete even when its degree exceeds the local M/2M bounds.
    Database clique_db(1); std::vector<PersonId> clique_authors;
    for (int i=0; i<12; ++i) {
        auto person=clique_db.add_person("one"); clique_authors.push_back(person);
        clique_db.add_post(person,"",{float(i)});
    }
    auto clique_config=config; clique_config.ef_search=1;
    clique_db.build_author_hnsw(clique_config);
    MergedHnswSearch clique(clique_db,clique_authors,clique_config);
    for (auto witness:clique.witnesses()) CHECK(clique.neighbors(witness.post).size()==11);
    same_hits(clique.search({9.5f},3),exact.search(clique_db.vectors(),{9.5f},3));

    // Metadata contains only topology and remains safe across database copies/moves.
    auto copy=db;
    auto moved=std::move(copy);
    same_hits(MergedHnswSearch(moved,{a,b,singleton},config).search({0,0},10),merged.search({0,0},10));
    auto old_a=db.person(a).hnsw, old_b=db.person(b).hnsw;
    rejects([&]{db.add_post(a,"invalid",{0});});
    CHECK(db.person(a).hnsw==old_a);
    db.add_knows(anchor,singleton);
    CHECK(merged_hnsw_search(db,anchor,1,{0,0},1,config).hits[0].post==lone);
    auto added=db.add_post(a,"new nearest",{0,0});
    CHECK(!db.person(a).hnsw && db.person(b).hnsw==old_b);
    CHECK(moved.person(a).hnsw==old_a);
    rejects([&]{merged.search({0,0},1);});
    rejects([&]{MergedHnswSearch stale(db,{a},config);});
    CHECK(MergedHnswSearch(db,{b},config).search({0,0},1).size()==1);
    for (int invalid=0; invalid<4; ++invalid) {
        auto bad=config;
        if (invalid==0) bad.m=1;
        if (invalid==1) bad.m=10001;
        if (invalid==2) bad.ef_search=0;
        if (invalid==3) bad.ef_construction=0;
        rejects([&]{db.build_author_hnsw(bad);});
        rejects([&]{MergedHnswSearch invalid_merge(db,{b},bad);});
        CHECK(db.person(b).hnsw==old_b); // Failed rebuild leaves metadata intact.
    }
    db.build_author_hnsw(config);
    CHECK(MergedHnswSearch(db,{a},config).search({0,0},1)[0].post==added);
    Database no_people(1); no_people.build_author_hnsw();
    CHECK(MergedHnswSearch(no_people,{}).search({0},1).empty());
    Database extreme(1); auto x=extreme.add_person("x");
    extreme.add_post(x,"",{std::numeric_limits<float>::max()}); extreme.build_author_hnsw();
    CHECK(std::isfinite(MergedHnswSearch(extreme,{x}).search({-std::numeric_limits<float>::max()},1)[0].squared_l2));

    // Use a normal construction degree for exact agreement on these seeded fixtures.
    // Very sparse HNSWs can have unreachable posts even at arbitrarily high ef.
    config.m=16;
    SyntheticConfig sc; sc.people=30; sc.posts=700; sc.dimension=7; sc.knows_per_person=3;
    std::mt19937 rng(9); std::uniform_real_distribution<float> value(-1,1);
    for (auto model:{EmbeddingModel::Uniform,EmbeddingModel::AuthorClustered}) {
        sc.model=model;
        auto random=generate_synthetic(sc); random.build_author_hnsw(config);
        for (std::uint32_t id=0; id<10; ++id) {
            std::vector<float> q(sc.dimension); for (auto& v:q) v=value(rng);
            auto actual=merged_hnsw_search(random,PersonId{id},2,q,10,config);
            auto truth=hybrid_search(random,exact,PersonId{id},2,q,10);
            same_hits(actual.hits,truth.hits,"random authors model="+std::to_string(int(model))+" anchor="+std::to_string(id));
        }
    }
}
void hnsw_tests() {
    Database db(2); auto a=db.add_person("a"), b=db.add_person("b"), c=db.add_person("c");
    db.add_knows(a,b); db.add_knows(b,c); db.add_knows(c,a);
    // Global nearest post is ineligible: filtering after top-k would miss b's posts.
    db.add_post(a,"ineligible nearest",{0,0});
    auto p=db.add_post(b,"eligible",{1,0});
    db.add_post(b,"eligible second",{2,0}); db.add_post(c,"two hops",{0,0.5f});
    HnswConfig cfg; cfg.ef_search=1000;
    HnswSearch index(db.vectors(),cfg); ExactL2Search exact;
    auto result=hybrid_search(db,index,a,1,{0,0},1);
    CHECK(result.eligible_authors==1 && result.eligible_posts==2 && result.hits.size()==1);
    CHECK(result.hits[0].post==p && result.hits[0].squared_l2==1);
    for (std::size_t hops: {0u,1u,2u,20u}) for(std::size_t k: {0u,1u,2u,99u}) {
        auto got=hybrid_search(db,index,a,hops,{0,0},k), truth=hybrid_search(db,exact,a,hops,{0,0},k);
        CHECK(got.hits.size()==truth.hits.size());
        for(std::size_t i=0;i<got.hits.size();++i)
            CHECK(got.hits[i].post==truth.hits[i].post && got.hits[i].squared_l2==truth.hits[i].squared_l2);
    }
    CHECK(index.search(db.vectors(),{0,0},10,[](PostId){return false;}).empty());
    CHECK(index.search(db.vectors(),{0,0},1)[0].post.value==0);
    rejects([&]{index.search(db.vectors(),{0},0);});
    rejects([&]{index.search(db.vectors(),{NAN,0},1);});
    rejects([&]{hybrid_search(db,index,PersonId{99},1,{0,0},1);});
    Database other(2); rejects([&]{index.search(other.vectors(),{0,0},1);});
    db.add_post(b,"new",{0,0}); rejects([&]{index.search(db.vectors(),{0,0},1);});
    HnswSearch rebuilt(db.vectors(),cfg); CHECK(hybrid_search(db,rebuilt,a,1,{0,0},1).hits[0].post.value==4);
    Database empty(2); std::size_t empty_reports=0;
    HnswSearch empty_index(empty.vectors(),{},[&](std::size_t done,std::size_t total) {
        CHECK(done==0 && total==0); ++empty_reports;
    });
    CHECK(empty_reports==1);
    CHECK(empty_index.search(empty.vectors(),{0,0},10).empty());
    cfg.m=1; rejects([&]{HnswSearch invalid(db.vectors(),cfg);}); cfg.m=16;
    cfg.ef_search=0; rejects([&]{HnswSearch invalid(db.vectors(),cfg);});
    cfg.ef_search=1000; cfg.ef_construction=0; rejects([&]{HnswSearch invalid(db.vectors(),cfg);});
    // Finite extreme vectors retain finite double distances.
    Database extreme(3); auto author=extreme.add_person("x");
    extreme.add_post(author,"x",{std::numeric_limits<float>::max(),0,0});
    HnswSearch extreme_index(extreme.vectors());
    CHECK(std::isfinite(extreme_index.search(extreme.vectors(),{-std::numeric_limits<float>::max(),0,0},1)[0].squared_l2));
    SyntheticConfig sc; sc.people=30; sc.posts=700; sc.dimension=7; sc.knows_per_person=3;
    auto random=generate_synthetic(sc); cfg.ef_construction=200;
    HnswSearch random_index(random.vectors(),cfg);
    std::mt19937 rng(9); std::uniform_real_distribution<float> value(-1,1);
    for(std::uint32_t a_id=0;a_id<10;++a_id) {
        std::vector<float> q(sc.dimension); for(auto& x:q) x=value(rng);
        auto got=hybrid_search(random,random_index,PersonId{a_id},1,q,10);
        auto truth=hybrid_search(random,exact,PersonId{a_id},1,q,10);
        CHECK(got.hits.size()==truth.hits.size());
        for(std::size_t i=0;i<got.hits.size();++i)
            CHECK(got.hits[i].post==truth.hits[i].post && got.hits[i].squared_l2==truth.hits[i].squared_l2);
    }
    // Construction reports actual completed insertions, including intermediate
    // checkpoints on long builds; the callback does not change search results.
    Database progress_db(1); auto progress_author=progress_db.add_person("progress");
    for (int i=0; i<1200; ++i) progress_db.add_post(progress_author,"",{float(i)});
    std::vector<std::size_t> reports;
    HnswSearch progress_index(progress_db.vectors(),{},[&](std::size_t done,std::size_t total) {
        CHECK(total==progress_db.post_count() && done<=total);
        CHECK(reports.empty() || done>reports.back());
        reports.push_back(done);
    });
    CHECK(reports.size()>=3 && reports.front()==0 && reports.back()==progress_db.post_count());
    same_hits(progress_index.search(progress_db.vectors(),{500.25f},3),
              exact.search(progress_db.vectors(),{500.25f},3));
}
void synthetic_tests() {
    SyntheticConfig c;
    c.people=17; c.posts=250; c.dimension=7; c.knows_per_person=5;
    auto db=generate_synthetic(c), repeat=generate_synthetic(c);
    CHECK(db.person_count()==17 && db.post_count()==250 && db.vectors().dimension()==7);
    for (std::uint32_t i=0;i<c.people;++i) {
        auto row=db.knows(PersonId{i});
        CHECK(row.size()==5 && row==repeat.knows(PersonId{i}));
        CHECK(std::find(row.begin(),row.end(),PersonId{i})==row.end());
        CHECK(std::adjacent_find(row.begin(),row.end())==row.end());
    }
    for (std::uint32_t i=0;i<c.posts;++i) {
        CHECK(db.post(PostId{i}).author==repeat.post(PostId{i}).author);
        for (std::size_t d=0;d<c.dimension;++d) {
            auto x=db.vectors().row(PostId{i})[d];
            CHECK(x>=0 && x<1 && x==repeat.vectors().row(PostId{i})[d]);
        }
    }
    c.seed=43; auto other=generate_synthetic(c);
    CHECK(db.vectors().row(PostId{0})[0]!=other.vectors().row(PostId{0})[0]);
    c.model=EmbeddingModel::AuthorClustered; c.noise=0;
    auto clustered=generate_synthetic(c);
    for (std::uint32_t a=0;a<c.people;++a) {
        auto posts=clustered.authored(PersonId{a});
        for (auto p:posts) for (std::size_t d=0;d<c.dimension;++d)
            CHECK(clustered.vectors().row(p)[d]==clustered.vectors().row(posts.front())[d]);
    }
    c.noise=0.3f; auto noisy=generate_synthetic(c);
    CHECK(noisy.vectors().row(PostId{0})[0]!=clustered.vectors().row(PostId{0})[0]);
    c.people=1; c.posts=3; c.knows_per_person=999;
    CHECK(generate_synthetic(c).knows(PersonId{0}).empty());
    c.people=4; auto dense=generate_synthetic(c);
    for (std::uint32_t a=0;a<4;++a) CHECK(dense.knows(PersonId{a}).size()==3);
    c.knows_per_person=0; CHECK(generate_synthetic(c).knows(PersonId{0}).empty());
    c.people=0; rejects([&]{generate_synthetic(c);});
    c.posts=0; CHECK(generate_synthetic(c).person_count()==0);
    c.dimension=0; rejects([&]{generate_synthetic(c);}); c.dimension=7;
    c.noise=-1; rejects([&]{generate_synthetic(c);});
    c.noise=INFINITY; rejects([&]{generate_synthetic(c);}); c.noise=0;
    c.people=std::numeric_limits<std::uint32_t>::max(); rejects([&]{generate_synthetic(c);});
}
void run(const std::filesystem::path& dir) {
    synthetic_tests();
    hnsw_tests();
    author_hnsw_tests();
    static_assert(!std::is_convertible<PersonId, PostId>::value, "IDs must be distinct");
    rejects([] { Database db(0); });
    Database db(2);
    auto a=db.add_person("Alice \"A\"\nname"), b=db.add_person("Bob"), c=db.add_person("Carol"), d=db.add_person("Isolated");
    CHECK(db.add_knows(a,b)); CHECK(!db.add_knows(a,b));
    db.add_knows(b,c); db.add_knows(c,a); db.add_knows(a,a);
    CHECK(db.knows(b).size()==1); CHECK(db.knows(d).empty());
    CHECK(db.reachable(a,0).empty()); CHECK(db.reachable(a,1)==std::vector<PersonId>{b});
    CHECK((db.reachable(a,100)==std::vector<PersonId>{b,c}));
    auto p0=db.add_post(a,"own",{0,0});
    auto p1=db.add_post(b,"quoted \"text\" \\ newline\n",{1,0});
    auto p2=db.add_post(b,"tie",{-1,0});
    auto p3=db.add_post(c,"distant",{0,2});
    CHECK(db.post(p1).author==b); CHECK(db.authored(b).size()==2);
    CHECK((db.posts_by({b,b})==std::vector<PostId>{p1,p2}));
    CHECK((db.posts_by_known(a,2)==std::vector<PostId>{p1,p2,p3}));
    rejects([&]{ db.add_post(b,"bad",{1}); });
    rejects([&]{ db.add_post(b,"bad",{NAN,0}); });
    rejects([&]{ db.add_post(PersonId{99},"bad",{0,0}); });
    rejects([&]{ db.add_knows(a,PersonId{99}); });
    rejects([&]{ db.vectors().row(PostId{99}); });
    rejects([&]{ db.author_filter({PersonId{99}}); });
    CHECK(db.post_count()==4); CHECK(db.vectors().size()==4);
    ExactL2Search search;
    auto hits=search.search(db.vectors(),{0,0},100,db.author_filter({b}));
    CHECK(hits.size()==2 && hits[0].post==p1 && hits[1].post==p2 && hits[0].squared_l2==1);
    CHECK(search.search(db.vectors(),{0,0},1)[0].post==p0);
    CHECK(search.search(db.vectors(),{0,0},0).empty());
    CHECK(search.search(db.vectors(),{0,0},10,db.author_filter({})).empty());
    rejects([&]{ search.search(db.vectors(),{0},0); });
    rejects([&]{ search.search(db.vectors(),{INFINITY,0},1); });
    Database empty(2); CHECK(search.search(empty.vectors(),{0,0},10).empty());
    db.build_author_hnsw();
    auto snapshot=(dir/"roundtrip.vg").string(); db.save(snapshot);
    rejects([&]{db.save(snapshot);});
    auto restored=Database::load(snapshot);
    CHECK(restored.person_count()==4 && restored.post_count()==4);
    for (std::uint32_t i=0;i<4;++i) {
        CHECK(restored.person(PersonId{i}).name==db.person(PersonId{i}).name);
        CHECK(!restored.person(PersonId{i}).hnsw); // Derived topology is rebuilt after load.
        CHECK(restored.knows(PersonId{i})==db.knows(PersonId{i}));
        CHECK(restored.authored(PersonId{i})==db.authored(PersonId{i}));
        CHECK(restored.post(PostId{i}).text==db.post(PostId{i}).text);
        CHECK(restored.post(PostId{i}).author==db.post(PostId{i}).author);
        for (int j=0;j<2;++j) CHECK(restored.vectors().row(PostId{i})[j]==db.vectors().row(PostId{i})[j]);
    }
    restored.build_author_hnsw();
    same_hits(MergedHnswSearch(restored,{b,c}).search({0,0},10),
              MergedHnswSearch(db,{b,c}).search({0,0},10));
    auto empty_file=(dir/"empty.vg").string(); empty.save(empty_file); CHECK(Database::load(empty_file).post_count()==0);
    const std::vector<std::string> corrupt={"", "VGRAPH 2\n2 0 0\n", "VGRAPH 1\n2 1 0\n\"x\"\n1 9\n",
        "VGRAPH 1\n2 1 1\n\"x\"\n0\n0 \"post\" 1", "VGRAPH 1\n2 0 0\ntrailing",
        "VGRAPH 1\n2 1 0\n\"x\"\n2 0 0\n", "VGRAPH 1\n2 1 1\n\"x\"\n0\n7 \"p\" 0 0\n"};
    for (const auto& text:corrupt) {
        auto file=(dir/"bad.vg").string(); { std::ofstream out(file); out<<text; }
        rejects([&]{ Database::load(file); });
    }
    // Independent full-sort oracle checks heap top-k and both filtering strategies.
    std::mt19937 rng(42); std::uniform_real_distribution<float> value(-3,3);
    for (int i=0;i<200;++i) db.add_post(PersonId{static_cast<std::uint32_t>(rng()%4)},"random",{value(rng),value(rng)});
    auto eligible=db.posts_by_known(a,2); auto lazy=db.author_filter(db.reachable(a,2));
    PostFilter materialized=[eligible](PostId p){return std::binary_search(eligible.begin(),eligible.end(),p);};
    for (int trial=0;trial<30;++trial) {
        std::vector<float> q{value(rng),value(rng)};
        std::vector<std::pair<double,std::uint32_t>> oracle;
        for (auto p:eligible) { auto v=db.vectors().row(p); double x=double(q[0])-v[0], y=double(q[1])-v[1]; oracle.emplace_back(x*x+y*y,p.value); }
        std::sort(oracle.begin(),oracle.end());
        auto k=static_cast<std::size_t>(trial);
        auto actual=search.search(db.vectors(),q,k,lazy), baseline=search.search(db.vectors(),q,k,materialized);
        CHECK(actual.size()==std::min(k,oracle.size()) && baseline.size()==actual.size());
        for (std::size_t i=0;i<actual.size();++i) {
            CHECK(actual[i].post.value==oracle[i].second && actual[i].squared_l2==oracle[i].first);
            CHECK(actual[i].post==baseline[i].post);
        }
    }
}
int main() {
    auto dir=std::filesystem::temp_directory_path()/("vgraph-tests-"+std::to_string(std::random_device{}()));
    try {
        CHECK(std::filesystem::create_directory(dir)); run(dir); std::filesystem::remove_all(dir);
        std::cout<<"All tests passed\n";
    } catch(const std::exception& e) { std::filesystem::remove_all(dir); std::cerr<<e.what()<<'\n'; return 1; }
}
