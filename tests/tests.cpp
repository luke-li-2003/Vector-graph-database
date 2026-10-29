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
    Database empty(2); HnswSearch empty_index(empty.vectors());
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
    auto snapshot=(dir/"roundtrip.vg").string(); db.save(snapshot);
    rejects([&]{db.save(snapshot);});
    auto restored=Database::load(snapshot);
    CHECK(restored.person_count()==4 && restored.post_count()==4);
    for (std::uint32_t i=0;i<4;++i) {
        CHECK(restored.person(PersonId{i}).name==db.person(PersonId{i}).name);
        CHECK(restored.knows(PersonId{i})==db.knows(PersonId{i}));
        CHECK(restored.authored(PersonId{i})==db.authored(PersonId{i}));
        CHECK(restored.post(PostId{i}).text==db.post(PostId{i}).text);
        CHECK(restored.post(PostId{i}).author==db.post(PostId{i}).author);
        for (int j=0;j<2;++j) CHECK(restored.vectors().row(PostId{i})[j]==db.vectors().row(PostId{i})[j]);
    }
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
