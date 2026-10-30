#include "vgraph/hnsw.hpp"
#include "vgraph/hybrid.hpp"
#include <charconv>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <locale>
#include <stdexcept>
#include <unordered_set>
using namespace vgraph;
namespace {
std::size_t number(const std::string& text) {
    std::size_t n=0; auto r=std::from_chars(text.data(),text.data()+text.size(),n);
    if (r.ec!=std::errc{} || r.ptr!=text.data()+text.size()) throw std::invalid_argument("invalid integer: "+text);
    return n;
}
void usage() {
    std::cerr << "Usage: vgraph_search FILE [options]\n"
                 "  --anchor N         person ID (default 0)\n"
                 "  --hops N           knows hop limit (default 1)\n"
                 "  --k N              result count (default 10)\n"
                 "  --index author|global  merged author HNSWs (default) or global filtered HNSW\n"
                 "  --ef N             HNSW search breadth (default 100)\n"
                 "  --m N              HNSW links (default 16)\n"
                 "  --ef-construction N  build breadth (default 200)\n"
                 "  --seed N           build and witness-selection seed (default 42)\n"
                 "  --query-file FILE  whitespace-separated floats, exactly dimension values\n"
                 "  --query-post N     use an existing post vector instead\n"
                 "  --compare-exact    report filtered exact results and recall\n"
                 "Default query: [1,0,...]. Query-file and query-post are mutually exclusive.\n";
}
}
int main(int argc,char** argv) {
    try {
        if (argc<2) { usage(); return 2; }
        if (std::string(argv[1])=="--help") {usage(); return 0;}
        std::size_t anchor=0,hops=1,k=10,query_post=0;
        bool compare=false,has_post=false; std::string query_file,mode="author"; HnswConfig config;
        for (int i=2;i<argc;++i) {
            std::string key=argv[i];
            if (key=="--compare-exact") {compare=true; continue;}
            if (++i>=argc) throw std::invalid_argument("missing value for "+key);
            std::string value=argv[i];
            if (key=="--query-file") query_file=value;
            else if (key=="--index") mode=value;
            else if (key=="--anchor") anchor=number(value);
            else if (key=="--hops") hops=number(value);
            else if (key=="--k") k=number(value);
            else if (key=="--ef") config.ef_search=number(value);
            else if (key=="--m") config.m=number(value);
            else if (key=="--ef-construction") config.ef_construction=number(value);
            else if (key=="--seed") {
                auto n=number(value);
                if(n>std::numeric_limits<std::uint32_t>::max()) throw std::invalid_argument("seed exceeds uint32");
                config.seed=static_cast<std::uint32_t>(n);
            } else if (key=="--query-post") {has_post=true;query_post=number(value);}
            else throw std::invalid_argument("unknown option: "+key);
        }
        if (has_post && !query_file.empty()) throw std::invalid_argument("choose query-file or query-post");
        if (mode!="author" && mode!="global") throw std::invalid_argument("index must be author or global");
        if (anchor>std::numeric_limits<std::uint32_t>::max() || query_post>std::numeric_limits<std::uint32_t>::max())
            throw std::invalid_argument("ID exceeds uint32");
        auto load_start=std::chrono::steady_clock::now();
        std::cout << "Loading snapshot " << argv[1] << "..." << std::endl;
        auto db=Database::load(argv[1]);
        db.person(PersonId{static_cast<std::uint32_t>(anchor)});
        auto loaded=std::chrono::steady_clock::now();
        std::vector<float> query(db.vectors().dimension(),0);query[0]=1;
        if (has_post) {
            auto row=db.vectors().row(PostId{static_cast<std::uint32_t>(query_post)});
            query.assign(row,row+query.size());
        } else if (!query_file.empty()) {
            std::ifstream in(query_file); in.imbue(std::locale::classic());
            if (!in) throw std::runtime_error("cannot open query file");
            for (auto& x:query) if (!(in>>x)) throw std::invalid_argument("invalid or short query file");
            in>>std::ws; if(!in.eof()) throw std::invalid_argument("query file has extra values or trailing data");
        }
        db.vectors().validate(query);
        if (mode=="author")
            std::cout << "Building author HNSWs for " << db.person_count() << " authors over " << db.post_count() << " posts..." << std::endl;
        else std::cout << "Building global HNSW over " << db.post_count() << " posts..." << std::endl;
        auto build_start=std::chrono::steady_clock::now();
        std::unique_ptr<HnswSearch> index;
        if (mode=="author") db.build_author_hnsw(config);
        else {
            auto last_report=build_start;
            HnswBuildProgress progress=[&](std::size_t done,std::size_t total) {
                auto now=std::chrono::steady_clock::now();
                if (done!=total && now-last_report<std::chrono::seconds(1)) return;
                last_report=now;
                std::cout << "global_build_posts=" << done << '/' << total
                          << " percent=" << (total ? 100.0*done/total : 100.0)
                          << " elapsed_s=" << std::chrono::duration<double>(now-build_start).count()
                          << std::endl;
            };
            index=std::make_unique<HnswSearch>(db.vectors(),config,progress);
        }
        auto built=std::chrono::steady_clock::now();
        std::cout << "Index built in " << std::chrono::duration<double>(built-build_start).count()
                  << " s. Searching " << mode << " index..." << std::endl;
        auto anchor_id=PersonId{static_cast<std::uint32_t>(anchor)};
        auto result=mode=="author" ? merged_hnsw_search(db,anchor_id,hops,query,k,config)
                                   : hybrid_search(db,*index,anchor_id,hops,query,k);
        std::cout << "index=" << mode << " load_ms=" << std::chrono::duration<double,std::milli>(loaded-load_start).count()
                  << " build_ms=" << std::chrono::duration<double,std::milli>(built-build_start).count()
                  << " M=" << config.m << " ef_construction=" << config.ef_construction
                  << " ef_search=" << config.ef_search << " seed=" << config.seed << '\n'
                  << "eligible_authors=" << result.eligible_authors << " eligible_posts=" << result.eligible_posts
                  << " witnesses=" << result.witnesses
                  << " returned=" << result.hits.size() << " graph_ms=" << result.graph_ms
                  << " merge_ms=" << result.merge_ms
                  << " hnsw_ms=" << result.vector_ms << '\n';
        for (const auto& hit:result.hits)
            std::cout << "post=" << hit.post.value << " author=" << db.post(hit.post).author.value
                      << " squared_l2=" << hit.squared_l2 << '\n';
        if (compare) {
            std::cout << "Comparing with exact search..." << std::endl;
            ExactL2Search exact;
            auto truth=hybrid_search(db,exact,PersonId{static_cast<std::uint32_t>(anchor)},hops,query,k);
            std::unordered_set<std::uint32_t> expected;
            for (auto h:truth.hits) expected.insert(h.post.value);
            std::size_t overlap=0; for(auto h:result.hits) overlap+=expected.count(h.post.value);
            std::cout << "exact_graph_ms=" << truth.graph_ms << " exact_vector_ms=" << truth.vector_ms;
            if (truth.hits.empty()) std::cout << " recall@" << k << "=n/a (empty exact result)\n";
            else std::cout << " recall@" << k << '=' << double(overlap)/truth.hits.size() << '\n';
            for(auto h:truth.hits) std::cout << "exact_post=" << h.post.value << " squared_l2=" << h.squared_l2 << '\n';
        }
    } catch(const std::exception& e) {std::cerr << "Error: " << e.what() << '\n';return 1;}
}
