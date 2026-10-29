#include "vgraph/database.hpp"
#include "vgraph/synthetic.hpp"
#include <charconv>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <limits>
#include <iostream>
#include <stdexcept>
using namespace vgraph;
Database example() {
    Database db(3);
    auto alice = db.add_person("Alice"), bob = db.add_person("Bob"), carol = db.add_person("Carol");
    db.add_knows(alice, bob); db.add_knows(bob, carol);
    db.add_post(alice, "Alice's own post", {0, 0, 0});
    db.add_post(bob, "Graph databases", {1, 0, 0});
    db.add_post(bob, "Vector storage", {0.8f, 0.2f, 0});
    db.add_post(carol, "Two hops away", {0, 1, 0});
    return db;
}
void usage() {
    std::cerr << "Usage: vgraph_demo [demo | save FILE | load FILE | generate FILE [options]]\n"
                 "Generate options (all optional):\n"
                 "  --people N       default 1000\n"
                 "  --posts N        default 100000\n"
                 "  --dimension N    default 128\n"
                 "  --degree N       default 20 (capped at people-1)\n"
                 "  --seed N         default 42 (uint32)\n"
                 "  --model uniform|clustered   default uniform\n"
                 "  --noise X        default 0.3 (clustered standard deviation)\n";
}
std::size_t count(const std::string& text) {
    std::size_t value = 0;
    auto parsed = std::from_chars(text.data(), text.data()+text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data()+text.size())
        throw std::invalid_argument("invalid nonnegative integer: " + text);
    return value;
}
int main(int argc, char** argv) {
    try {
        std::string mode = argc > 1 ? argv[1] : "demo";
        if (mode == "--help" && argc == 2) { usage(); return 0; }
        if (mode == "generate") {
            if (argc < 3 || (argc-3)%2) { usage(); return 2; }
            SyntheticConfig c;
            for (int i=3; i<argc; i+=2) {
                std::string key=argv[i], value=argv[i+1];
                if (key=="--people") c.people=count(value);
                else if (key=="--posts") c.posts=count(value);
                else if (key=="--dimension") c.dimension=count(value);
                else if (key=="--degree") c.knows_per_person=count(value);
                else if (key=="--seed") {
                    auto seed=count(value);
                    if (seed > std::numeric_limits<std::uint32_t>::max()) throw std::invalid_argument("seed exceeds uint32");
                    c.seed=static_cast<std::uint32_t>(seed);
                } else if (key=="--model") {
                    if (value=="uniform") c.model=EmbeddingModel::Uniform;
                    else if (value=="clustered") c.model=EmbeddingModel::AuthorClustered;
                    else throw std::invalid_argument("model must be uniform or clustered");
                } else if (key=="--noise") {
                    std::size_t used=0; c.noise=std::stof(value, &used);
                    if (used!=value.size() || !std::isfinite(c.noise) || c.noise<0)
                        throw std::invalid_argument("noise must be finite and nonnegative");
                } else throw std::invalid_argument("unknown option: " + key);
            }
            if (std::filesystem::exists(argv[2]))
                throw std::runtime_error("snapshot path already exists");
            auto start=std::chrono::steady_clock::now();
            auto db=generate_synthetic(c);
            auto generated=std::chrono::steady_clock::now();
            std::size_t edges=0;
            for (std::size_t i=0;i<db.person_count();++i)
                edges+=db.knows(PersonId{static_cast<std::uint32_t>(i)}).size();
            std::cout << "Generated " << db.person_count() << " people, " << db.post_count()
                      << " posts, " << edges << " directed knows edges; dimension="
                      << c.dimension
                      << "; seed=" << c.seed << "; model="
                      << (c.model==EmbeddingModel::Uniform ? "uniform" : "clustered")
                      << "; noise=" << c.noise << "\n"
                      << "Raw vector storage: "
                      << (double(db.post_count())*c.dimension*sizeof(float)/(1024*1024))
                      << " MiB; generation: " << std::chrono::duration<double>(generated-start).count() << " s\n"
                      << "Saving snapshot..." << std::endl;
            db.save(argv[2]);
            std::cout << "Saved " << argv[2] << " ("
                      << std::chrono::duration<double>(std::chrono::steady_clock::now()-generated).count() << " s)\n";
            return 0;
        }
        if ((mode!="demo" && mode!="save" && mode!="load") ||
            (mode=="demo" && argc>2) || (mode!="demo" && argc!=3)) {
            usage(); return 2;
        }
        auto db = mode == "load" ? Database::load(argv[2]) : example();
        if (mode == "save") { db.save(argv[2]); std::cout << "Saved " << argv[2] << '\n'; }
        std::cout << "People: " << db.person_count() << ", posts: " << db.post_count()
                  << ", dimension: " << db.vectors().dimension() << '\n';
        if (!db.person_count()) return 0;
        PersonId anchor{0};
        auto authors = db.reachable(anchor, 1);
        auto posts = db.posts_by(authors);
        std::cout << "Anchor: " << db.person(anchor).name << "; one-hop authors: " << authors.size()
                  << "; eligible posts: " << posts.size() << '\n';
        std::vector<float> query(db.vectors().dimension(), 0); query[0] = 1;
        ExactL2Search search;
        for (const auto& hit : search.search(db.vectors(),
                                             query,
                                             3,
                                             db.author_filter(authors)))
            std::cout << "post=" << hit.post.value
                      << " author=" << db.person(db.post(hit.post).author).name
                      << " squared_l2=" << hit.squared_l2 << " text=" << db.post(hit.post).text << '\n';
    } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << '\n'; return 1; }
}
