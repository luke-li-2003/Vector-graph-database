#include "vgraph/database.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <stdexcept>
namespace vgraph {
namespace {
template<class T> void reserve_one(std::vector<T>& v) {
    if (v.size() == v.max_size()) throw std::length_error("container full");
    if (v.size() == v.capacity()) v.reserve(v.capacity() <= v.max_size()/2 ? std::max<std::size_t>(1, v.capacity()*2) : v.max_size());
}
void check_capacity(std::size_t size) {
    if (size >= std::numeric_limits<std::uint32_t>::max()) throw std::length_error("ID space exhausted");
}
void require(bool condition) { if (!condition) throw std::runtime_error("invalid or truncated vgraph snapshot"); }
}
PersonId Database::add_person(std::string name) {
    check_capacity(people_.size());
    reserve_one(people_); reserve_one(knows_); reserve_one(authored_);
    PersonId id{static_cast<std::uint32_t>(people_.size())};
    people_.push_back({id, std::move(name)}); knows_.emplace_back(); authored_.emplace_back();
    return id;
}
PostId Database::add_post(PersonId author, std::string text, const std::vector<float>& embedding) {
    person(author); vectors_.validate(embedding); check_capacity(posts_.size());
    reserve_one(posts_); reserve_one(authored_[author.value]); vectors_.reserve_one();
    PostId id{static_cast<std::uint32_t>(posts_.size())};
    posts_.push_back({id, author, std::move(text)});
    authored_[author.value].push_back(id); vectors_.append(embedding);
    return id;
}
const Person& Database::person(PersonId id) const { return people_.at(id.value); }
const Post& Database::post(PostId id) const { return posts_.at(id.value); }
const std::vector<PersonId>& Database::knows(PersonId id) const { return knows_.at(id.value); }
const std::vector<PostId>& Database::authored(PersonId id) const { return authored_.at(id.value); }
bool Database::add_knows(PersonId from, PersonId to) {
    person(from); person(to);
    auto& row = knows_[from.value];
    if (std::find(row.begin(), row.end(), to) != row.end()) return false;
    row.push_back(to); return true;
}
std::vector<PersonId> Database::reachable(PersonId anchor, std::size_t hops) const {
    person(anchor);
    std::vector<bool> seen(people_.size(), false); seen[anchor.value] = true;
    std::vector<PersonId> frontier{anchor}, result;
    for (std::size_t h = 0; h < hops && !frontier.empty(); ++h) {
        std::vector<PersonId> next;
        for (auto p : frontier) for (auto n : knows_[p.value]) if (!seen[n.value]) {
            seen[n.value] = true; next.push_back(n); result.push_back(n);
        }
        frontier = std::move(next);
    }
    std::sort(result.begin(), result.end()); return result;
}
std::vector<PostId> Database::posts_by(const std::vector<PersonId>& authors) const {
    std::vector<bool> selected(people_.size(), false);
    for (auto a : authors) { person(a); selected[a.value] = true; }
    std::vector<PostId> result;
    for (std::size_t a = 0; a < selected.size(); ++a) if (selected[a])
        result.insert(result.end(), authored_[a].begin(), authored_[a].end());
    std::sort(result.begin(), result.end()); return result;
}
std::vector<PostId> Database::posts_by_known(PersonId anchor, std::size_t hops) const {
    return posts_by(reachable(anchor, hops));
}
PostFilter Database::author_filter(const std::vector<PersonId>& authors) const {
    std::vector<bool> selected(people_.size(), false);
    for (auto a : authors) { person(a); selected[a.value] = true; }
    return [this, selected = std::move(selected)](PostId p) {
        auto a = post(p).author.value;
        return a < selected.size() && selected[a];
    };
}
void Database::save(const std::string& path) const {
    if (std::filesystem::exists(path)) throw std::runtime_error("snapshot path already exists");
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot create snapshot");
    out.imbue(std::locale::classic());
    out << "VGRAPH 1\n" << vectors_.dimension() << ' ' << people_.size() << ' ' << posts_.size() << '\n';
    for (const auto& p : people_) out << std::quoted(p.name) << '\n';
    for (const auto& row : knows_) {
        out << row.size(); for (auto n : row) out << ' ' << n.value; out << '\n';
    }
    out << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (const auto& p : posts_) {
        out << p.author.value << ' ' << std::quoted(p.text);
        auto row = vectors_.row(p.id);
        for (std::size_t d = 0; d < vectors_.dimension(); ++d) out << ' ' << row[d];
        out << '\n';
    }
    out.close(); if (!out) throw std::runtime_error("snapshot write failed; remove partial file before retrying");
}
Database Database::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open snapshot");
    in.imbue(std::locale::classic());
    std::string magic; unsigned version; std::size_t dim, people, posts;
    require(static_cast<bool>(in >> magic >> version >> dim >> people >> posts));
    require(magic == "VGRAPH" && version == 1);
    require(people < std::numeric_limits<std::uint32_t>::max() && posts < std::numeric_limits<std::uint32_t>::max());
    require(people != 0 || posts == 0);
    Database db(dim);
    for (std::size_t i = 0; i < people; ++i) {
        std::string name; in >> std::ws; require(in.peek() == '"');
        require(static_cast<bool>(in >> std::quoted(name))); db.add_person(std::move(name));
    }
    for (std::size_t i = 0; i < people; ++i) {
        std::size_t count; require(static_cast<bool>(in >> count)); require(count <= people);
        for (std::size_t j = 0; j < count; ++j) {
            std::uint64_t dst; require(static_cast<bool>(in >> dst)); require(dst < people);
            require(db.add_knows(PersonId{static_cast<std::uint32_t>(i)}, PersonId{static_cast<std::uint32_t>(dst)}));
        }
    }
    for (std::size_t i = 0; i < posts; ++i) {
        std::uint64_t author; std::string text;
        require(static_cast<bool>(in >> author)); require(author < people);
        in >> std::ws; require(in.peek() == '"'); require(static_cast<bool>(in >> std::quoted(text)));
        std::vector<float> v(dim); for (auto& x : v) require(static_cast<bool>(in >> x));
        db.add_post(PersonId{static_cast<std::uint32_t>(author)}, std::move(text), v);
    }
    in >> std::ws; require(in.eof()); return db;
}
}
