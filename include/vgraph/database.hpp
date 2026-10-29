#pragma once
#include "vgraph/vector.hpp"
#include <string>
#include <vector>
namespace vgraph {
struct PersonId {
    std::uint32_t value;
    friend bool operator==(PersonId a, PersonId b) { return a.value == b.value; }
    friend bool operator<(PersonId a, PersonId b) { return a.value < b.value; }
};
struct Person { PersonId id; std::string name; };
struct Post { PostId id; PersonId author; std::string text; };
class Database {
public:
    explicit Database(std::size_t dimension) : vectors_(dimension) {}
    PersonId add_person(std::string name);
    PostId add_post(PersonId author, std::string text, const std::vector<float>& embedding);
    bool add_knows(PersonId from, PersonId to); // duplicate is a no-op; self edges allowed
    const Person& person(PersonId) const;
    const Post& post(PostId) const;
    const std::vector<PersonId>& knows(PersonId) const;
    const std::vector<PostId>& authored(PersonId) const;
    // Distinct vertices reachable in 1..max_hops; excludes anchor, even on cycles.
    std::vector<PersonId> reachable(PersonId anchor, std::size_t max_hops) const;
    std::vector<PostId> posts_by(const std::vector<PersonId>& authors) const;
    std::vector<PostId> posts_by_known(PersonId anchor, std::size_t max_hops = 1) const;
    // Owns an author bitmap, so it can outlive the input author list. Database must
    // outlive the callback; later posts by these authors are also eligible.
    PostFilter author_filter(const std::vector<PersonId>& authors) const;
    const VectorStore& vectors() const { return vectors_; }
    std::size_t person_count() const { return people_.size(); }
    std::size_t post_count() const { return posts_.size(); }
    // Portable versioned text snapshot. Refuses to overwrite an existing path.
    void save(const std::string& path) const;
    static Database load(const std::string& path);
private:
    std::vector<Person> people_;
    std::vector<Post> posts_;
    std::vector<std::vector<PersonId>> knows_;
    std::vector<std::vector<PostId>> authored_;
    VectorStore vectors_;
};
} // namespace vgraph
