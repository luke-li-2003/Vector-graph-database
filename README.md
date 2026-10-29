# Minimal graph/vector database (C++17)

A self-contained, in-memory research prototype for graph-filtered vector retrieval.
It stores typed person/post vertices, directed `knows` edges, one author per post,
and dense post embeddings. Hybrid search uses vendored hnswlib v0.8.0 headers (Apache-2.0); no network
access or package installation is required to build. Exact search remains available.

## Build and run

Requires CMake 3.12+ and a C++17 compiler with `std::filesystem` support
(e.g. GCC 9+, modern Clang, or MSVC).

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
./build/vgraph_demo
./build/vgraph_demo save example.vg
./build/vgraph_demo load example.vg
```

For multi-configuration generators, the executable may be in `build/Release/`.
`save` creates the small example and saves it; `load` reads any snapshot and runs
the example query anchored at person 0 with query vector `[1, 0, ...]`.
Errors print to stderr and return a nonzero exit status. Use a new snapshot path
for each save; existing paths are refused.

Expected default output:

```text
People: 3, posts: 4, dimension: 3
Anchor: Alice; one-hop authors: 1; eligible posts: 2
post=1 author=Bob squared_l2=0 text=Graph databases
post=2 author=Bob squared_l2=0.08 text=Vector storage
```

## Hybrid search with an HNSW pre-filter

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
# Use your existing snapshot:
./build/vgraph_search synthetic.vg --anchor 0 --hops 1 --k 10 --ef 200 --compare-exact

# Or start with a smaller experiment:
./build/vgraph_demo generate hnsw-example.vg --people 1000 --posts 20000 --dimension 32
./build/vgraph_search hnsw-example.vg --anchor 7 --hops 2 --k 10 --ef 200 --compare-exact
```

The query means: find the nearest posts authored by people reachable from the
anchor in 1..hops directed `knows` edges. The anchor is excluded even on cycles.
The pipeline is **graph traversal -> materialized eligible-post bitmap ->
filtered HNSW search** over a global index. This is a precomputed graph filter
applied *during* ANN search, not filtering a global top-k afterward. HNSW can
navigate through ineligible posts; only eligible posts can enter the result set.
It does not build a separate HNSW index for each eligible subset.

The default query is `[1,0,...]`. Supply a real numerical vector as exactly
`dimension` whitespace-separated finite floats in a file:

```sh
./build/vgraph_search hnsw-example.vg --anchor 0 --k 5 --query-file query.txt --compare-exact
# Convenience for a sanity check; includes the query post itself if eligible:
./build/vgraph_search hnsw-example.vg --anchor 0 --query-post 12 --ef 200
./build/vgraph_search --help
```

`--query-file` and `--query-post` are mutually exclusive. Existing-post queries
are useful for checks, but use held-out query vectors for research measurements.
Options: `--anchor 0`, `--hops 1`, `--k 10`, `--ef 100`, `--m 16`,
`--ef-construction 200`, `--seed 42` are defaults. M must be 2..10000; both ef
values must be positive. hnswlib uses at least M for construction breadth and
at least the requested k for search breadth. Increasing `--ef` generally trades
more work for better recall. Selective filters may make traversal expensive.

Output separates snapshot loading, global index construction, graph-filter
materialization, and vector-search timings. `--compare-exact` runs the same
materialized graph filter with exact search, prints exact hits, and reports
ID-overlap recall against `min(k, eligible_posts)` exact results. Empty ground
truth reports recall as n/a. Exact search breaks equal-distance ties by PostId;
HNSW may choose a different subset at a tied cutoff, lowering ID-overlap recall.
A single query is a sanity check, not a performance benchmark.

Eligibility is exact; nearest-neighbor retrieval is approximate. HNSW may miss
true nearest posts or return fewer than the requested count. There is no silent
exact-search fallback. Returned hits sort by squared L2 and then PostId.
Distances use scalar double accumulation, matching the exact baseline and
avoiding float-distance overflow. This prioritizes numerical consistency over
SIMD throughput. Upstream manual vectorization/prefetch is disabled via
`NO_MANUAL_VECTORIZATION` because v0.8.0 prefetch neighbor reads trigger ASan
at a full adjacency list. Vendored headers themselves are unmodified.

Library use (reuse the index across queries):

```cpp
#include "vgraph/hnsw.hpp"
#include "vgraph/hybrid.hpp"
auto db = vgraph::Database::load("synthetic.vg");
vgraph::HnswConfig config;
config.ef_search = 200;
vgraph::HnswSearch index(db.vectors(), config);
std::vector<float> query(db.vectors().dimension(), 0.0f);
query[0] = 1.0f;
auto result = vgraph::hybrid_search(db, index, vgraph::PersonId{0}, 1, query, 10);
// result.hits, eligible_authors, eligible_posts, graph_ms, vector_ms
```

`HnswSearch` implements `VectorSearch`, so it can also accept `author_filter`
for future lazy-filter comparisons. `hybrid_search` deliberately materializes
post IDs and an O(post_count) bitmap. Zero k and empty eligible sets skip the
vector backend after graph/query validation.

The index copies vectors and adds topology and bookkeeping; allow additional
memory beyond database vector storage. Building a million-post index can take
substantial time. The CLI rebuilds the global index on every invocation; timings
for graph and HNSW search exclude that construction time. Index persistence and
incremental index updates are not implemented. Database snapshots still store
only graph and vectors. Rebuild HNSW after loading or adding posts. Appended-post
and different-source-store searches throw; the database/store must otherwise
remain alive, at the same address, and unmodified/unassigned for the index's
lifetime. Moving/replacing a store is unsupported. Graph-only edge additions
are allowed between queries and reflected in newly evaluated filters. Concurrent
mutation is unsupported.

Dependency provenance and license: `third_party/hnswlib/NOTICE.md` and
`third_party/hnswlib/LICENSE`. Upstream source:
https://github.com/nmslib/hnswlib/tree/v0.8.0

## Generate a larger synthetic dataset

```sh
# Defaults: 1,000 people, 100,000 posts, 128 dimensions, degree 20, seed 42.
./build/vgraph_demo generate synthetic.vg

# Larger dataset with author-correlated vectors (~488 MiB of raw vectors).
./build/vgraph_demo generate clustered.vg --people 10000 --posts 1000000 \
  --dimension 128 --degree 20 --seed 42 --model clustered --noise 0.3

# Reload and run the existing graph-filtered exact-search demo.
./build/vgraph_demo load synthetic.vg
./build/vgraph_demo --help
```

Every option is optional. `--people`, `--posts`, `--dimension`, `--degree`, and
`--seed` accept nonnegative integers; dimension must be 1..65536 and seed must
fit uint32. Zero people requires zero posts. Degree is capped at `people - 1`;
a single person has no `knows` edges. Generated edges are directed, unique, and
exclude self edges. Authors are assigned uniformly at random. Names/text are
small synthetic labels (`Person N`, `Post N`). Existing output files are refused.
The command reports counts, raw vector memory, generation time, and save time.

`--model uniform` (default) samples each component independently in [0,1),
providing an uncorrelated control. `--model clustered` samples a standard-normal
center per author, then adds independent Gaussian noise with standard deviation
`--noise` (default 0.3). Noise must be finite and nonnegative; it is ignored for
the uniform model. Noise 0 makes all posts by an author share one embedding.
Social edges remain random and independent of semantic centers in both models;
this generator does not model social communities or realistic text semantics.
Vectors are not normalized.

The seed controls independent graph, authorship, and vector RNG streams. Identical
options reproduce the same database with the same executable/standard library;
C++ random distributions do not guarantee identical output across toolchains.
The snapshot header does not record generator options, so retain the command
alongside research results.

Raw vector memory is `posts * dimension * 4` bytes on typical platforms, plus
graph, metadata, allocation capacity, and (for clustered generation) an extra
`people * dimension * 4` bytes for centers. Text snapshots are larger and slower
than raw binary vectors. Start with defaults before scaling to millions of posts.
Sampling uses O(degree) scratch space per person; duplicate checks during edge
insertion cost O(people * degree²), so this prototype targets sparse graphs.

The generator is also available as a library API:

```cpp
#include "vgraph/synthetic.hpp"
vgraph::SyntheticConfig config;
config.people = 10000;
config.posts = 1000000;
config.dimension = 64;
config.seed = 123;
config.model = vgraph::EmbeddingModel::AuthorClustered;
config.noise = 0.2f;
auto db = vgraph::generate_synthetic(config);
```

## Use the library

Link the CMake target `vgraph` and include `vgraph/database.hpp`.

```cpp
#include "vgraph/database.hpp"
using namespace vgraph;

Database db(3); // Dimension fixed for this database, not at compile time.
auto alice = db.add_person("Alice");
auto bob = db.add_person("Bob");
db.add_knows(alice, bob); // Does not add Bob -> Alice.
auto post = db.add_post(bob, "Research notes", {0.8f, 0.1f, 0.2f});

auto authors = db.reachable(alice, 1);
auto eligible_posts = db.posts_by(authors); // Materialized graph result.
ExactL2Search search;
auto hits = search.search(db.vectors(), {1, 0, 0}, 10,
                          db.author_filter(authors)); // Lazy author membership.
db.save("research.vg");
auto restored = Database::load("research.vg");
```

To compare a materialized filter to lazy author membership, turn the sorted
`eligible_posts` into a membership callback:

```cpp
#include <algorithm>
PostFilter materialized = [eligible_posts](PostId p) {
    return std::binary_search(eligible_posts.begin(), eligible_posts.end(), p);
};
auto baseline = search.search(db.vectors(), {1, 0, 0}, 10, materialized);
```

Both searches are exact and return identical results. These are correctness
baselines; neither demonstrates ANN acceleration or a research contribution.

## Data model and semantics

- `PersonId` and `PostId` are distinct aggregate types with dense, stable 32-bit
  values assigned from zero. IDs are database-local, not globally unique.
- Person has a name; Post has text and exactly one existing Person author.
  Author adjacency represents `Person -> authored -> Post`; `post(id).author`
  provides the reverse lookup. Embeddings live separately in `VectorStore`.
- Embedding dimension is 1..65536. Insertions and queries reject dimension
  mismatches and non-finite values. Contiguous row-major float storage uses
  approximately `posts * dimension * sizeof(float)` bytes, excluding metadata.
- `knows` is directed. Duplicate edges are no-ops; self edges are allowed.
- `reachable(anchor, h)` is bounded breadth-first traversal, returning unique
  people reachable in 1..h hops. It excludes the anchor, including through cycles
  or self edges. Zero hops yields no authors. Results are sorted by ID.
- `posts_by` deduplicates authors and returns posts sorted by ID.
  `posts_by_known` combines bounded reachability and authorship expansion.
- `ExactL2Search` scans all post IDs, tests eligibility before computing distance,
  and maintains a top-k heap. Scores are **squared Euclidean distance**, accumulated
  in double precision. Smaller is better; ties sort by PostId. k=0 returns empty;
  k larger than the eligible set returns the whole eligible set. Query validation
  still applies to k=0 and empty stores. Vectors are not normalized.
- Unknown IDs throw `std::out_of_range`; invalid vectors throw
  `std::invalid_argument`; snapshot failures throw exceptions. Failed insertion
  does not add a partial vertex or authorship edge.

The author-filter callback owns its membership bitmap but borrows the database.
Keep that database alive at the same address: do not move, assign, or destroy it
while using the callback. The bitmap captures the selected authors at creation;
new posts by those authors qualify. References/pointers returned by vertex and
vector accessors may be invalidated by subsequent insertion. Do not mutate the
database while searching or traversing it.

## Persistence

`save`/`load` implement a version-1 text snapshot with a `VGRAPH 1` header,
dimension, counts, quoted person names, directed adjacency, and authored posts
with float vectors. IDs are implicit in row order. Classic locale and
`max_digits10` preserve finite float values exactly across a round trip.
Quotes, backslashes, whitespace, and newlines in strings are supported.

Loading builds a separate database and validates version, dimensions, counts,
references, duplicate edges, vectors, truncation, and trailing data. Files have
no checksum, so valid-looking edits cannot be detected. Loading is intended for
local research snapshots, not adversarial input; no memory/file-size quota is
imposed. Snapshots are not a transaction log. Saving is not atomic or crash-safe;
a failed write can leave a partial file. Concurrent writers are unsupported,
including the existing-path check. Use a new file name and retain prior snapshots.

## Layout and extension point

```text
include/vgraph/database.hpp  Typed graph and query API
include/vgraph/vector.hpp    VectorStore, VectorSearch, ExactL2Search
src/database.cpp            Graph operations and snapshots
src/vector.cpp              Vector validation and exact top-k search
include/vgraph/hnsw.hpp      HNSW configuration and backend API
include/vgraph/hybrid.hpp    Materialized graph-filter query API
src/hnsw.cpp                hnswlib adapter and double L2 space
src/hybrid.cpp              Hybrid query execution
src/search_main.cpp         HNSW search CLI
third_party/hnswlib/         Vendored headers, license, provenance
src/main.cpp                CLI/demo and generation command
include/vgraph/synthetic.hpp Generator configuration API
src/synthetic.cpp           Seeded synthetic graph/vector generation
tests/tests.cpp            Graph, search, persistence, and failure tests
```

`HnswSearch` and `ExactL2Search` implement the same `VectorSearch` interface.
ANN topology stays separate from graph/vector storage. Graph-derived seeds or
joint navigation would need an additional interface later.

Insertion uses geometrically grown arrays. Reachability is O(V + visited edges)
with O(V) scratch space; materializing posts also allocates an author bitmap.
Exact search costs O(N * filter-cost + eligible * (dimension + log k)) with O(k)
result memory (bounded by eligible count). Lazy filtering avoids an endpoint-ID
list but still scans all posts; it is not automatically faster for selective queries.

This deliberately omits deletion, updates, transactions, concurrency control,
WAL, query languages, and large-scale storage management. HNSW is an in-memory
secondary search structure, not a persistent database storage index.

## Validation

Tests remain active in Release builds (no disabled `assert` checks). They cover
cycles, directionality, duplicates, hop bounds, typed IDs, authorship, invalid
IDs/vectors, empty stores, tie ordering, k boundaries, snapshot round trips and
malformed files. Generator tests cover seed reproducibility, uniform ranges,
author clustering, degree caps, empty graphs, and invalid configurations. A seeded random test compares top-k to an independent full-sort
oracle and compares materialized and lazy filters.

Optional GCC/Clang sanitizer build:

```sh
cmake -S . -B build-sanitize -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"
cmake --build build-sanitize -j2
ctest --test-dir build-sanitize --output-on-failure
```

Validated with GCC 11.4: Release build, CTest, and demo pass. AddressSanitizer
and UndefinedBehaviorSanitizer tests pass with `ASAN_OPTIONS=detect_leaks=0`;
LeakSanitizer cannot run in the ptrace-based execution environment. Leak
detection was therefore not verified.

HNSW tests cover filtered-vs-postfiltered semantics, no eligible posts, k=0,
k exceeding eligible count, hop limits, invalid configuration/query/anchor,
empty indexes, source mismatch, stale index rejection and rebuild, extreme
finite vectors, and high-ef agreement with exact search on seeded random data.
In environments where LeakSanitizer cannot run under ptrace, use:

```sh
ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1 \
  ctest --test-dir build-sanitize --output-on-failure
```
