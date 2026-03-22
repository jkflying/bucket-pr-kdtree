#include <flinn.h>
#include <nanoflann.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using Clock = std::chrono::high_resolution_clock;
using Dsec = std::chrono::duration<double>;
using Dms = std::chrono::duration<double, std::milli>;

static constexpr double SAMPLE_MS = 500.0;
static const int KS[] = {1, 2, 4, 8, 16, 32, 64};
static const int NK = sizeof(KS) / sizeof(KS[0]);

static double elapsed_ms(Clock::time_point t0) { return Dms(Clock::now() - t0).count(); }

// nanoflann adaptor
template <int Dims> struct PointCloud
{
    std::vector<std::array<double, Dims>> pts;
    std::size_t kdtree_get_point_count() const { return pts.size(); }
    double kdtree_get_pt(std::size_t idx, std::size_t dim) const { return pts[idx][dim]; }
    template <class BBOX> bool kdtree_get_bbox(BBOX&) const { return false; }
};

template <int Dims, int BucketSize>
void benchFlinn(const std::string& dataset,
                const std::vector<std::array<double, Dims>>& points,
                const std::vector<std::array<double, Dims>>& queries,
                int k)
{
    flinn::FlinnIndex<int, Dims, BucketSize> tree;
    for (std::size_t i = 0; i < points.size(); i++)
        tree.addPoint(points[i], static_cast<int>(i), false);
    tree.splitOutstanding();

    auto searcher = tree.searcher();
    volatile int sink = 0;
    std::size_t count = 0;
    std::size_t qi = 0;
    auto t0 = Clock::now();
    while (elapsed_ms(t0) < SAMPLE_MS)
    {
        sink = static_cast<int>(searcher.search(queries[qi], std::numeric_limits<double>::max(), k).size());
        ++count;
        if (++qi >= queries.size())
            qi = 0;
    }
    double secs = Dsec(Clock::now() - t0).count();

    std::cout << "flinn," << dataset << "," << Dims << "," << points.size() << "," << k << "," << count << "," << secs
              << "\n";
}

template <int Dims>
void benchNanoflann(const std::string& dataset,
                    const std::vector<std::array<double, Dims>>& points,
                    const std::vector<std::array<double, Dims>>& queries,
                    int k)
{
    using adaptor_t = nanoflann::L2_Simple_Adaptor<double, PointCloud<Dims>>;
    using tree_t = nanoflann::KDTreeSingleIndexAdaptor<adaptor_t, PointCloud<Dims>, Dims>;

    PointCloud<Dims> cloud;
    cloud.pts = points;

    tree_t index(Dims, cloud);
    index.buildIndex();

    std::vector<uint32_t> ret_idx(k);
    std::vector<double> ret_dist(k);
    volatile int sink = 0;
    std::size_t count = 0;
    std::size_t qi = 0;
    auto t0 = Clock::now();
    while (elapsed_ms(t0) < SAMPLE_MS)
    {
        sink = index.knnSearch(queries[qi].data(), static_cast<uint32_t>(k), ret_idx.data(), ret_dist.data());
        ++count;
        if (++qi >= queries.size())
            qi = 0;
    }
    double secs = Dsec(Clock::now() - t0).count();

    std::cout << "nanoflann," << dataset << "," << Dims << "," << points.size() << "," << k << "," << count << ","
              << secs << "\n";
}

template <int Dims, int BucketSize>
void runConfig(const std::string& dataset,
               const std::vector<std::array<double, Dims>>& points,
               const std::vector<std::array<double, Dims>>& queries,
               int k)
{
    benchFlinn<Dims, BucketSize>(dataset, points, queries, k);
    benchNanoflann<Dims>(dataset, points, queries, k);
    std::cerr << "  D=" << Dims << " N=" << points.size() << " k=" << k << " done\n";
}

template <int Dims>
void makeData(std::size_t nPoints,
              std::size_t nQueries,
              std::vector<std::array<double, Dims>>& points,
              std::vector<std::array<double, Dims>>& queries)
{
    std::mt19937_64 rng(0xc0ffee42);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    points.resize(nPoints);
    queries.resize(nQueries);
    for (auto& p : points)
        for (auto& x : p)
            x = uni(rng);
    for (auto& q : queries)
        for (auto& x : q)
            x = uni(rng);
}

template <int Dims, int BucketSize> void runDim()
{
    const std::size_t NQ = 10000;
    const std::size_t sizes[] = {10000, 100000, 1000000};
    for (auto n : sizes)
    {
        std::vector<std::array<double, Dims>> pts, qry;
        makeData<Dims>(n, NQ, pts, qry);
        for (int i = 0; i < NK; i++)
            runConfig<Dims, BucketSize>("uniform", pts, qry, KS[i]);
    }
}

template <int Dims> std::vector<std::array<double, Dims>> loadPoints(const std::string& path)
{
    std::vector<std::array<double, Dims>> points;
    std::ifstream in(path);
    if (!in.is_open())
    {
        std::cerr << "  ERROR: cannot open " << path << "\n";
        return points;
    }
    std::string line;
    while (std::getline(in, line))
    {
        std::istringstream iss(line);
        std::array<double, Dims> pt;
        bool ok = true;
        for (int d = 0; d < Dims; d++)
        {
            if (!(iss >> pt[d]))
            {
                ok = false;
                break;
            }
        }
        if (ok)
            points.push_back(pt);
    }
    return points;
}

template <int Dims, int BucketSize> void runRealDataset(const std::string& path, const std::string& name)
{
    auto points = loadPoints<Dims>(path);
    if (points.empty())
        return;

    const std::size_t NQ = std::min<std::size_t>(10000, points.size());
    std::vector<std::array<double, Dims>> queries(NQ);
    std::mt19937_64 rng(0xbeef);
    std::vector<std::size_t> indices(points.size());
    for (std::size_t i = 0; i < indices.size(); i++)
        indices[i] = i;
    std::shuffle(indices.begin(), indices.end(), rng);

    flinn::FlinnIndex<int, Dims, BucketSize> qtree;
    for (std::size_t i = 0; i < points.size(); i++)
        qtree.addPoint(points[i], static_cast<int>(i), false);
    qtree.splitOutstanding();
    auto qsearcher = qtree.searcher();

    for (std::size_t i = 0; i < NQ; i++)
    {
        const auto& neighbors = qsearcher.search(points[indices[i]], std::numeric_limits<double>::max(), 3);

        constexpr double weights[] = {3.0, 2.0, 1.0};
        double wsum = 0.0;
        std::array<double, Dims> centroid {};
        for (std::size_t j = 0; j < neighbors.size(); j++)
        {
            double w = weights[j];
            wsum += w;
            for (int d = 0; d < Dims; d++)
                centroid[d] += w * points[neighbors[j].payload][d];
        }
        for (int d = 0; d < Dims; d++)
            centroid[d] /= wsum;
        queries[i] = centroid;
    }

    std::cerr << "  " << name << ": " << points.size() << " points, D=" << Dims << "\n";
    for (int i = 0; i < NK; i++)
        runConfig<Dims, BucketSize>(name, points, queries, KS[i]);
}

void runSynthetic()
{
    runDim<2, 16>();
    runDim<3, 16>();
    runDim<4, 32>();
    runDim<8, 64>();
}

void runReal(const std::string& dataDir)
{
    runRealDataset<2, 16>(dataDir + "/fire_hydrants_california.csv", "fire hydrants CA");
    runRealDataset<2, 16>(dataDir + "/street_lamps_california.csv", "street lamps CA");
    runRealDataset<6, 32>(dataDir + "/colored_meshes_xyzrgb.csv", "colored meshes XYZRGB");
}

int main(int argc, char* argv[])
{
    std::cout << "library,dataset,dims,n,k,num_queries,query_time_s\n";

    bool doSynthetic = true;
    bool doReal = true;
    std::string dataDir = "..";

    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--synthetic")
        {
            doSynthetic = true;
            doReal = false;
        }
        else if (arg == "--real")
        {
            doReal = true;
            doSynthetic = false;
        }
        else if (arg == "--data-dir" && i + 1 < argc)
        {
            dataDir = argv[++i];
        }
    }

    if (doSynthetic)
        runSynthetic();
    if (doReal)
        runReal(dataDir);

    return 0;
}
