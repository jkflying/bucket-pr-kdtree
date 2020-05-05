#include <KDTree.h>

#include <cstdlib>
#include <ctime>
#include <iostream>

double drand() { return (rand() / (RAND_MAX + 1.)); }
void example();
void accuracyTest();
void duplicateTest();
void performanceTest();

int main()
{
    example();
    accuracyTest();
    duplicateTest();
    performanceTest();
    return 0;
}

void example()
{
    std::cout << "Example starting..." << std::endl;
    // setup
    using tree_t = jk::tree::KDTree<std::string, 2>;
    using point_t = std::array<double, 2>;
    tree_t tree;
    tree.addPoint(point_t{{1, 2}}, "George");
    tree.addPoint(point_t{{1, 3}}, "Harold");
    tree.addPoint(point_t{{7, 7}}, "Melvin");

    // KNN search
    point_t lazyMonsterLocation{{6, 6}}; // this monster will always try to eat the closest people
    const std::size_t monsterHeads = 2; // this monster can eat two people at once
    auto lazyMonsterVictims = tree.searchKnn(lazyMonsterLocation, monsterHeads);
    for (const auto& victim : lazyMonsterVictims)
    {
        std::cout << victim.payload << " closest to lazy monster, with distance " << sqrt(victim.distance) << "!"
                  << std::endl;
    }

    // ball search
    point_t stationaryMonsterLocation{{8, 8}}; // this monster doesn't move, so can only eat people that are close
    const double neckLength = 6.0; // it can only reach within this range
    auto potentialVictims = tree.searchBall(stationaryMonsterLocation, neckLength * neckLength); // metric is SquaredL2
    std::cout << "Stationary monster can reach any of " << potentialVictims.size() << " people!" << std::endl;

    // hybrid KNN/ball search
    auto actualVictims
        = tree.searchCapacityLimitedBall(stationaryMonsterLocation, neckLength * neckLength, monsterHeads);
    std::cout << "The stationary monster will try to eat ";
    for (const auto& victim : actualVictims)
    {
        std::cout << victim.payload << " and ";
    }
    std::cout << "nobody else." << std::endl;
    std::cout << "Example completed" << std::endl;
}

void accuracyTest()
{
    // GIVEN: a tree, a bunch of random points to put in it, and dumb brute force methods to compare results to

    std::cout << "Accuracy tests starting..." << std::endl;
    static const int dims = 4;
    std::vector<std::array<double, dims>> points;
    using tree_t = jk::tree::KDTree<int, dims>;
    tree_t tree;
    int count = 0;
    std::srand(1234567);

    auto bruteforceKNN = [&](const std::array<double, dims>& searchLoc, size_t K) -> std::vector<std::pair<double, int>> {
        std::vector<std::pair<double, int>> dists;
        for (std::size_t i = 0; i < points.size(); i++)
        {
            double distance = tree_t::distance_t::distance(searchLoc, points[i]);
            dists.emplace_back(distance, i);
        }
        size_t actualK = std::min(points.size(), K);
        std::partial_sort(dists.begin(), dists.begin() + actualK, dists.end());
        dists.resize(actualK);
        return dists;
    };

    auto bruteforceRadius
        = [&](const std::array<double, dims>& searchLoc, double radius) -> std::vector<std::pair<double, int>> {
        std::vector<std::pair<double, int>> dists;
        for (std::size_t i = 0; i < points.size(); i++)
        {
            double distance = tree_t::distance_t::distance(searchLoc, points[i]);
            dists.emplace_back(distance, i);
        }
        std::sort(dists.begin(), dists.end());
        auto iter = std::lower_bound(dists.begin(), dists.end(), std::make_pair(radius, int(0)));
        std::size_t inliers = std::distance(dists.begin(), iter);
        dists.resize(inliers);
        return dists;
    };

    auto randomPoint = []() {
        std::array<double, dims> loc;
        for (std::size_t j = 0; j < dims; j++)
        {
            loc[j] = drand();
        }
        return loc;
    };

    // THEN: the tree size should match
    if (tree.size() != 0)
    {
        std::cout << "Count doesn't match!!!" << std::endl;
    }

    auto searcher = tree.searcher();
    for (std::size_t j = 0; j < 2000; j++)
    {
        const std::array<double, dims> loc = randomPoint();

        // WHEN: we search for the KNN with the tree and the brute force
        const std::size_t k = 50;
        auto tnn = tree.searchKnn(loc, k);
        auto bnn = bruteforceKNN(loc, k);
        auto snn = searcher.search(loc, 1e9, k);



        // THEN: the returned result sizes should match
        if (tnn.size() != bnn.size() || snn.size() != bnn.size() || bnn.size() > std::min(k, points.size()))
        {
            std::cout << "Searched for " << k << ", found " << tnn.size() << std::endl;
        }

        if (bnn.size() > 0)
        {
            auto nn = tree.search(loc);
            if (nn.payload != bnn[0].second)
            {
                std::cout << "1nn payloads not equal" << std::endl;
            }
            if (std::abs(bnn[0].first - nn.distance) > 1e-10)
            {
                std::cout << "1nn distances not equal" << std::endl;
            }

        }

        // AND: the entries should match - both index, and distance
        for (std::size_t i = 0; i < tnn.size(); i++)
        {
            if (std::abs(bnn[i].first - tnn[i].distance) > 1e-10)
            {
                std::cout << "distances not equal" << std::endl;
            }
            if (std::abs(bnn[i].first - snn[i].distance) > 1e-10)
            {
                std::cout << "distances not equal" << std::endl;
            }
            if (bnn[i].second != tnn[i].payload)
            {
                std::cout << "payloads not equal" << std::endl;
            }
            if (bnn[i].second != snn[i].payload)
            {
                std::cout << "payloads not equal" << std::endl;
            }
        }

        // WHEN: we add the point we searched for to the tree for next time
        tree.addPoint(loc, count++);
        points.push_back(loc);

        // THEN: the tree size should match
        if (tree.size() != points.size())
        {
            std::cout << "Count doesn't match!!!" << std::endl;
        }

    }

    tree_t tree2;

    for (std::size_t j = 0; j < points.size(); j++)
    {
        tree2.addPoint(points[j], j, false);
    }
    tree2.splitOutstanding();

    for (std::size_t j = 0; j < points.size(); j++)
    {
        const std::array<double, dims> loc = randomPoint();
        const double radius = 0.7;

        auto tnn = tree2.searchBall(loc, radius);
        auto bnn = bruteforceRadius(loc, radius);
        if (tnn.size() != bnn.size())
        {
            std::cout << "Brute force results are not the same size as tree results" << std::endl;
            continue;
        }

        if (tnn.size() && tnn.back().distance > radius)
        {
            std::cout << "Searched for max radius " << radius << ", found " << tnn.back().distance << std::endl;
        }
        for (std::size_t i = 0; i < tnn.size(); i++)
        {
            if (std::abs(bnn[i].first - tnn[i].distance) > 1e-10)
            {
                std::cout << "distances not equal" << std::endl;
            }
            if (bnn[i].second != tnn[i].payload)
            {
                std::cout << "payloads not equal" << std::endl;
            }
        }
    }

    std::cout << "Accuracy tests completed" << std::endl;
}

void duplicateTest()
{
    std::cout << "Duplicate tests started" << std::endl;

    // GIVEN: the same point added to the tree lots and lots of times (multiple buckets worth)
    static const int dims = 11;
    auto randomPoint = []() {
        std::array<double, dims> loc;
        for (std::size_t j = 0; j < dims; j++)
        {
            loc[j] = drand();
        }
        return loc;
    };
    using tree_t = jk::tree::KDTree<int, dims>;
    tree_t tree;

    const std::array<double, dims> loc = randomPoint();

    for (int i = 0; i < 5000; i++) {
        tree.addPoint(loc, i, false);
    }
    auto almostLoc = loc;
    almostLoc[0] = std::nextafter(loc[0], 1e9);
    tree.addPoint(almostLoc, tree.size(), false); // and another point, just so not the entire treee is one point

    // WHEN: the tree is split and queried
    tree.splitOutstanding();
    auto tnn = tree.searchKnn(loc, 80);


    // THEN: it should still behave normally - correct K for KNN, no crashes, good code coverage, etc
    if (tnn.size() != 80)
    {
        std::cout << "Incorrect K: " << tnn.size() << std::endl;
    }
    std::cout << "Duplicate tests completed" << std::endl;

}

#define DURATION double(((previous = current) * 0 + (current = std::clock()) - previous) / double(CLOCKS_PER_SEC))
void performanceTest()
{
    std::cout << "Performance tests starting..." << std::endl;
    std::clock_t previous = std::clock(), current = previous;

    static const int dims = 2;
    std::cout << "adding ";
    std::vector<std::array<double, dims>> points;
    jk::tree::KDTree<int, dims, 8> tree;

    int count = 0;
    std::srand(1234567);

    auto randomPoint = []() {
        std::array<double, dims> loc;
        for (std::size_t j = 0; j < dims; j++)
        {
            loc[j] = drand();
        }
        return loc;
    };

    for (int i = 0; i < 400 * 1000; i++)
    {
        std::array<double, dims> loc = randomPoint();
        tree.addPoint(loc, count++, false);

        points.push_back(loc);
    }

    std::vector<std::array<double, dims>> searchPoints;
    for (int i = 0; i < 100 * 1000; i++)
    {
        searchPoints.push_back(randomPoint());
    }

    std::cout << DURATION << "s" << std::endl;
    std::cout << "splitting ";
    tree.splitOutstanding();
    std::cout << DURATION << "s" << std::endl;
    for (int j = 0; j < 3; j++)
    {
        std::cout << "searching " << (j + 1) << " ";

        for (auto p : searchPoints)
        {
            const int k = 3;
            auto nn = tree.searchKnn(p, k);

            if (nn.size() != k)
            {
                std::cout << nn.size() << " instead of " << k << " ERROR" << std::endl;
            }
        }
        std::cout << DURATION << "s" << std::endl;
    }
    for (int j = 0; j < 3; j++)
    {
        std::cout << "bulk searching " << (j + 1) << " ";

        const int k = 3;
        auto searcher = tree.searcher();
        for (auto p : searchPoints)
        {
            const auto& nn = searcher.search(p, std::numeric_limits<double>::max(), k);

            if (nn.size() != k)
            {
                std::cout << nn.size() << " instead of " << k << " ERROR" << std::endl;
            }
        }
        std::cout << DURATION << "s" << std::endl;
    }
    std::cout << "Performance tests completed" << std::endl;
}
