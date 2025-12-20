#include <KDTree.h>

#include <cstdlib>
#include <ctime>
#include <iostream>

double drand() { return (rand() / (RAND_MAX + 1.)); }
void example();
void accuracyTest();
void duplicateTest();
void performanceTest();
void iteratorTest();
void rebalanceTest();
void eightDTest();
void removalTest();
void l1DistanceTest();

int main()
{
    example();
    accuracyTest();
    duplicateTest();
    iteratorTest();
    rebalanceTest();
    eightDTest();
    removalTest();
    l1DistanceTest();
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
    tree.addPoint(point_t {{1, 2}}, "George");
    tree.addPoint(point_t {{1, 3}}, "Harold");
    tree.addPoint(point_t {{7, 7}}, "Melvin");

    // KNN search
    point_t lazyMonsterLocation {{6, 6}}; // this monster will always try to eat the closest people
    const std::size_t monsterHeads = 2; // this monster can eat two people at once
    auto lazyMonsterVictims = tree.searchKnn(lazyMonsterLocation, monsterHeads);
    for (const auto& victim : lazyMonsterVictims)
    {
        std::cout << victim.payload << " closest to lazy monster, with distance " << sqrt(victim.distance) << "!"
                  << std::endl;
    }

    // ball search
    point_t stationaryMonsterLocation {{8, 8}}; // this monster doesn't move, so can only eat people that are close
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

    auto bruteforceKNN = [&](const std::array<double, dims>& searchLoc, size_t K) -> std::vector<std::pair<double, int>>
    {
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
        = [&](const std::array<double, dims>& searchLoc, double radius) -> std::vector<std::pair<double, int>>
    {
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

    auto randomPoint = []()
    {
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
    auto randomPoint = []()
    {
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

    for (int i = 0; i < 5000; i++)
    {
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

void iteratorTest()
{
    std::cout << "Iterator tests started" << std::endl;

    // GIVEN: an empty tree
    static const int dims = 3;
    using tree_t = jk::tree::KDTree<int, dims>;
    tree_t tree;

    // THEN: iterator should be at end
    if (tree.begin() != tree.end())
    {
        std::cout << "Empty tree begin != end" << std::endl;
    }

    // WHEN: adding points
    std::vector<std::array<double, dims>> points;
    for (int i = 0; i < 1000; i++)
    {
        std::array<double, dims> p;
        for (int d = 0; d < dims; d++)
            p[d] = drand();
        tree.addPoint(p, i, false);
        points.push_back(p);
    }

    // THEN: we should be able to iterate over all of them (unsplit)
    int count = 0;
    for (auto it = tree.begin(); it != tree.end(); ++it)
    {
        count++;
    }
    if (count != 1000)
    {
        std::cout << "Unsplit tree iterator count mismatch: " << count << " != 1000" << std::endl;
    }

    // WHEN: splitting the tree
    tree.splitOutstanding();

    // THEN: we should still be able to iterate over all of them (split)
    count = 0;
    std::vector<int> foundPayloads;
    for (const auto& lp : tree)
    {
        count++;
        foundPayloads.push_back(lp.payload);
    }
    if (count != 1000)
    {
        std::cout << "Split tree iterator count mismatch: " << count << " != 1000" << std::endl;
    }

    std::sort(foundPayloads.begin(), foundPayloads.end());
    for (int i = 0; i < 1000; i++)
    {
        if (foundPayloads[i] != i)
        {
            std::cout << "Payload mismatch at " << i << ": " << foundPayloads[i] << " != " << i << std::endl;
            break;
        }
    }

    // WHEN: adding more points to cause more splits
    for (int i = 1000; i < 2000; i++)
    {
        std::array<double, dims> p;
        for (int d = 0; d < dims; d++)
            p[d] = drand();
        tree.addPoint(p, i, true);
    }

    // THEN: iterator still works
    count = 0;
    for (auto& lp : tree)
    {
        count++;
    }
    if (count != 2000)
    {
        std::cout << "Post-autosplit tree iterator count mismatch: " << count << " != 2000" << std::endl;
    }

    std::cout << "Iterator tests completed" << std::endl;
}

void rebalanceTest()
{
    std::cout << "Rebalance tests started" << std::endl;

    // GIVEN: a tree built with many dynamic insertions
    static const int dims = 2;
    using tree_t = jk::tree::KDTree<int, dims>;
    tree_t tree;

    for (int i = 0; i < 2000; i++)
    {
        std::array<double, dims> p = {{drand(), drand()}};
        tree.addPoint(p, i, true);
    }

    size_t sizeBefore = tree.size();

    // WHEN: rebalancing the tree
    tree.rebalance();

    // THEN: size should be the same
    if (tree.size() != sizeBefore)
    {
        std::cout << "Size mismatch after rebalance: " << tree.size() << " != " << sizeBefore << std::endl;
    }

    // AND: all points should still be there
    std::vector<int> foundPayloads;
    for (const auto& lp : tree)
    {
        foundPayloads.push_back(lp.payload);
    }
    std::sort(foundPayloads.begin(), foundPayloads.end());
    for (int i = 0; i < 2000; i++)
    {
        if (foundPayloads[i] != i)
        {
            std::cout << "Payload mismatch after rebalance at " << i << std::endl;
            break;
        }
    }

    std::cout << "Rebalance tests completed" << std::endl;
}

void eightDTest()
{
    std::cout << "8D tests started" << std::endl;
    static const int dims = 8;
    using tree_t = jk::tree::KDTree<int, dims>;
    tree_t tree;

    std::srand(123);
    for (int i = 0; i < 1000; i++)
    {
        std::array<double, dims> p;
        for (int d = 0; d < dims; d++)
            p[d] = drand();
        tree.addPoint(p, i);
    }

    for (int i = 0; i < 100; i++)
    {
        std::array<double, dims> p;
        for (int d = 0; d < dims; d++)
            p[d] = drand();
        auto results = tree.searchKnn(p, 5);
        if (results.size() != 5)
        {
            std::cout << "8D search failed to find 5 neighbors" << std::endl;
        }
    }
    std::cout << "8D tests completed" << std::endl;
}

void removalTest()
{
    std::cout << "Removal tests started" << std::endl;
    static const int dims = 2;
    using tree_t = jk::tree::KDTree<int, dims>;
    tree_t tree;

    std::vector<std::array<double, dims>> points;
    for (int i = 0; i < 1000; i++)
    {
        std::array<double, dims> p = {{drand(), drand()}};
        tree.addPoint(p, i);
        points.push_back(p);
    }

    // Remove half of the points
    for (int i = 0; i < 500; i++)
    {
        if (!tree.removePoint(points[i], i))
        {
            std::cout << "Failed to remove point " << i << std::endl;
        }
    }

    if (tree.size() != 500)
    {
        std::cout << "Size mismatch after removal: " << tree.size() << " != 500" << std::endl;
    }

    // Verify remaining points
    int count = 0;
    for (const auto& lp : tree)
    {
        count++;
        if (lp.payload < 500)
        {
            std::cout << "Removed point still present in iterator: " << lp.payload << std::endl;
        }
    }
    if (count != 500)
    {
        std::cout << "Iterator count mismatch after removal: " << count << " != 500" << std::endl;
    }

    // Verify searches don't find removed points
    for (int i = 0; i < 500; i++)
    {
        auto results = tree.searchKnn(points[i], 1);
        if (!results.empty() && results[0].payload == i && results[0].distance < 1e-10)
        {
            std::cout << "Search found removed point " << i << std::endl;
        }
    }

    // Rebalance and verify
    tree.rebalance();
    if (tree.size() != 500)
    {
        std::cout << "Size mismatch after rebalance: " << tree.size() << " != 500" << std::endl;
    }

    std::cout << "Removal tests completed" << std::endl;
}

void l1DistanceTest()
{
    std::cout << "L1 Distance tests started" << std::endl;
    static const int dims = 3;
    using tree_t = jk::tree::KDTree<int, dims, 32, jk::tree::L1>;
    tree_t tree;

    std::vector<std::array<double, dims>> points;
    std::srand(12345);

    auto bruteForceL1 = [&](const std::array<double, dims>& loc, std::size_t k)
    {
        std::vector<std::pair<double, int>> dists;
        for (std::size_t i = 0; i < points.size(); ++i)
        {
            double d = 0;
            for (int j = 0; j < dims; ++j)
                d += std::abs(loc[j] - points[i][j]);
            dists.push_back({d, (int)i});
        }
        std::sort(dists.begin(), dists.end());
        if (dists.size() > k)
            dists.resize(k);
        return dists;
    };

    for (int i = 0; i < 1000; i++)
    {
        std::array<double, dims> p = {{drand(), drand(), drand()}};
        tree.addPoint(p, i);
        points.push_back(p);
    }

    for (int i = 0; i < 100; i++)
    {
        std::array<double, dims> loc = {{drand(), drand(), drand()}};
        auto knn = tree.searchKnn(loc, 10);
        auto brute = bruteForceL1(loc, 10);

        if (knn.size() != brute.size())
        {
            std::cout << "L1 KNN size mismatch" << std::endl;
            continue;
        }

        for (std::size_t j = 0; j < knn.size(); ++j)
        {
            if (std::abs(knn[j].distance - brute[j].first) > 1e-10)
            {
                std::cout << "L1 distance mismatch at " << j << ": " << knn[j].distance << " != " << brute[j].first
                          << std::endl;
            }
        }
    }

    std::cout << "L1 Distance tests completed" << std::endl;
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

    auto randomPoint = []()
    {
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
