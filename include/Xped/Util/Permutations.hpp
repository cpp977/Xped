#ifndef XPED_PERMUTATIONS_HPP_
#define XPED_PERMUTATIONS_HPP_

#include <algorithm>
#include <array>
#include <cassert>
#include <fstream>
#include <iostream>
#include <numeric>
#include <set>
#include <vector>

#include <boost/algorithm/string.hpp>
#include <boost/functional/hash.hpp>

namespace Xped::util {

struct Permutation;

struct Transposition
{
    Transposition(){};
    Transposition(const std::size_t source_in, const std::size_t target_in)
        : source(source_in)
        , target(target_in){};
    Transposition(const std::array<std::size_t, 2> data)
        : source(data[0])
        , target(data[1]){};
    std::string print() const
    {
        std::stringstream ss;
        ss << source << " <===> " << target;
        return ss.str();
    }
    std::size_t source = 0;
    std::size_t target = 0;
};

struct Permutation
{
    typedef std::vector<std::size_t> Cycle;

    Permutation(){};

    template <typename Container>
    Permutation(const Container& in)
    {
        pi.resize(in.size());
        std::copy(in.begin(), in.end(), pi.begin());
        initialize();
    }

    Permutation(const std::string filename)
    {
        std::ifstream stream(filename, std::ios::in);
        std::string line;
        if(stream.is_open()) {
            while(std::getline(stream, line)) {
                std::vector<std::string> results;

                boost::split(results, line, [](char c) { return c == '\t'; });
                if(results[0].find("#") != std::string::npos) { continue; } // skip lines with a hashtag
                int source = stoi(results[0]);
                int target = stoi(results[1]);
                assert(source >= 0 and "Invalid permutation data in file.");
                assert(target >= 0 and "Invalid permutation data in file.");
                if(source >= pi.size()) { pi.resize(source + 1); }
                pi[source] = target;
            }
            stream.close();
        }
        N = pi.size();

        // consistency check
        std::set<std::size_t> bijectivePermutation;
        for(std::size_t i = 0; i < N; i++) {
            assert(pi[i] < N and "Invalid permutation data in file.");
            auto it = bijectivePermutation.find(pi[i]);
            if(it == bijectivePermutation.end()) {
                bijectivePermutation.insert(pi[i]);
            } else {
                assert(false and "Invalid permutation data in file.");
            }
        }
        initialize();
    };

    void initialize()
    {
        N = pi.size();
        pi_inv.resize(N);
        for(std::size_t i = 0; i < N; i++) { pi_inv[pi[i]] = i; }

        std::vector<bool> visited(N, false);
        while(!std::all_of(visited.cbegin(), visited.cend(), [](bool b) { return b; })) {
            auto start_it = std::find(visited.cbegin(), visited.cend(), false);
            std::size_t start = std::distance(visited.cbegin(), start_it);

            visited[pi_inv[start]] = true;

            Cycle cycle;
            cycle.push_back(start);
            auto tmp = start;
            for(std::size_t i = 0; i < N; i++) {
                auto next_power = pi[tmp];
                if(next_power == start) {
                    continue;
                } else {
                    cycle.push_back(next_power);
                    visited[pi_inv[next_power]] = true;
                    tmp = next_power;
                }
            }
            cycles.push_back(cycle);
        }
    }

    friend std::size_t hash_value(const Permutation& p)
    {
        std::size_t seed = 0;
        boost::hash_combine(seed, p.pi);
        return seed;
    }

    bool operator==(const Permutation& other) const { return pi == other.pi; }

    static Permutation Identity(std::size_t N)
    {
        std::vector<std::size_t> id(N);
        std::iota(id.begin(), id.end(), 0ul);
        return Permutation(id);
    }

    static std::vector<Permutation> all(std::size_t N)
    {
        std::vector<std::size_t> data(N);
        std::iota(data.begin(), data.end(), 0ul);
        std::vector<Permutation> out;
        do {
            Permutation p(data);
            out.push_back(p);
        } while(std::next_permutation(data.begin(), data.end()));
        return out;
    }

    std::string print() const
    {
        std::stringstream ss;
        for(const auto& c : cycles) {
            ss << "(";
            for(std::size_t i = 0; i < c.size(); i++) {
                if((i < c.size() - 1)) {
                    ss << c[i] << ",";
                } else {
                    ss << c[i];
                }
            }
            ss << ")";
        }
        ss << std::endl;
        size_t count = 0;
        for(const auto& elem : pi) {
            ss << count << " ==> " << elem << std::endl;
            count++;
        }
        return ss.str();
    };

    std::vector<Transposition> transpositions() const
    {
        std::vector<Transposition> out;
        for(const auto& c : cycles) {
            for(std::size_t i = 0; i < c.size() - 1; i++) {
                Transposition t(c[0], c[c.size() - 1 - i]);
                out.push_back(t);
            }
        }
        return out;
    };

    std::vector<std::vector<Transposition>> independentTranspositions() const
    {
        std::vector<std::vector<Transposition>> out(cycles.size());
        std::size_t count = 0;
        for(const auto& c : cycles) {
            for(std::size_t i = 0; i < c.size() - 1; i++) {
                Transposition t(c[0], c[c.size() - 1 - i]);
                out[count].push_back(t);
            }
            count++;
        }
        return out;
    };

    std::vector<std::size_t> decompose() const
    {
        std::vector<bool> visited(N, false);
        std::vector<std::size_t> a(N);
        std::iota(a.begin(), a.end(), 0ul);

        std::vector<std::size_t> out;
        if(a == pi) { return out; } // early return if it is the identity permutation

        for(std::size_t i = 0; i < N; i++) {
            auto index = std::distance(a.begin(), std::find(a.begin(), a.end(), pi[i]));
            // std::cout << "i=" << i << ", pi[i]=" << pi[i] << ", index=" << index << ", a="; for (const auto& o:a) {std::cout << o << " ";}
            // std::cout << std::endl;
            if(index == 0) { continue; }
            if(i == 0) {
                for(std::size_t j = index - 1; j < index; --j) {
                    // std::cout << "j=" << j << std::endl;
                    std::swap(a[j], a[j + 1]);
                    out.push_back(j);
                }
            } else {
                for(std::size_t j = index - 1; j >= i; --j) {
                    // std::cout << "j=" << j << std::endl;
                    std::swap(a[j], a[j + 1]);
                    out.push_back(j);
                }
            }
        }
        assert(a == pi and "Error when determining the swaps of a permutation.");
        return out;
    }

    std::size_t parity()
    {
        std::size_t out = 0;
        for(const auto& c : cycles) { out += (c.size() - 1) % 2; }
        return out % 2;
    }

    template <typename Container>
    void apply(Container& c) const
    {
        if(c.size() == 1) { return; }
        Container tmp(c);
        for(std::size_t i = 0; i < c.size(); i++) { tmp[i] = c[pi[i]]; }
        c = tmp;
    }

    Permutation inverse() const
    {
        Permutation out(pi_inv);
        return out;
    }

    template <typename IndexType>
    std::vector<IndexType> pi_as_index() const
    {
        std::vector<IndexType> out(N);
        std::copy(pi.begin(), pi.end(), out.begin());
        return out;
    }

    std::size_t N;
    std::vector<std::size_t> pi;
    std::vector<std::size_t> pi_inv;
    std::vector<Cycle> cycles;
};

} // namespace Xped::util
#endif
