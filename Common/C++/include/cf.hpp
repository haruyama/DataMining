#pragma once

#include <boost/multi_array.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/ptr_container/ptr_map.hpp>
#include <iostream>
#include <set>
#include <boost/bind.hpp>
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include "similarity.hpp"

using namespace std;
using namespace boost;

namespace cf {

    template<typename IT>
    void printout_score(IT item_id, const vector<pair<IT, double> >&  scores, size_t max_items) {

        cout << item_id << ':';
        typedef pair<IT, double> score;
        BOOST_FOREACH(score score_detail, scores) {
            cout << score_detail.second << '|';
            cout << score_detail.first << ',';
        }
        cout << endl;
    }

    template<typename IT, typename UT>
    void top_matches(const ptr_map<IT, set<UT> >&  items,
            const size_t max_items, const int cache) {

        typedef boost::shared_ptr<multi_array<double, 2> > cache_array_ptr;
        
        cache_array_ptr memo;

        if (cache) {
            memo = cache_array_ptr(new multi_array<double, 2>(extents[items.size()][items.size()]));
        }
        

        typedef ptr_container_detail::ref_pair<IT, const set<UT>* const> item_info;

        size_t i = 0;
        BOOST_FOREACH(item_info item1, items) {
            vector< pair<IT, double> > scores;
            size_t item1_size = item1.second->size();

            size_t j = 0;
            BOOST_FOREACH(item_info item2, items) {
                //if (item1.first == item2.first) {
                double score;
                if (i == j) {
                    ++j;
                    continue;
                } else if (cache && i > j) {
                    score = (*memo)[j][i];
                } else {
                    score = similarity::tanimoto(item1.second, item2.second, item1_size);
                    if (cache) {
                        (*memo)[i][j] = score;
                    }
                }

                if (score > 0) {
                    scores.push_back(pair<IT, double>(item2.first, score));
                }
                ++j;
            }

            if (scores.size() > max_items) {
                partial_sort(scores.begin(),
                        scores.begin() + max_items,
                        scores.end(),
                        bind(greater<double>(),
                            bind(&pair<IT,double>::second, _1),
                            bind(&pair<IT,double>::second, _2)));

                scores.erase(scores.begin() + max_items, scores.end());
            } else {
                sort(scores.begin(), scores.end(),
                        bind(greater<double>(),
                            bind(&pair<IT,double>::second, _1),
                            bind(&pair<IT,double>::second, _2)));
            }
            if (!scores.empty()) {
                printout_score(item1.first, scores, max_items);
            }
            ++i;
        }
    }


    template<typename IT, typename UT>
    void transform_prefs(const ptr_vector<vector<IT> >& users,
            ptr_map<IT, set<UT> >& items) {
        size_t user = 0;

        BOOST_FOREACH(vector<IT> item_vector, users) {
            BOOST_FOREACH(IT item_id, item_vector){
                typename ptr_map<IT, set<UT> >::iterator 
                    item_iter = items.find(item_id);

                if (item_iter != items.end()) {
                    (item_iter->second)->insert(user);
                } else {
                    set<UT>* item_set(new set<UT>);
                    item_set->insert(user);
                    items.insert(item_id, item_set);
                }
            }
            ++user;
        }
    }
}
