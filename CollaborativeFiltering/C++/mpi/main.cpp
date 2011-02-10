#include <iostream>
#include <set>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/ptr_container/ptr_map.hpp>
#include <boost/multi_array.hpp>
#include <boost/bind.hpp>
#include <boost/foreach.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>
#include <cstdlib>
#include <boost/mpi.hpp>
#include <boost/serialization/set.hpp>
#include <boost/ptr_container/serialize_ptr_map.hpp>

#include "similarity.hpp"
#include "csv.hpp"

using namespace std;
using namespace boost;

static const size_t MAX_RELATED =10;

const double NOT_SCORED = -100;

typedef long item_type;
typedef ptr_vector<vector<item_type> > user_items;
typedef ptr_map<item_type, set<size_t> > item_users;
typedef ptr_container_detail::ref_pair<item_type, const set<size_t>* const> item_info;

void printout_score(item_type item_id, const vector<pair<item_type, double> >&  scores) {

    cout << item_id << ':';
    typedef pair<item_type, double> score;
    BOOST_FOREACH(score score_detail, scores) {
        cout << score_detail.second << '|';
        cout << score_detail.first << ',';
    }
    cout << endl;
}


void top_matches(const item_type item1_id, const ptr_map<item_type, set<size_t> >&  items,
				 const size_t max_items) {


    vector< pair<item_type, double> > scores;

    const set<size_t>* item1_set = items.find(item1_id)->second;

    BOOST_FOREACH(item_info item2, items) {
        if (item1_id == item2.first) {
            continue;
        }

        double score = similarity::tanimoto(item1_set, item2.second);
        if (score > 0) {
            scores.push_back(pair<item_type, double>(item2.first, score));
        }
    }

    if (!scores.empty()) {

        if (scores.size() > max_items) {
            partial_sort(scores.begin(),
                    scores.begin() + max_items,
                    scores.end(),
                    bind(greater<double>(),
                        bind(&pair<item_type,double>::second, _1),
                        bind(&pair<item_type,double>::second, _2)));


            scores.erase(scores.begin() + max_items, scores.end());
        } else {
            sort(scores.begin(), scores.end(),
                    bind(greater<double>(),
                        bind(&pair<item_type,double>::second, _1),
                        bind(&pair<item_type,double>::second, _2)));
        }
        printout_score(item1_id, scores);
    }
}




static void transform_prefs(const user_items& users,
        item_users& items,
        set<item_type>& item_set) {
    size_t user = 0;

    BOOST_FOREACH(vector<item_type> item_vector, users) {
        BOOST_FOREACH(item_type item_id, item_vector){
            item_set.insert(item_id);
            ptr_map<item_type, set<size_t> >::iterator item_iter
                = items.find(item_id);
            if (item_iter != items.end()) {
                (item_iter->second)->insert(user);
            } else {
                set<size_t>* item_set(new set<size_t>);
                item_set->insert(user);
                items.insert(item_id, item_set);
            }
        }
        ++user;
    }
}

struct str2long {
    long operator()(string s) {
        return atol(s.c_str());
    }
};

int main(int argc, char** argv) {

    user_items users;
    item_users items;
    set<item_type> item_set;

    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;


    if (world.rank() == 0) {
        /* read data */
        string line;
        while(getline(cin,line)) {
            vector<string> row = csv::parse_line(line);
            vector<item_type>* item_vector(new vector<item_type>);
            transform(row.begin() + 1, row.end(), inserter(*item_vector, item_vector->begin()), str2long());
            item_vector->erase(remove(item_vector->begin(), item_vector->end(), 0), item_vector->end());
            users.push_back(item_vector);
        }

        transform_prefs(users, items, item_set);


    }

    mpi::broadcast(world, items, 0);

//    cerr << world.rank() << " " << items.size() << endl;


    if (world.rank() == 0) {
        vector<item_type> item_vector;
        copy(item_set.begin(), item_set.end(), inserter(item_vector, item_vector.begin()));
        size_t index = 0;
        size_t item_size = item_vector.size();
        size_t child_size = world.size() -1 ;
        while(true) {

//            cerr << index << ' ' << child_size << ' ' << item_size << endl;

            if (index >= item_size) {
                break;
            }

            size_t size;
            if (index + child_size <= item_size) {
                size = child_size;
            } else {
                size = item_size - index;
            }

            for(size_t i = 1; i <= size; ++i) {
                world.send(i, 0, item_vector[index + i - 1]);
            }

            index += child_size;

        }

        item_type fin = 0;
        for(size_t i = 1; i <= child_size; ++i) {
            world.send(i, 0, fin);
        }


    } else {
        while(true) {
            item_type item_id;
            world.recv(0, 0, item_id);

            if (item_id == 0) {
                break;
            }

            top_matches(item_id, items, MAX_RELATED);
        }


    }

    return 0;
}
