#include <iostream>
#include <set>
#include <vector>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/ptr_container/ptr_map.hpp>
#include <boost/multi_array.hpp>
#include <boost/bind.hpp>
#include <boost/foreach.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>
#include <cstdlib>

#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>

#include "csv.hpp"

using namespace std;
using namespace boost;
using namespace boost::numeric::ublas;
static const size_t MAX_RELATED =10;

static const double NOT_SCORED = -100;

typedef unsigned long item_type;

typedef ptr_vector<std::vector<item_type> > user_items;
typedef mapped_matrix<int> item_users;


static double tanimoto(const matrix_row<mapped_matrix<int> >& r1, const matrix_row<mapped_matrix<int> >& r2, const size_t r1_size, const size_t r2_size) {

    int intersection = prec_inner_prod(r1, r2);

    if (intersection == 0) {
        return 0.0;
    }

    return ((double)intersection)/(r1_size + r2_size - intersection);

}

static void printout_score(item_type item_id, const std::vector<pair<item_type, double> >&  scores) {

    cout << item_id << ':';
    typedef pair<item_type, double> score;
    BOOST_FOREACH(score score_detail, scores) {
        cout << score_detail.second << '|';
        cout << score_detail.first << ',';
    }
    cout << endl;
}


static void top_matches(const map<item_type, size_t>&  items,
        item_users& m,
        const size_t max_items) {

    typedef map<item_type, size_t>::value_type item_info;

    BOOST_FOREACH(item_info item1, items) {
        std::vector< pair<item_type, double> > scores;

        matrix_row<item_users> mr1(m, item1.first);

        BOOST_FOREACH(item_info item2, items) {
            if (item1.first == item2.first) {
                continue;
            }

            matrix_row<item_users> mr2(m, item2.first);

            double score = tanimoto(mr1, mr2,
                    item1.second, item2.second);

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

            printout_score(item1.first, scores);
        }
    }
}




static void transform_prefs(const user_items& users,
        map<item_type, size_t>& items, item_users& m) {
    size_t user = 0;

    BOOST_FOREACH(std::vector<item_type> item_vector, users) {
        BOOST_FOREACH(item_type item_id, item_vector){
            map<item_type, size_t>::iterator iter = items.find(item_id);

            if (iter != items.end()) {
                ++items[item_id];
            } else {
                items[item_id] = 1;
            }
            m(item_id, user) = 1;
        }
        ++user;
    }
}


struct str2item_type {
    long operator()(string s) {
        return atol(s.c_str());
    }
};


int main(void) {

    user_items users;
    map<item_type, size_t> items;


    /* read data */
    string line;
    while(getline(cin,line)) {
        std::vector<string> row = csv::parse_line(line);
        std::vector<item_type>* item_vector(new std::vector<item_type>);
        transform(row.begin() + 1, row.end(), inserter(*item_vector, item_vector->begin()), str2item_type());
        item_vector->erase(remove(item_vector->begin(), item_vector->end(), 0), item_vector->end());
        users.push_back(item_vector);
    }

    item_users m(30000000000000L, users.size());

    transform_prefs(users, items, m);

    top_matches(items, m, MAX_RELATED);
    return 0;
}
