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

#include "similarity.hpp"
#include "csv.hpp"

using namespace std;
using namespace boost;

static const size_t MAX_RELATED =10;

const double NOT_SCORED = -100;

typedef long item_type;

void top_matches(const map<item_type, set<size_t> >&  items,
                 const size_t max_items) {

    multi_array<double, 2> memo(extents[items.size()][items.size()]);

    for(size_t i = 0; i < items.size(); ++i) {
        for(size_t j = 0; j < items.size(); ++j) {
            memo[i][j] = NOT_SCORED;
        }
    }
   typedef pair<item_type, set<size_t> > item;

   size_t i = 0;
   BOOST_FOREACH(item item1, items) {
       vector< pair<item_type, double> > scores;

       size_t j = 0;
       BOOST_FOREACH(item item2, items) {
           if (i == j) {
               ++j;
               continue;
           }

           double memo_score = memo[j][i];
           double score;
           if (memo_score == NOT_SCORED) {
               score = similarity::tanimoto(item1.second, item2.second);
               memo[i][j] = score;
           } else {
               score = memo_score;
           }
           if (score > 0) {
               scores.push_back(pair<item_type, double>(item2.first, score));
           }
           ++j;
       }

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

       cout << item1.first << ':';
       typedef pair<item_type, double> score;
       BOOST_FOREACH(score score_detail, scores) {
           cout << score_detail.second << '|';
           cout << score_detail.first << ',';
       }
       cout << endl;
       ++i;
   }
}

static map<item_type, set<size_t> > transform_prefs(const vector<vector<item_type> >& user_items) {
    size_t user = 0;

    map<item_type, set<size_t> > items;

    BOOST_FOREACH(vector<item_type> item_vector, user_items){
        BOOST_FOREACH(item_type item_id, item_vector){
            map<item_type, set<size_t> >::iterator item_iter
                = items.find(item_id);
            if (item_iter != items.end()) {
                (item_iter->second).insert(user);
            } else {
                set<size_t> item_set;
                item_set.insert(user);
                items[item_id] = item_set;
            }
        }
        ++user;
    }
    return items;

}

struct str2long {
    long operator()(string s) {
        return atol(s.c_str());
    }
};

int main(void) {

    vector<vector<item_type> > user_items;
    map<item_type, set<size_t> > items;

    /* read data */
    string line;
    while(getline(cin,line)) {
        vector<string> row = (csv::parse_line(line));
        vector<item_type> item_vector;
        //copy(row.begin() + 1, row.end(), inserter(items, items.begin()));
        transform(row.begin() + 1, row.end(), inserter(item_vector, item_vector.begin()), str2long());
        user_items.push_back(item_vector);
    }

    items = transform_prefs(user_items);


    //    typedef pair<string, set<string> > user_item;
    //    BOOST_FOREACH(user_item u, user_items) {
    //        cout << u.first << endl;
    //        cout << u.second[1] << endl;

    //    }



    //    /* transformPrefs */
    //    transform_prefs(users, items);

    /* topMatches */
    top_matches(items, MAX_RELATED);

    //    cout << showpoint ;
    //    /* printout results */
    //    printout_scores(item_scores);

    return 0;
}
