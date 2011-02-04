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

void top_matches(const map<string, set<item_type> >&  user_items,
                 const size_t max_items) {

    multi_array<double, 2> memo(extents[user_items.size()][user_items.size()]);

    for(size_t i = 0; i < user_items.size(); ++i) {
        for(size_t j = 0; j < user_items.size(); ++j) {
            memo[i][j] = NOT_SCORED;
        }
    }
   typedef pair<string, set<item_type> > user_item;

   size_t i = 0;
   BOOST_FOREACH(user_item user1, user_items) {
       vector< pair<string, double> > scores;

       size_t j = 0;
       BOOST_FOREACH(user_item user2, user_items) {
           if (i == j) {
               ++j;
               continue;
           }

           double memo_score = memo[j][i];
           double score;
           if (memo_score == NOT_SCORED) {
               score = similarity::tanimoto(user1.second, user2.second);
               memo[i][j] = score;
           } else {
               score = memo_score;
           }
           if (score > 0) {
               scores.push_back(pair<string, double>(user2.first, score));
           }
           ++j;
       }

       if (scores.size() > max_items) {
           partial_sort(scores.begin(),
                   scores.begin() + max_items,
                   scores.end(),
                   bind(greater<double>(),
                       bind(&pair<string,double>::second, _1),
                       bind(&pair<string,double>::second, _2)));


           scores.erase(scores.begin() + max_items, scores.end());
       } else {
           sort(scores.begin(), scores.end(),
                   bind(greater<double>(),
                       bind(&pair<string,double>::second, _1),
                       bind(&pair<string,double>::second, _2)));
       }

       cout << user1.first << ':';
       typedef pair<string, double> score;
       BOOST_FOREACH(score score_detail, scores) {
           cout << score_detail.second << '|';
           cout << score_detail.first << ',';
       }
       cout << endl;
       ++i;
   }
}

//void printout_scores(const ptr_map<item_type, vector<pair<item_type, double> > >&  item_scores) {

//  typedef boost::ptr_container_detail::ref_pair<item_type, const std::vector<std::pair<item_type, double>, std::allocator<std::pair<item_type, double> > >* const> score_info;

//  BOOST_FOREACH(score_info score_details, item_scores) {
//    cout << score_details.first << ':';
//    const vector<pair<item_type, double> >* const scores = score_details.second;

//    typedef pair<item_type, double> score;
//    BOOST_FOREACH(score score_detail, *scores) {
//      cout << score_detail.second << '|';
//      cout << score_detail.first << ',';
//    }
//    cout << endl;
//  }
//}
//
//
struct str2long {
    long operator()(string s) {
        return atol(s.c_str());
    }
};

int main(void) {

    map<string, set<item_type> > user_items;

    /* read data */
    string line;
    while(getline(cin,line)) {
        vector<string> row = (csv::parse_line(line));
        string user = row[0];
        set<item_type> items;
        //copy(row.begin() + 1, row.end(), inserter(items, items.begin()));
        transform(row.begin() + 1, row.end(), inserter(items, items.begin()), str2long());
        user_items[user] = items;
    }


    //    typedef pair<string, set<string> > user_item;
    //    BOOST_FOREACH(user_item u, user_items) {
    //        cout << u.first << endl;
    //        cout << u.second[1] << endl;

    //    }



    //    /* transformPrefs */
    //    transform_prefs(users, items);

    /* topMatches */
    top_matches(user_items, MAX_RELATED);

    //    cout << showpoint ;
    //    /* printout results */
    //    printout_scores(item_scores);

    return 0;
}
