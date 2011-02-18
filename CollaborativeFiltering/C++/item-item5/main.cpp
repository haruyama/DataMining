#include <iostream>
#include <vector>
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
typedef ptr_vector<vector<item_type> > user_items;
typedef ptr_map<item_type, vector<size_t> > item_users;

void printout_score(item_type item_id, const vector<pair<item_type, double> >&  scores) {

    cout << item_id << ':';
    typedef pair<item_type, double> score;
    BOOST_FOREACH(score score_detail, scores) {
        cout << score_detail.second << '|';
        cout << score_detail.first << ',';
    }
    cout << endl;
}


void top_matches(const ptr_map<item_type, vector<int> >&  items,
				 const size_t max_items) {
  typedef ptr_container_detail::ref_pair<item_type, const vector<int>* const> item_info;

  BOOST_FOREACH(item_info item1, items) {
	vector< pair<item_type, double> > scores;

	BOOST_FOREACH(item_info item2, items) {
	  if (item1.first == item2.first) {
		continue;
	  }

	  double score = similarity::tanimoto(item1.second, item2.second);
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
        item_users& items) {
    size_t user = 0;

    BOOST_FOREACH(vector<item_type> item_vector, users) {
        BOOST_FOREACH(item_type item_id, item_vector){
            ptr_map<item_type, vector<size_t> >::iterator item_iter
                = items.find(item_id);
            if (item_iter != items.end()) {
                (item_iter->second)->push_back(user);
            } else {
                vector<size_t>* item_set(new vector<size_t>);
                item_set->push_back(user);
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

int main(void) {

    user_items users;
    item_users items;

    /* read data */
    string line;
    while(getline(cin,line)) {
        vector<string> row = csv::parse_line(line);
        vector<item_type>* item_vector(new vector<item_type>);
        transform(row.begin() + 1, row.end(), inserter(*item_vector, item_vector->begin()), str2long());
        item_vector->erase(remove(item_vector->begin(), item_vector->end(), 0), item_vector->end());
        users.push_back(item_vector);
    }


    transform_prefs(users, items);

    //    typedef pair<string, set<string> > user_item;
    //    BOOST_FOREACH(user_item u, user_items) {
    //        cout << u.first << endl;
    //        cout << u.second[1] << endl;

    //    }



    //    /* transformPrefs */
    //    transform_prefs(users, items);

    /* topMatches */
    top_matches(items, MAX_RELATED);
    return 0;
}
