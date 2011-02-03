#include <iostream>
#include <set>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/ptr_container/ptr_map.hpp>
#include <boost/multi_array.hpp>
#include <boost/tokenizer.hpp>
#include <boost/bind.hpp>
#include <boost/foreach.hpp>

#include "similarity.hpp"

using namespace std;
using namespace boost;


static const size_t MAX_RELATED_ITEMS=10;

typedef long long item_type;

static set<item_type>* readline(const string& line) {
  char_separator<char> sep(",");
  tokenizer<char_separator<char> > tok(line,sep);
  set<item_type>* user(new set<item_type>);

  /* TODO?: error handling */
  for (tokenizer<char_separator<char> >::iterator it = ++(tok.begin());
	   it != tok.end(); ++it) {
	user->insert(atoll((*it).c_str()));
  }
  return user;
}

void transform_prefs(const ptr_vector<set<item_type> >& users,
							  ptr_map<item_type, set<int> >&  items) {
  int user = 0;
  BOOST_FOREACH(set<item_type> user_info, users){
	BOOST_FOREACH(item_type item_id, user_info){
	  ptr_map<item_type, set<int> >::iterator item_iter
		= items.find(item_id);
	  if (item_iter != items.end()) {
		(item_iter->second)->insert(user);
	  } else {
		set<int>* item(new set<int>);
		item->insert(user);
		items.insert(item_id, item);
	  }
	}
	++user;
  }

}

const double NOT_SCORED = -100;

void top_matches(const ptr_map<item_type, set<int> >&  items,
				 ptr_map<item_type, vector<pair<item_type, double> > >&  item_scores,
				 const size_t max_items) {

  multi_array<double, 2> memo(extents[items.size()][items.size()]);

  for(size_t i = 0; i < items.size(); ++i) {
	for(size_t j = 0; j < items.size(); ++j) {
	  memo[i][j] = NOT_SCORED;
	}
  }


  typedef ptr_container_detail::ref_pair<item_type, const set<int>* const> item_info;

  size_t i = 0;
  BOOST_FOREACH(item_info item1, items) {
	vector< pair<item_type, double> >* scores(new vector<pair<item_type, double> >);

	// 1st arg of ptr_map::insert must not be 'const'.
	item_type item1_id = item1.first;
	item_scores.insert(item1_id, scores);

	size_t j = 0;
	BOOST_FOREACH(item_info item2, items) {
	  if (item1.first == item2.first) {
		++j;
		continue;
	  }

	  double memo_score = memo[j][i];
	  double score;
	  if (memo_score > 0) {
		score = memo_score;
	  } else {
		score = similarity::tanimoto(item1.second, item2.second);
		memo[i][j] = score;
	  }
	  if (score > 0) {
		scores->push_back(pair<item_type, double>(item2.first, score));
	  }
	  ++j;
	}

	if (scores->size() > max_items) {
	  partial_sort(scores->begin(),
				   scores->begin() + max_items,
				   scores->end(),
				   bind(greater<double>(),
						bind(&pair<item_type,double>::second, _1),
						bind(&pair<item_type,double>::second, _2)));


	  scores->erase(scores->begin() + max_items, scores->end());
	} else {
	  sort(scores->begin(), scores->end(),
		   bind(greater<double>(),
				bind(&pair<item_type,double>::second, _1),
				bind(&pair<item_type,double>::second, _2)));
	}
	++i;
  }
}

void printout_scores(const ptr_map<item_type, vector<pair<item_type, double> > >&  item_scores) {

  typedef boost::ptr_container_detail::ref_pair<item_type, const std::vector<std::pair<item_type, double>, std::allocator<std::pair<item_type, double> > >* const> score_info;

  BOOST_FOREACH(score_info score_details, item_scores) {
	cout << score_details.first << ':';
	const vector<pair<item_type, double> >* const scores = score_details.second;

	typedef pair<item_type, double> score;
	BOOST_FOREACH(score score_detail, *scores) {
	  cout << score_detail.second << '|';
	  cout << score_detail.first << ',';
	}
	cout << endl;
  }
}

int main(void) {

  ptr_vector<set<item_type> > users;
  ptr_map<item_type, set<int> >  items;
  ptr_map<item_type, vector<pair<item_type, double> > >  item_scores;


  /* read data */
  string line;
  while(getline(cin,line)) {
	users.push_back(readline(line));
  }

  /* transformPrefs */
  transform_prefs(users, items);

  /* topMatches */
  top_matches(items, item_scores, MAX_RELATED_ITEMS);

  //cout << showpoint ;
  /* printout results */
  printout_scores(item_scores);

  return 0;
}
