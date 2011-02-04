#include <vector>
#include <boost/test/included/unit_test.hpp>

#include "csv.hpp"
using namespace std;
using boost::unit_test_framework::test_suite;

void test1() {
    vector<string> row = csv::parse_line("hoge,1,2");

    BOOST_CHECK_EQUAL(row.size(), 3);
    BOOST_CHECK_EQUAL(row[0], "hoge");
    BOOST_CHECK_EQUAL(row[1], "1");
    BOOST_CHECK_EQUAL(row[2], "2");


}


test_suite* init_unit_test_suite( int argc, char* argv[] ) {
    test_suite* test= BOOST_TEST_SUITE( "Example" );

    test->add( BOOST_TEST_CASE( &test1));

    return test;
}
