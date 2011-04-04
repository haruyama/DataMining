import scala.collection.immutable.Set
import scala.collection.immutable.Map
//import scala.collection.mutable.ListBuffer
import java.io._


object Recommend {
    val MAX_RELATED_ITEMS=10

    private def readline(line : String) : Set[Long] = {
        val words = line.split(",")
        var user = Set.empty[Long]

        for (i <- 1 until words.length) {
           user += words(i).toLong 
        }
        return user
    }

    private def transform_prefs(users : List[Set[Long]]) : Map[Long, Set[Int]] = {
        var items : Map[Long, Set[Int]] = Map.empty
        var user = 1
        for (user_info <- users  ) {
            for (item_id <- user_info) {
                if (items.contains(item_id)) {
                    items = items + (item_id -> (items(item_id) + user))
                } else {
                    items = items + (item_id -> Set(user))
                }

            }
            user += 1

        }

        return items
    }

    def top_matches(items : Map[Long, Set[Int]], max_items : Int) : Map[Long, List[(Long, Double)]] = {
        val memo = new Array[Array[Double]](items.size, items.size)
        for (k <- 0 until items.size) {
            for (l <- 0 until items.size) {
                memo(k)(l) = -100.0
            }
        }

        var item_scores = Map.empty[Long, List[(Long, Double)]]

        var i = 0
        for (item1 <- items) {
            var j = 0
            //var scores_buffer : ListBuffer[(Long, Double)] = new ListBuffer
            var scores_tmp : List[(Long, Double)] = List()
            for (item2 <- items) {
                if (item1._1 != item2._1) {
                    val memo_score = memo(j)(i)
                    var score : Double = 0.0
                    if (memo_score > -2) {
                        score = memo_score
                    } else {
                        score = sim_tanimoto(item1._2, item2._2)
                        memo(i)(j) = score
                    }

                    if (score > 0) {
                        scores_tmp = (item2._1, score) :: scores_tmp
                    }
                }
                j += 1
            }
            val scores : List[(Long, Double)]  = {
                if (scores_tmp.size > max_items) {
                    (scores_tmp sort (_._2 > _._2)).take(max_items)
                } else {
                    scores_tmp sort (_._2 > _._2)
                }
            }

            item_scores += (item1._1 -> scores)
            i += 1
        }
        return item_scores 
    }

    def sim_tanimoto[T](set1 : Set[T], set2 : Set[T]) : Double = {
        val intersect = set1.intersect(set2)
        //val intersect = set1.clone
        //intersect.intersect(set2)
        return intersect.size.toDouble / (set1.size + set2.size - intersect.size)
    }
    def printout_scores(item_scores : Map[Long, List[(Long, Double)]]) {
        for (score_details <- item_scores) {
            print(score_details._1 + ":")
            for (score_detail <- score_details._2) {
                print (score_detail._2 + "|")
                print (score_detail._1 + ",")
            }
            println("")
        }
    }
    def main(args : Array[String]) {
        val stream = new InputStreamReader(System.in)
        val reader = new BufferedReader(stream)
        // read data
        var line : String = null
        //var users_buffer : ListBuffer[Set[Long]] = new ListBuffer
        var users : List[Set[Long]] = List()

        while ({ line = reader.readLine; line != null}) {
            users = readline(line) :: users
        }

        reader.close
        stream.close
        val items = transform_prefs(users)

        val item_scores = top_matches(items, MAX_RELATED_ITEMS)
        printout_scores(item_scores)
    }


}

// vim: set ts=4 sw=4 et:

