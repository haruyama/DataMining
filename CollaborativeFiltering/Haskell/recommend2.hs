import List
import System.IO
import System.Exit
import System.Environment(getArgs)
import Control.Monad(when)
import qualified Data.Map as Map
import qualified Data.Set as Set

main = do
  args <- getArgs
  when (length args /= 1) $ do
         putStrLn "Syntax: recomend filename"
         exitFailure
  content <- readFile (head args)
  let userinfos = parseLines content
  -- mapM_ (putStrLn . unwords)  (map snd users)
  let productinfos = transform userinfos                  
  let scores = top_matches productinfos
  --  mapM_ putStrLn (map fst (Map.toList productinfos))
  --  mapM_ (putStrLn . unwords . Set.toList) (map snd (Map.toList productinfos))
  --  putStrLn (Map.showTree scores)
  printscores scores           

        
printscores :: [(String , [(String, Double)])] -> IO ()
printscores[] =   putStr ""
printscores scoreslist = 
    let productscores = head scoreslist
    in
      do
        putStr (fst productscores)
        putStr ":"
        printscores_sub (snd productscores)
        printscores (tail scoreslist)

printscores_sub = 
    printscoreslist_sub2 . 
                         take 10 . 
                              sortBy 
                              (\x y -> snd y `compare` snd x)

printscoreslist_sub2 [] = putStrLn "\n"
printscoreslist_sub2 list = 
  let productscore = head list
  in
    do
      putStr (show (snd productscore))
      putStr "|"
      putStr (fst productscore)
      putStr ","
      printscoreslist_sub2 (tail list)

top_matches :: [(String, Set.Set String)]
            -> [(String , [(String, Double)])]
top_matches productinfos =
    top_matches_sub  productinfos productinfos []


top_matches_sub :: [(String, Set.Set String)]
                -> [(String, Set.Set String)]
                -> [(String , [(String, Double)])]
                -> [(String , [(String, Double)])]
top_matches_sub [] _ result = result
top_matches_sub productinfos allproductinfos result =
        let product = fst. head $ productinfos
            productset = snd . head $ productinfos
        in
          top_matches_sub (tail productinfos) 
                          allproductinfos
                           ((product , 
                             (top_matches_sub2 product 
                                               productset
                                               allproductinfos
                                                [])) : result)



top_matches_sub2 :: String -> 
                    Set.Set String ->
                   [(String, Set.Set String)] ->
                   [(String, Double)] -> [(String, Double)]
top_matches_sub2 _ _ [] result = result
top_matches_sub2  product productset unchecked_productinfos result = 
    let checking_product = fst .head $ unchecked_productinfos
    in
      if checking_product == product 
      then
          top_matches_sub2 product productset 
                               (tail unchecked_productinfos) result
      else
          let 
              score = tanimoto productset (snd . head $ unchecked_productinfos)
          in
            if score > 0
            then
                top_matches_sub2 product productset
                                     (tail unchecked_productinfos) 
                                     ((checking_product, score) : result)
            else
                top_matches_sub2 product productset 
                                     (tail unchecked_productinfos)
                                     result
            

tanimoto :: Ord a => (Set.Set a) -> (Set.Set a) -> Double
tanimoto us1 us2 
         | Set.size us1 > Set.size us2 = tanimoto us2 us1

tanimoto us1 us2 =
    let intersection_size = Set.size (Set.intersection us1 us2)
    in
      if intersection_size == 0
      then
         0.0
      else
          ((fromIntegral intersection_size)::Double) / fromIntegral (Set.size us1 + Set.size us2 - intersection_size)

parseLines :: String -> [(String, [String])]
parseLines = map parseline . lines

parseline :: String -> (String, [String])
parseline input = 
    let fields = split ',' input
    in (head fields, tail fields)

split :: Eq a => a -> [a] -> [[a]]
split _ [] = [[]]
split delim str =
    let (before, remainder) = span (/= delim) str
    in
      before : case remainder of
                [] -> []
                x -> split delim (tail x)
                                
transform :: [(String, [String])] -> [(String,Set.Set String)]
transform userinfos =
    Map.toList $ transform_sub userinfos Map.empty

transform_sub :: [(String,[String])] -> Map.Map String (Set.Set String) -> Map.Map String (Set.Set String)
transform_sub [] result = result
transform_sub userinfos result =
    let userinfo = head userinfos
        user = fst userinfo
        products = snd userinfo
    in
      transform_sub (tail userinfos) (transform_sub2 user products result)
    
transform_sub2 :: String -> [String] -> Map.Map String (Set.Set String) -> Map.Map String (Set.Set String)
transform_sub2 user  = 
    Map.unionWith Set.union . Map.fromList . map (\p -> (p, (Set.fromList [user])))


