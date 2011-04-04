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

printscoreslist_sub2 [] =
  putStrLn ""
printscoreslist_sub2 list = 
  let productscore = head list
  in
    do
      putStr (show (snd productscore))
      putStr "|"
      putStr (fst productscore)
      putStr ","
      printscoreslist_sub2 (tail list)

top_matches :: Map.Map String (Set.Set String) 
            -> [(String , [(String, Double)])]
top_matches productinfos =
    let products = Map.keys productinfos
    in
      top_matches_sub products products productinfos []


top_matches_sub :: [String] -> [String]
                -> Map.Map String (Set.Set String) 
                -> [(String , [(String, Double)])]
                -> [(String , [(String, Double)])]
top_matches_sub [] _ _ result = result
top_matches_sub products allproducts productinfos result =
        let product = head products
            productset = productinfos Map.! product
        in
          top_matches_sub (tail products) 
                          allproducts productinfos 
                           ((product , 
                             (top_matches_sub2 product 
                                               productset
                                               allproducts productinfos
                                                [])) : result)



top_matches_sub2 :: String -> 
                    Set.Set String ->
                   [String] -> Map.Map String (Set.Set String) ->
                   [(String, Double)] -> [(String, Double)]
top_matches_sub2 _ _ [] _ result = result
top_matches_sub2  product productset unchecked_products productinfos result = 
    let checking_product = head unchecked_products
    in
      if checking_product == product 
      then
          top_matches_sub2 product productset (tail unchecked_products)
                           productinfos result
      else
          let 
              us2 = productinfos Map.! checking_product
              score = tanimoto productset us2
          in
            if score > 0
            then
                top_matches_sub2 product productset (tail unchecked_products) productinfos
                                     ((checking_product, score) : result)
            else
                top_matches_sub2 product productset (tail unchecked_products) 
                                 productinfos result
            

tanimoto :: Ord a => (Set.Set a) -> (Set.Set a) -> Double
--tanimoto :: (Set.Set String) -> (Set.Set String) -> Double
tanimoto us1 us2 =
    let intersection_size = Set.size (Set.intersection us1 us2)
    in
      ((fromIntegral intersection_size)::Double) / (fromIntegral (Set.size us1 + Set.size us2 - intersection_size))::Double

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
                                
transform :: [(String, [String])] -> Map.Map String (Set.Set String)
transform userinfos =
    transform_sub userinfos Map.empty

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


