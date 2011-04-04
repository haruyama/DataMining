{-# LANGUAGE BangPatterns #-}

import List
import System.IO
import System.Exit
import System.Environment(getArgs)
import Control.Monad(when)
import qualified Data.Map as Map
--import qualified Data.Set as Set
import qualified Data.IntSet as IntSet

main = do
--  args <- getArgs
--  when (length args /= 1) $ do
--         putStrLn "Syntax: recomend filename"
--         exitFailure
--  content <- readFile (head args)
  content <- getContents
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

printscoreslist_sub2 [] = putStrLn ""
printscoreslist_sub2 list = 
  let productscore = head list
  in
    do
      putStr (show (snd productscore))
      putStr "|"
      putStr (fst productscore)
      putStr ","
      printscoreslist_sub2 (tail list)

top_matches :: [(String, IntSet.IntSet)]
            -> [(String , [(String, Double)])]
top_matches productinfos =
    top_matches_sub  productinfos productinfos []


top_matches_sub :: [(String, IntSet.IntSet)]
                -> [(String, IntSet.IntSet)]
                -> [(String , [(String, Double)])]
                -> [(String , [(String, Double)])]
top_matches_sub [] _ result = result
top_matches_sub productinfos allproductinfos result =
        let head_product = head productinfos
        in
          top_matches_sub (tail productinfos) 
                          allproductinfos
                           ((fst head_product , 
                             (top_matches_sub2 head_product
                                               [x | x <- allproductinfos, x /= head_product]
                                                [])) : result)



top_matches_sub2 :: (String, IntSet.IntSet) ->
                   [(String, IntSet.IntSet)] ->
                   [(String, Double)] -> [(String, Double)]
top_matches_sub2 _ [] result = result
top_matches_sub2  productinfo unchecked_productinfos result = 
    let product = fst productinfo
        productset = snd productinfo
        checking_product = fst .head $ unchecked_productinfos
        score = tanimoto_wrap product checking_product
                productset (snd . head $ unchecked_productinfos)
    in
      if score > 0
      then
          top_matches_sub2 productinfo
                               (tail unchecked_productinfos) 
                               ((checking_product, score) : result)
      else
          top_matches_sub2 productinfo
                               (tail unchecked_productinfos)
                               result
            
tanimoto_wrap p1 p2 s1 s2 =
    if p1 > p2 
    then tanimoto s1 s2
    else tanimoto s2 s1

strictTanimoto us1 us2 = us1 `seq` us2 `seq` tanimoto us1 us2

tanimoto3 :: IntSet.IntSet -> IntSet.IntSet -> Double
tanimoto3 s1 s2 =
    let  (bunsi, bunbo) = IntSet.partition (\x -> IntSet.member x (IntSet.intersection s1 s2))
                          (IntSet.union s1 s2)
    in
      if IntSet.null bunsi
      then 0.0
      else ((fromIntegral (IntSet.size bunsi))::Double) / (fromIntegral (IntSet.size bunbo))

tanimoto2 :: IntSet.IntSet -> IntSet.IntSet -> Double
tanimoto2 s1 s2 =
    let  us = IntSet.union s1 s2
         intersection_size = IntSet.size us - IntSet.size s1 - IntSet.size s2
    in
      if intersection_size == 0
         then 
             0.0
         else
             ((fromIntegral intersection_size)::Double) / fromIntegral (IntSet.size us)


tanimoto :: IntSet.IntSet -> IntSet.IntSet -> Double
tanimoto us1 us2 =
    let intersection = IntSet.intersection us1 us2
    in
      if IntSet.null intersection
      then
         0.0
      else
          ((fromIntegral (IntSet.size intersection))::Double) / fromIntegral (IntSet.size us1 + IntSet.size us2 - IntSet.size intersection)

tanimoto4 :: IntSet.IntSet -> IntSet.IntSet -> Double
tanimoto4 us1 us2 =
    let intersection = IntSet.intersection us1 us2
    in
      if IntSet.null intersection
      then
         0.0
      else
          ((fromIntegral (IntSet.size intersection))::Double) / fromIntegral (IntSet.size (IntSet.union us1 us2))


parseLines :: String -> [(Int, [String])]
parseLines = parseline 0 . lines

parseline :: Int -> [String] -> [(Int, [String])]
parseline _ [] = []
parseline n lines = 
    let fields = split ',' (head lines)
    in (n, tail fields) : parseline (n + 1) (tail lines)

split :: Eq a => a -> [a] -> [[a]]
split _ [] = [[]]
split delim str =
    let (before, remainder) = span (/= delim) str
    in
      before : case remainder of
                [] -> []
                x -> split delim (tail x)
                                
transform :: [(Int, [String])] -> [(String,IntSet.IntSet)]
transform userinfos =
    Map.toList $ transform_sub userinfos Map.empty

transform_sub :: [(Int,[String])] -> Map.Map String (IntSet.IntSet) -> Map.Map String (IntSet.IntSet)
transform_sub [] result = result
transform_sub userinfos result =
    let userinfo = head userinfos
        user = fst userinfo
        products = snd userinfo
    in
      transform_sub (tail userinfos) (transform_sub2 user products result)
    
transform_sub2 :: Int -> [String] -> Map.Map String (IntSet.IntSet) -> Map.Map String (IntSet.IntSet)
transform_sub2 user  = 
    Map.unionWith IntSet.union . Map.fromList . map (\p -> (p, (IntSet.fromList [user])))


