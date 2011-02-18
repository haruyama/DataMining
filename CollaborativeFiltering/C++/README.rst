Collaborative Filtering Sample
===============================

File Formats
--------------

Input
######

user_id,item_id,item_id.....

Example
^^^^^^^
219.99.236.178,10003034161186,10002834941181
210.141.112.34,10002835301181,10002802701186,10002812981181,10002820141181,20664208510938,10003039991181,10002812171181

Output
#######

item_id:score|item_id,score|item_id...

Example
^^^^^^^

10001946891177:0.16|10003125591177,0.1|10003297171177,0.08|10003294401177,0.0769231|10002827001177,0.0769231|10003258030989,0.0769231|10003208731177,0.0769231|10002884920113,0.0769231|10003340900769,0.0769231|10003121840813,0.0769231|10002308561176,

Programs
--------


StandAlone
##########

https://github.com/haruyama/DataMining/tree/master/CollaborativeFiltering/C++/item-item5tem-item5

./main <  users.csv > output

Performance
^^^^^^^^^^^

120000 users and 280000 items: about 80 minites

Using MPI
#########

https://github.com/haruyama/DataMining/tree/master/CollaborativeFiltering/C++/mpi3


rm output.*
mpirun -np 6 ./main -stdin 0  <  users.csv
cat output.* | sort output


Performance
^^^^^^^^^^^

120000 users and 280000 items | 6 processes (1 control process and 5 calcurating processes): about 20 minites
