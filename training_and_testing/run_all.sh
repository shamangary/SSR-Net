max=4
for i in `seq 1 $max`
do
    echo "$i"
    KERAS_BACKEND=tensorflow python TYY_train.py --input ../data/imdb_db.npz --db imdb --netType $i
    KERAS_BACKEND=tensorflow python TYY_train.py --input ../data/wiki_db.npz --db wiki --netType $i --batch_size 50
    KERAS_BACKEND=tensorflow python TYY_train.py --input ../data/morph2_db_align.npz --db morph --netType $i --batch_size 50

done