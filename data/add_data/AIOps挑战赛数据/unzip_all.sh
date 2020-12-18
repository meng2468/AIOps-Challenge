for i in `seq 22 31`
do
unzip -P $(cat passwd.txt) 2020_05_${i}_lock.zip
unzip 2020_05_${i}.zip
done
