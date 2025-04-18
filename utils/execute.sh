#!/bin/bash
#for i in {0..15}; do
#    for j in {11..15}; do
#        echo "command /mnt/users/kotssvasiliou/ALF_mod/pyalf/bin/python ED_chains_final.py $i $j"
#        addqueue -m 1 -c "50 mins" /mnt/users/kotssvasiliou/ALF_mod/pyalf/bin/python ED_chains_final.py $i $j
#        sleep 30 #sleep for 30 seconds. trying to avoid writing in file at same moment....
#    done
#done
##########
#02/04 19:00 did i 0-->15 and j  0..2
#02/04 19:10 did i 0--->15 and j 3..5
#03/04 12:24 did ......... and j 6..10
#03/04 14:00 did ......... and j 11..15
#
#
#now do only non-zero elements
#first, 16 diagonal ones, plus off 16 off diagonal ones
for i in {0..15}; do
    if (( i % 2 == 0 )); then
        j=$((i+1))
    else
        j=$((i-1))
    fi
    echo "command /mnt/users/kotssvasiliou/ALF_mod/pyalf/bin/python /mnt/users/kotssvasiliou/ED/utils/ED_chains_final.py $i $i"
    addqueue -m 1 -c "50 mins" /mnt/users/kotssvasiliou/ALF_mod/pyalf/bin/python /mnt/users/kotssvasiliou/ED/utils/ED_chains_final.py $i $i
    sleep 10

    echo "command /mnt/users/kotssvasiliou/ALF_mod/pyalf/bin/python /mnt/users/kotssvasiliou/ED/utils/ED_chains_final.py $i $j"
    addqueue -m 1 -c "50 mins" /mnt/users/kotssvasiliou/ALF_mod/pyalf/bin/python /mnt/users/kotssvasiliou/ED/utils/ED_chains_final.py $i $j
    sleep 10
done
