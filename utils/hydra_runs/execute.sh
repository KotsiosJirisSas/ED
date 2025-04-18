#!/bin/bash
#L=2 square non-zero green's function elements
for i in {0..15}; do
    if (( i % 2 == 0 )); then
        j=$((i+1))
    else
        j=$((i-1))
    fi
    echo "command /mnt/users/kotssvasiliou/ALF_mod/pyalf/bin/python /mnt/users/kotssvasiliou/ED/utils/ED_chains_final.py $i $i"
    addqueue -m 1 -c "30 mins" /mnt/users/kotssvasiliou/ALF_mod/pyalf/bin/python /mnt/users/kotssvasiliou/ED/utils/ED_chains_final.py $i $i
    sleep 20

    echo "command /mnt/users/kotssvasiliou/ALF_mod/pyalf/bin/python /mnt/users/kotssvasiliou/ED/utils/ED_chains_final.py $i $j"
    addqueue -m 1 -c "30 mins" /mnt/users/kotssvasiliou/ALF_mod/pyalf/bin/python /mnt/users/kotssvasiliou/ED/utils/ED_chains_final.py $i $j
    sleep 20
done
