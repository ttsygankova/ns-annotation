input="empty_docs.txt"

count1_b4=$(ls session1 | wc -l)
count2_b4=$(ls session2 | wc -l)
count3_b4=$(ls session3 | wc -l)
count4_b4=$(ls session4 | wc -l)
count5_b4=$(ls session5 | wc -l)

while IFS= read -r line
do
    rm -f session1/$line
    rm -f session2/$line
    rm -f session3/$line
    rm -f session4/$line
    rm -f session5/$line
    
done < "$input"

count1_af=$(ls session1 | wc -l)
count2_af=$(ls session2 | wc -l)
count3_af=$(ls session3 | wc -l)
count4_af=$(ls session4 | wc -l)
count5_af=$(ls session5 | wc -l)

diff1=$((count1_b4-count1_af))
diff2=$((count2_b4-count2_af))
diff3=$((count3_b4-count3_af))
diff4=$((count4_b4-count4_af))
diff5=$((count5_b4-count5_af))

total=$((diff1+diff2+diff3+diff4+diff5))

echo "Total documents deleted: $total"
