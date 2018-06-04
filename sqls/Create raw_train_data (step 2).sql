UPDATE raw_train_data AS r, date170416 AS d
SET r.d1 = d.twoweeks
WHERE r.userid = d.userid AND r.gameid = d.gameid;