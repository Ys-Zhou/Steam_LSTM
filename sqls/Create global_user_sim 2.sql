INSERT INTO global_user_sim (user1, user2, cossim)
  WITH cte AS (
      SELECT
        userid,
        SQRT(SUM(POW(rating, 2))) AS norm
      FROM global_train_data
      GROUP BY userid
  )
  SELECT
    sub1.user1,
    sub1.user2,
    sub1.dot / (sub2.norm * sub3.norm) AS cossim
  FROM (
         SELECT
           a.userid                 AS user1,
           b.userid                 AS user2,
           SUM(a.rating * b.rating) AS dot
         FROM global_train_data AS a
           JOIN global_train_data AS b
             ON a.userid > b.userid AND a.gameid = b.gameid
         GROUP BY a.userid, b.userid
       ) AS sub1
    JOIN cte AS sub2
      ON sub1.user1 = sub2.userid
    JOIN cte AS sub3
      ON sub1.user2 = sub3.userid;