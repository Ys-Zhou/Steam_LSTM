INSERT INTO global_train_data
  WITH cte AS (
      SELECT
        userid,
        gameid,
        d1 + d2 + d3 + d4 + d5 + d6 AS playtime
      FROM raw_train_data
  )
  SELECT
    cte.userid,
    cte.gameid,
    cte.playtime / sq.usertime AS rating
  FROM cte
    JOIN (
           SELECT
             userid,
             SUM(playtime) AS usertime
           FROM cte
           GROUP BY userid
         ) sq
      ON cte.userid = sq.userid;