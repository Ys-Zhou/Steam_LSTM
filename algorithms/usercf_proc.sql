CREATE PROCEDURE usercf(IN uid VARCHAR(17))
  BEGIN
    SELECT
      ref.gameid,
      SUM(sim.cossim * ref.rating) AS pri
    FROM (
           SELECT
             user1,
             user2,
             cossim
           FROM global_user_sim
           WHERE user1 = uid
           UNION
           SELECT
             user2,
             user1,
             cossim
           FROM global_user_sim
           WHERE user2 = uid
         ) AS sim
      JOIN global_train_data AS ref
        ON sim.user2 = ref.userid
           AND ref.gameid NOT IN (
        SELECT gameid
        FROM global_train_data
        WHERE userid = uid
      )
    GROUP BY ref.gameid
    ORDER BY pri DESC
    LIMIT 50;
  END;