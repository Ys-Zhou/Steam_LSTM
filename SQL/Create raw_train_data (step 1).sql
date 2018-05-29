INSERT INTO raw_train_data (userid, gameid)
  SELECT
    userid,
    gameid
  FROM date170416
  UNION
  SELECT
    userid,
    gameid
  FROM date170430
  UNION
  SELECT
    userid,
    gameid
  FROM date170514
  UNION
  SELECT
    userid,
    gameid
  FROM date170528
  UNION
  SELECT
    userid,
    gameid
  FROM date170611
  UNION
  SELECT
    userid,
    gameid
  FROM date170625;