CREATE TABLE global_train_data (
  userid VARCHAR(17),
  gameid VARCHAR(6),
  rating DOUBLE NOT NULL,
  PRIMARY KEY (userid, gameid)
)