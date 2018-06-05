CREATE TABLE global_user_sim (
  user1  VARCHAR(17),
  user2  VARCHAR(17),
  cossim DOUBLE NOT NULL,
  primary key (user1, user2)
);