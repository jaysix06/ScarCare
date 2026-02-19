CREATE DATABASE IF NOT EXISTS scar_system;
USE scar_system;

CREATE TABLE IF NOT EXISTS users (
  id INT AUTO_INCREMENT PRIMARY KEY,
  first_name VARCHAR(80) NOT NULL,
  last_name VARCHAR(80) NOT NULL,
  email VARCHAR(190) NOT NULL UNIQUE,
  password_hash VARCHAR(255) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS analyses (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NOT NULL,
  image_path VARCHAR(255) NOT NULL,
  predicted_label ENUM('hypertrophic', 'keloid', 'atrophic') NOT NULL,
  prob_hypertrophic FLOAT NOT NULL,
  prob_keloid FLOAT NOT NULL,
  prob_atrophic FLOAT NOT NULL,
  created_at DATETIME NOT NULL,
  INDEX idx_user_created (user_id, created_at),
  CONSTRAINT fk_analyses_user
    FOREIGN KEY (user_id) REFERENCES users(id)
    ON DELETE CASCADE
);
