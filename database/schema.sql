-- ====================================================================
-- RoadSafe AI – Real-Time Accident Hotspot Prediction System
-- Database Schema: accident_hotspot_db
-- ====================================================================

CREATE DATABASE IF NOT EXISTS `accident_hotspot_db` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE `accident_hotspot_db`;

-- 1. Accidents Table
CREATE TABLE IF NOT EXISTS `accidents` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `accident_date` DATE NOT NULL,
    `accident_time` VARCHAR(10) NOT NULL,
    `latitude` DOUBLE NOT NULL,
    `longitude` DOUBLE NOT NULL,
    `location` VARCHAR(150) NOT NULL,
    `weather` VARCHAR(50) NOT NULL,
    `road_condition` VARCHAR(50) NOT NULL,
    `road_type` VARCHAR(50) NOT NULL,
    `traffic_volume` INT NOT NULL,
    `vehicle_count` INT NOT NULL,
    `visibility` DOUBLE NOT NULL,
    `speed_limit` INT NOT NULL,
    `previous_accidents` INT NOT NULL DEFAULT 0,
    `severity` VARCHAR(50) NOT NULL,
    `risk_level` VARCHAR(20) NOT NULL,
    `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX `idx_lat` (`latitude`),
    INDEX `idx_lon` (`longitude`),
    INDEX `idx_acc_date` (`accident_date`),
    INDEX `idx_risk` (`risk_level`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 2. Predictions Table
CREATE TABLE IF NOT EXISTS `predictions` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `latitude` DOUBLE NOT NULL,
    `longitude` DOUBLE NOT NULL,
    `prediction_time` DATETIME NOT NULL,
    `weather` VARCHAR(50) NOT NULL,
    `traffic_volume` INT NOT NULL,
    `predicted_risk` VARCHAR(20) NOT NULL,
    `probability` DOUBLE NOT NULL,
    `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX `idx_pred_lat` (`latitude`),
    INDEX `idx_pred_lon` (`longitude`),
    INDEX `idx_pred_risk` (`predicted_risk`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 3. Users Table
CREATE TABLE IF NOT EXISTS `users` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `username` VARCHAR(80) NOT NULL UNIQUE,
    `email` VARCHAR(120) NOT NULL UNIQUE,
    `password_hash` VARCHAR(255) NOT NULL,
    `role` VARCHAR(20) NOT NULL DEFAULT 'Planner', -- 'Admin' or 'Planner'
    `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
