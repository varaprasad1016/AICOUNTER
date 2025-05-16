#!/usr/bin/env python3
"""
Database Setup Script for People Counter Application
---------------------------------------------------
This script creates the necessary database and tables for the people counter app.
"""

import mysql.connector
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    "host": "localhost",
    "user": "root",  # Update with your MySQL username
    "password": "password",  # Update with your MySQL password
}

# Database and tables setup
def setup_database():
    """Create the necessary database and tables"""
    try:
        # Connect without specifying database
        conn = mysql.connector.connect(
            host=DB_CONFIG["host"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"]
        )
        cursor = conn.cursor()
        
        # Create database
        database_name = "people_count"
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database_name}")
        cursor.execute(f"USE {database_name}")
        logger.info(f"Database '{database_name}' created or already exists")
        
        # Create main count table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS people_count_main (
                id INT AUTO_INCREMENT PRIMARY KEY,
                current_count INT NOT NULL,
                people_in INT NOT NULL,
                people_out INT NOT NULL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
        """)
        logger.info("Table 'people_count_main' created or already exists")
        
        # Create transaction table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS people_count_transactions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                event_type ENUM('IN', 'OUT') NOT NULL,
                previous_count INT NOT NULL,
                new_count INT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                camera_id VARCHAR(50) DEFAULT 'main',
                confidence FLOAT
            )
        """)
        logger.info("Table 'people_count_transactions' created or already exists")
        
        # Initialize main count table with a default record if it doesn't exist
        cursor.execute("""
            INSERT INTO people_count_main (current_count, people_in, people_out)
            SELECT 0, 0, 0 FROM DUAL
            WHERE NOT EXISTS (SELECT * FROM people_count_main)
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        logger.info("Database setup completed successfully")
        return True
    except mysql.connector.Error as err:
        logger.error(f"MySQL Error: {err}")
        return False
    except Exception as e:
        logger.error(f"Error: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting database setup...")
    
    if setup_database():
        logger.info("Database setup completed successfully")
    else:
        logger.error("Database setup failed")
        sys.exit(1)
