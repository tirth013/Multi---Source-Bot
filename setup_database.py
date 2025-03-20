import sqlite3
import os

def create_database():
    """Create a sample database for research data."""
    # Remove existing database if it exists
    if os.path.exists("research_data.db"):
        os.remove("research_data.db")
    
    # Connect to the database
    conn = sqlite3.connect("research_data.db")
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
    CREATE TABLE research_topics (
        id INTEGER PRIMARY KEY,
        topic TEXT NOT NULL,
        category TEXT NOT NULL,
        description TEXT
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE research_sources (
        id INTEGER PRIMARY KEY,
        source_name TEXT NOT NULL,
        source_type TEXT NOT NULL,
        reliability_score FLOAT,
        topic_id INTEGER,
        FOREIGN KEY (topic_id) REFERENCES research_topics (id)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE research_findings (
        id INTEGER PRIMARY KEY,
        source_id INTEGER,
        finding_text TEXT NOT NULL,
        date_added DATE,
        FOREIGN KEY (source_id) REFERENCES research_sources (id)
    )
    ''')
    
    # Insert sample data
    # Topics
    topics = [
        (1, "Climate Change", "Environmental", "Research on global climate patterns and human impact"),
        (2, "Artificial Intelligence", "Technology", "Development and applications of AI systems"),
        (3, "Renewable Energy", "Environmental", "Research on sustainable energy sources"),
        (4, "COVID-19", "Health", "Research on the coronavirus pandemic"),
        (5, "Blockchain", "Technology", "Distributed ledger technology and applications")
    ]
    
    cursor.executemany("INSERT INTO research_topics VALUES (?, ?, ?, ?)", topics)
    
    # Sources
    sources = [
        (1, "IPCC Reports", "Academic", 9.5, 1),
        (2, "Nature Journal", "Academic", 9.8, 1),
        (3, "arXiv", "Preprint", 8.2, 2),
        (4, "IEEE", "Academic", 9.3, 2),
        (5, "IEA", "Government", 8.9, 3),
        (6, "WHO", "Government", 9.4, 4),
        (7, "CDC", "Government", 9.2, 4),
        (8, "Bitcoin Whitepaper", "Primary", 8.7, 5)
    ]
    
    cursor.executemany("INSERT INTO research_sources VALUES (?, ?, ?, ?, ?)", sources)
    
    # Findings
    findings = [
        (1, 1, "Global temperatures have risen by 1.1°C since pre-industrial times", "2023-01-15"),
        (2, 2, "Arctic sea ice is declining at a rate of 13.1% per decade", "2023-02-20"),
        (3, 3, "Transformer models have revolutionized natural language processing", "2023-03-10"),
        (4, 4, "Edge computing enables AI applications with reduced latency", "2023-04-05"),
        (5, 5, "Solar PV capacity increased by 22% in 2022", "2023-05-12"),
        (6, 6, "COVID-19 variants continue to evolve with varying transmission rates", "2023-06-18"),
        (7, 7, "Long COVID affects approximately 10-30% of COVID-19 patients", "2023-07-22"),
        (8, 8, "Blockchain technology enables trustless transactions without intermediaries", "2023-08-30")
    ]
    
    cursor.executemany("INSERT INTO research_findings VALUES (?, ?, ?, ?)", findings)
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print("Sample database created successfully.")

if __name__ == "__main__":
    create_database()