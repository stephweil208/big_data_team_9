# InsideInsight: Agentic AI for Airbnb Pricing Strategy and Performance Optimization

**Team 9**  
Bhavisha Chafekar · Jyothirmai Sri Peesapati · Phoenix Ferrari · Stephen Weiler · Tzu-Yu Chen  

## InsideInsight Flyer
[View Project Flyer](flyer.pdf)

---

## Executive Summary

This project repository is created in partial fulfillment of the requirements for the Big Data Analytics course offered by the Master of Science in Business Analytics program at the Carlson School of Management, University of Minnesota.

InsideInsight is an end-to-end Big Data and Agentic AI system designed to transform large-scale Airbnb data into actionable insights for hosts and property managers. Using the Inside Airbnb dataset, which includes listings, calendar availability, and guest reviews across multiple cities, we developed a scalable data pipeline and analytics framework to evaluate pricing strategy, occupancy performance, and customer sentiment.

The system is built on a medallion architecture (Bronze → Silver → Gold) using Apache Spark and Databricks, enabling efficient processing and feature engineering on large datasets. Structured outputs such as occupancy rates, pricing benchmarks, competitive scores, and sentiment insights are generated and stored in analysis-ready tables.

To extend beyond traditional analytics, we implemented a multi-agent AI system using LangGraph and large language models. This system converts structured insights into grounded, data-driven recommendations, allowing users to generate actionable pricing and performance strategies for individual Airbnb listings.

The final deliverable includes:
- A scalable data pipeline for Airbnb data processing  
- Feature engineering for pricing and occupancy analysis  
- NLP-based sentiment and topic analysis on guest reviews  
- An AI-powered recommendation engine for host decision-making  
- An interactive Streamlit dashboard for exploration and insights  

---

## Setup & Usage Instructions

### Quick Start (Recommended Workflow)

This project is organized into phases. To fully reproduce the system:

#### 1. Data Pipeline (Phase 1)
- Builds Bronze → Silver → Gold tables  
- Output tables:
  - `gold_listings`
  - `gold_reviews`
  - `gold_calendar`
- 📄 See: [Phase 1 Instructions](docs/Phase_1_Medallion_Pipeline_Reuse_Instructions.pdf)

#### 2. Feature Engineering (Phase 2)
- Builds the `gold_features` table used for analytics and AI  
- Includes pricing benchmarks, occupancy rates, and competitive scoring  
- 📄 See: [Phase 2 Instructions](docs/Phase_2_Analytics_Reuse_Instructions.pdf)

#### 3. NLP Processing (Phase 3)
- Builds `gold_nlp_features` from review sentiment and topic analysis  
- 📄 See: `/docs/phase3_nlp.pdf`

#### 4. AI Agent (Phase 4)
- Combines structured + NLP features to generate recommendations  

    action_plan(listing_id)

- 📄 See: `/docs/phase4_ai_agent.pdf`

#### 5. Dashboard (Optional)

Run locally using Streamlit:

    python -m streamlit run app/app.py

- 📄 See: `/docs/dashboard.pdf`

---

### Accessing Final Data

All final tables are stored in Databricks:

    spark.table('workspace.default.gold_listings')
    spark.table('workspace.default.gold_features')
    spark.table('workspace.default.gold_nlp_features')

---

### Requirements

- Databricks workspace with access to `workspace.default` schema  
- Python (3.10+)  
- PySpark / Pandas  
- Required packages (see Phase 4 or dashboard instructions)  
- Groq API key (for AI agent)

---

### Notes

- Calendar pricing fields were null across all cities; listing price is used instead  
- Some reviews without matching listings were removed  
- Multi-unit listings may appear duplicated by design  

---

## Dataset

- **Inside Airbnb Dataset**  
  https://insideairbnb.com/get-the-data  
  - Listings data (pricing, amenities, location)  
  - Calendar data (availability)  
  - Review data (guest feedback)  

---

## Tools & Technologies

- Apache Spark / Databricks  
- Python (PySpark, Pandas)  
- Delta Lake / Hive Metastore  
- NLP (VADER, Transformer-based methods)  
- LangGraph + LLM (Groq / LLaMA)  
- Streamlit  

---

## Repository

GitHub: https://github.com/stephweil208/big_data_team_9
