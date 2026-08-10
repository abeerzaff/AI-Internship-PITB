# AI-Internship-PITB


This repository documents my learning journey throughout the AI Internship Program. It contains assignments, reports, notes, and project work completed during the internship. My primary goal is to expand my knowledge of Artificial Intelligence, Generative AI, Large Language Models (LLMs), Prompt Engineering, Retrieval-Augmented Generation (RAG), and Vector Databases while gaining hands-on experience with real-world AI applications. Through this internship, I aim to strengthen my technical, problem-solving, and collaboration skills, explore modern AI development practices, and build a strong foundation for a career in Artificial Intelligence and Data Science.

#WEEK-01

Internship orientation completed

Learning roadmap prepared

GitHub repository created

AI learning goals defined

#WEEK-02

In week-02 I have  revised Python fundamentals , practiced Lists, Tuples, Dictionaries . Implemented File Handling and CSV Handling and practiced Exception Handling.And for  the weekly  task I had built Student Performance Summary program and  Data Analyze program which analyze data and give important information about data like total number of rows and columns , column names and missing values. I had also built a mini utility called Data Health Checker  which inspects and analyze data.


#WEEK-03

In Week 03, I worked on a real-world dataset where I performed several data cleaning steps. After cleaning the data, I conducted Exploratory Data Analysis (EDA). Through EDA, I learned how to analyze and visualize data and how to convert raw data into meaningful insights.

##WEEK-04

This project is my Week 04 Data Visualization and Dashboard Thinking assignment. In this project, I used the cleaned Netflix dataset from Week 03 to create different visualizations that help identify trends, compare categories, and analyze patterns in the data. I also developed an interactive Streamlit dashboard to present these insights in a clear and user-friendly way, making it easier to understand the data and support better decision-making.


Dashboard (Streamlit): Public Data Insights Dashboard Includes:


Title

Dataset overview

Filters

Multiple charts

Key insights

Recommendations

Conclusion section


How to Run the Streamlit App Locally.

Clone the repository

Install the required dependencies- pip install streamlit pandas numpy matplotlib seaborn plotly

Run the app- streamlit run app.py


##week-05

This assignment focused on learning the fundamentals of Large Language Models (LLMs) and applying prompt engineering techniques to generate better AI responses. Different prompt versions Basic, Improved, and Professional were created for eight real-world use cases. The outputs were compared to understand how adding clear instructions, context, constraints, and structure improves the quality, accuracy, and reliability of AI-generated content. This assignment strengthened my understanding of effective prompt design and response evaluation.




##WEEK-06

This assignment focuses on understanding the practical applications of Generative AI by designing real-world AI use cases. The main objective is to learn how AI can be used to solve everyday problems, especially in public-sector workflows, while understanding its capabilities, limitations, and responsible use.

In this assignment, I created five practical Generative AI use cases:
- AI Email Assistant
- AI Report Summarizer
- AI Meeting Notes Generator
- AI Document Q&A Assistant
- AI Idea Generator for Public Services

For each use case, I explained the problem statement, target users, required inputs, AI process, expected outputs, limitations, and possible improvements. I also prepared a concept note, a use case table, and prompt examples to demonstrate how Generative AI can be applied effectively in real-world scenarios.


##WEEK 7

Environment Setup

This project was built and run inside a Python virtual environment (venv) to keep dependencies isolated from the system Python installation.

python3 -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

Install dependencies

pip install sentence-transformers chromadb torch


### Dependency Conflicts

During the setup, I faced some package version compatibility issues. After adjusting the package versions, the environment worked correctly.


## Project Overview 

This project builds a small semantic search system that retrieves document chunks based on meaning rather than exact keyword matching. It loads three sample documents, splits them into chunks, generates embeddings for each chunk using a sentence-transformer model, stores them in ChromaDB, and allows a user to ask a question and retrieve the most relevant chunks. A basic keyword search was also built to compare traditional word-matching against semantic retrieval.
