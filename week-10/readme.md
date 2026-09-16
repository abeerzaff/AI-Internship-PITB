
AI-Powered Data Insights Dashboard 

A simple dashboard that lets you upload a CSV file and instantly get a full picture of your data statistics, charts, and an AI-written summary with everything explained in plain language, no data-analysis background needed. 

Working: 

Upload a CSV, and the dashboard automatically walks through: 

Dataset Overview rows, columns, duplicates, and missing values at a glance 

Missing Value Check exactly which columns have gaps, and how much 

Basic Statistics averages, minimums, maximums for every column 

Charts a histogram and a scatter plot, each with a plain-English caption explaining what it means 

Correlation Analysis a heatmap showing which columns move together, explained in normal words 

Key Insights an automatic bullet-point summary of the numeric data 

AI-Generated Insights  a one-click AI analysis (powered by Google Gemini) that writes out findings, trends, data quality notes, and recommendations 

Setup — how to run this yourself 

Clone the repo and go into the project folder 

                git clone <repo> 
                cd <project> 
  

Create a virtual environment (recommended) 

               python3 -m venv .venv 
              source .venv/bin/activate      # on Mac/Linux 
  

Install the dependencies 

               pip install -r requirements.txt 
  

Set up your API key 

           Create a file named .env in the project folder and add: 

           GEMINI_API_KEY=your_api_key_here 
           MODEL_NAME=gemini-3.6-flash 
  

You can get a free Gemini API key from Google AI Studio. 

Run the app 

             streamlit run app.py 
  

Try it out 

Upload sample_dataset.csv (included in this repo) to see every section working, or upload your own CSV file. 

Sample questions this dashboard answers for you 

Does my data have missing or duplicate rows I should clean up? 

What's the typical range of values in a column, and are there outliers pulling the average up or down? 

Do two columns actually move together, or is that just a coincidence? 

What would a data analyst say about this dataset if I asked them to look at it? 

 