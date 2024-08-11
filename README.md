## Introduction

This repository contains a Python script designed to analyze veterinary prescriptions and calculate important dosing information using the OpenAI API (GPT-3.5-turbo or GPT-4-turbo model). The script reads input data from text files, interacts with the OpenAI API to analyze the data, and outputs the results to an Excel file.

## Prerequisites

- **Python 3.7 or higher**
- **OpenAI API Key**: You need an API key from OpenAI to run this script.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/havocy28/prescription-text-analyzer.git
   cd prescription-text-analyzer
   ```

2. **Create a Virtual Environment**

   It is recommended to use a virtual environment to manage dependencies. You can create and activate a virtual environment as follows:

   - On **Windows**:

     ```bash
     python -m venv venv
     .\venv\Scripts\activate
     ```

   - On **Linux/Mac**:

     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install Required Python Packages**

   Once the virtual environment is activated, install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up OpenAI API Key**

   To use the OpenAI API, you'll need to set up the following environment variables:

   - **OPENAI_ORGANIZATION**: Your OpenAI organization ID
   - **OPENAI_API_KEY**: Your OpenAI API key

   You can obtain these values from your OpenAI account.

   ### Setting Up Environment Variables

   - On **Windows**:

     ```bash
     setx OPENAI_ORGANIZATION "your-organization-id"
     setx OPENAI_API_KEY "your-api-key"
     ```

   - On **Linux/Mac**:

     ```bash
     export OPENAI_ORGANIZATION="your-organization-id"
     export OPENAI_API_KEY="your-api-key"
     ```

   ### Alternatively, Modify the Script Directly

   If you prefer not to set environment variables, you can directly modify the script by replacing:

   ```python
   openai_organization = os.getenv('OPENAI_ORGANIZATION')
   openai_api_key = os.getenv('OPENAI_API_KEY')
   ```

   With:

   ```python
   openai_organization = "your-organization-id"
   openai_api_key = "your-api-key"
   ```

## Usage

1. **Input Files**

   - **Synthetic Prescriptions**: A text file where each line contains information on the patient's weight, medication unit size, and total units dispensed.
   - **Ingredients List**: A text file where each line contains a possible antimicrobial ingredient.

2. **Running the Script**

   You can run the script by simply executing:

   ```bash
   python run_prompts.py
   ```

   The output will be an Excel file containing the calculated dosing information, saved to the path specified in the script.

3. **Example Input and Output**

   - **Input Files**:
     - `synthetic_rx.txt` contains synthetic prescriptions.
     - `ingredient_list.csv` contains possible antimicrobial ingredients.

   - **Output File**:
     - `dose_predictions.xlsx` will be created with structured dosing information.

## Testing Your OpenAI Setup

Before running the main script, you can test whether your OpenAI configuration is set up correctly using the `openai_check.py` script. This script will attempt to make a simple request to the OpenAI API and confirm if the environment variables are correctly configured.

To run the test:

```bash
python openai_check.py
```

If everything is set up correctly, you should see a confirmation message along with a response from OpenAI.

## Customization

- **Input and Output Paths**: You can customize the paths for input and output files directly in the script under the `CFG` class:

  ```python
  class CFG:
      ingredients_file = './path_to_ingredient_list.csv'
      input_file = './path_to_synthetic_rx.txt'
      output_file = './path_to_output_file.xlsx'
  ```

- **OpenAI Model**: By default, the script uses the GPT-3.5-turbo model. If you prefer to use GPT-4-turbo, you can change the model name in the `CFG` class:

  ```python
  model_name = 'gpt-4-turbo'
  ```

## Testing

The script includes synthetic prescriptions in the `synthetic_rx.txt` file for testing purposes. You can use this data to verify that the script is functioning correctly before applying it to real data.

## Troubleshooting

If you encounter issues while running the script, consider the following:

1. **Check Environment Variables**: Ensure that `OPENAI_ORGANIZATION` and `OPENAI_API_KEY` are correctly set.
2. **API Limits**: Be aware of OpenAI's rate limits and quotas, which may impact script execution if exceeded.
3. **Error Handling**: The script includes retry logic with exponential backoff in case of API errors. Check the console output for any specific error messages.

## Reference

This code accompanies the paper:

**Brian Hur, Lucy Lu Wang, Laura Hardefeldt, and Meliha Yetisgen. 2024. [*Is That the Right Dose? Investigating Generative Language Model Performance on Veterinary Prescription Text Analysis*](https://aclanthology.org/2024.bionlp-1.30/). In Proceedings of the 23rd Workshop on Biomedical Natural Language Processing, pages 390–397, Bangkok, Thailand. Association for Computational Linguistics.**

