
# Splitwise Bill Splitting with AI Agent

This project uses a AI agent to automate the process of splitting a bill from a PDF file among a group of people. The agent can identify items, calculate each person's share including tax, and provide a clear summary of who owes what.

## Features

- **PDF Bill Parsing**: Extracts text from a PDF bill to identify items and prices.
- **Fair Splitting**: Divides the total bill based on who purchased which items.
- **Tax Calculation**: Allocates tax proportionally to each person's subtotal.
- **Clear Summaries**: Provides a detailed breakdown of each person's share.

## Tools and Technologies

- **CrewAI**: For creating and managing the AI agent.
- **VisionParser**: For parsing PDF files and extracting text.
- **Ollama**: For running the large language model locally.
- **LLM (Large Language Model)**: For understanding the bill and performing calculations.

## How It Works

1. **PDF to Markdown**: The `pdf_to_markdown` function reads a PDF bill and converts its content into Markdown format.
2. **AI Agent**: An AI agent, built with the `crewai` library, processes the Markdown content.
3. **Task Definition**: The agent is assigned a task to parse the bill, calculate each person's share, and generate a summary.
4. **LLM Integration**: The agent uses a large language model (LLM) to understand the bill and perform the calculations.
5. **Output**: The final output is a detailed summary of who owes what, including a breakdown of items and taxes.

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Splitwise-with-AIAgent-crewai.git
   ```
2. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the script**:
   ```bash
   python splitwise-crewai-agent.py
   ```

## Usage

1. **Place your bill in the project directory**: Make sure the bill is in PDF format and named `bill.pdf`.
2. **Define who bought what**: In the `splitwise-crewai-agent.py` script, update the `details` dictionary with the names of the participants and the items they purchased.
3. **Run the script**: The script will process the bill and print a summary of who owes what.

## Example

Here's an example of how to define the `details` dictionary:

```python
inputs = {
    "bill_content": markdown_content,
    "details": {
        "Jay": ["eggs", "milk", "apples"], 
        "Bob": ["carrots", "balloons"],
        "Christina": ["oranges"]
    }
}
```

This will split the bill among Jay, Bob, and Christina based on the items they purchased.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or suggestions.
