import crewai as crew
import vision_parse
from vision_parse import VisionParser, PDFPageConfig
from crewai import Agent, Task, Crew, Process, LLM

llm = LLM(model="ollama/llama3.2", 
          base_url="http://localhost:11434")

def pdf_to_markdown(pdf_path: str) -> str:
    parser = VisionParser(
        model_name="llama3.2-vision:11b", 
        temperature=0.4,
        top_p=0.5,
        image_mode="url", 
        detailed_extraction=False, 
        enable_concurrency=False,
    )

    markdown_pages = parser.convert_pdf(pdf_path)
    result = ""
    for i, page_content in enumerate(markdown_pages):
        print(f"\n--- Page {i+1} ---\n{page_content}")
        result += f"\n--- Page {i+1} ---\n{page_content}\n"
    return result

splitwise_agent = Agent(
    role="An intelligent financial assistant skilled at splitting bills fairly based on item ownership and tax allocation.",
    goal=(
        "To accurately divide the total bill among participants by identifying who bought which items, "
        "calculating each person's share including tax, and providing clear summaries of amounts owed. "
        "Assign the portion of the bill to each person based on the items they purchased."
    ),
    backstory=(
        "I was created to make group bill splitting effortless, especially in scenarios where different people "
        "purchase different items. I carefully factor in taxes and ensure that each participant pays exactly for "
        "what they bought plus their fair share of taxes, so the process is transparent and fair."
    ),
    llm=llm  
)

splitwise_task = Task( 
    description="""
        You are given a bill in markdown format and details about who purchased which items.
        
        Bill Content: {bill_content}
        
        Purchase Details: {details}
        
        Your task is to:
        1. Parse the bill content to identify all items and their prices
        2. Calculate the subtotal, tax amount, and total
        3. Assign each item to the correct person based on the purchase details
        4. Calculate each person's subtotal (sum of their items)
        5. Allocate tax proportionally based on each person's subtotal
        6. Provide a clear breakdown showing:
           - Each person's items and their costs
           - Each person's subtotal
           - Each person's tax portion
           - Each person's total amount owed
        
        Make sure all calculations are accurate and the sum of individual totals equals the bill total.
    """,
    expected_output="""
        A detailed summary showing:
        1. Bill overview (subtotal, tax, total)
        2. For each person:
           - Items purchased and their individual costs
           - Subtotal for their items
           - Tax portion (proportional to their subtotal)
           - Total amount owed
        3. Verification that all individual totals sum to the bill total
    """,
    agent=splitwise_agent
)

crew = Crew(
    agents=[splitwise_agent],
    tasks=[splitwise_task],
    process=Process.sequential
)

if __name__ == "__main__":
    bill_path = "bill.pdf"
    
    try:
        markdown_content = pdf_to_markdown(bill_path)
        
        inputs = {
            "bill_content": markdown_content,
            "details": {
                "Jay": ["eggs", "milk", "apples"], 
                "Bob": ["carrots", "balloons"],
                "Christina": ["oranges"]  
            }
        }
        
        result = crew.kickoff(inputs=inputs)  
        
        print("\n" + "="*50)
        print("BILL SPLITTING RESULT")
        print("="*50)
        print(result)
        
    except Exception as e:
        print(f"Error processing bill: {e}")
        print("Make sure 'bill.pdf' exists and Ollama is running with the llama3.2-vision model.")