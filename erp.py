from typing import List, Dict, Literal, Optional, Union, Any
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, MessagesState
from langgraph.types import Command
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.agents import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import os
import sys
import io

def clean_numeric_value(value):
    """Clean numeric values by removing commas and converting to float if possible."""
    if pd.isna(value):
        return value
    
    str_value = str(value)
    try:
        cleaned = str_value.replace(',', '').replace('%', '')
        return float(cleaned)
    except:
        return value

def detect_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Detect columns that might contain numeric values with commas and percentage signs."""
    numeric_columns = []
    
    for col in df.columns:
        sample = df[col].dropna().head(100)
        numeric_count = 0
        
        for val in sample:
            if isinstance(val, (int, float)):
                numeric_count += 1
                continue
                
            if isinstance(val, str):
                cleaned_val = val.strip()
                if not cleaned_val:
                    continue
                
                cleaned_val = cleaned_val.replace('%', '').strip()
                numeric_pattern = re.compile(r'^-?[\d,]*\.?\d+$')
                if numeric_pattern.match(cleaned_val):
                    numeric_count += 1
        
        if len(sample) > 0 and numeric_count / len(sample) > 0.5 and col != 'Item Code':
            numeric_columns.append(col)
            
    return numeric_columns

class ExcelPreprocessor:
    def __init__(self, header_scan_rows: int = 5):
        self.header_scan_rows = header_scan_rows

    def process_excel_folder(self, excel_folder: str, output_dir: str) -> List[str]:
        """Process all Excel files in a folder and convert them to cleaned CSV format."""
        os.makedirs(output_dir, exist_ok=True)
        created_files = []
        
        for file_name in os.listdir(excel_folder):
            if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
                excel_path = os.path.join(excel_folder, file_name)
                xl = pd.ExcelFile(excel_path)
                
                for sheet in xl.sheet_names:
                    try:
                        df = pd.read_excel(excel_path, sheet_name=sheet, header=None)
                        scan_range = min(self.header_scan_rows, len(df))
                        header_row = df.iloc[:scan_range].apply(lambda x: x.notna().sum(), axis=1).idxmax()
                        
                        df.columns = df.iloc[header_row]
                        df = df.iloc[header_row + 1:].reset_index(drop=True)
                        df.columns = df.columns.str.strip().str.replace('\n', ' ')
                        
                        numeric_columns = detect_numeric_columns(df)
                        for col in numeric_columns:
                            df[col] = df[col].apply(clean_numeric_value)
                
                        csv_filename = f"{os.path.splitext(file_name)[0]}_{sheet}.csv"
                        csv_path = os.path.join(output_dir, csv_filename)
                        df.to_csv(csv_path, index=False)
                        created_files.append(csv_path)
                    except Exception as e:
                        print(f"Error processing sheet '{sheet}' in file '{file_name}': {str(e)}")
        
        return created_files

class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        return AgentFinish(
            return_values={"output": llm_output},
            log=llm_output
        )

class ERPPlanner:
    def __init__(self, excel_folder: str, processed_folder: str, openai_api_key: str):
        self.excel_folder = excel_folder
        self.processed_folder = processed_folder
        self.file_data = {}
        self.file_descriptions = {}
        self.agent_executor = None
        
        self.llm = ChatOpenAI(
            temperature=0,
            api_key=openai_api_key
        )
    
        self.preprocessor = ExcelPreprocessor()
        self._process_and_load_files()
    
    def _process_and_load_files(self):
        """Process Excel files and load resulting CSVs into memory"""
        self.preprocessor.process_excel_folder(self.excel_folder, self.processed_folder)
        
        for file_name in os.listdir(self.processed_folder):
            if file_name.endswith('.csv'):
                file_path = os.path.join(self.processed_folder, file_name)
                try:
                    df = pd.read_csv(file_path)
                    self.file_data[file_name] = df
                except Exception as e:
                    print(f"Error loading {file_name}: {str(e)}")

    def add_file_description(self, file_name: str, column_descriptions: str):
        """Add description for a CSV file's columns and set up the agent."""
        self.file_descriptions[file_name] = column_descriptions
        self._setup_agent()

    def execute_analysis(self, steps: str) -> Dict[str, Any]:
        """Execute the analysis steps and return both code and execution status."""
        try:
            if not self.file_data:
                raise ValueError("No data files have been loaded")
            
            code = self._generate_analysis_code(
                self.llm, 
                self.file_descriptions, 
                steps,
                self.file_data  # Pass the dictionary of DataFrames
            )
            stdout_buffer = io.StringIO()
            sys.stdout = stdout_buffer  # Redirect stdout
            # Execute the code
            local_vars = {'files_data': self.file_data, 'plt': plt, 'pd': pd, 'np': np, 'sns': sns}
            exec(code, globals(), local_vars)
            sys.stdout = sys.__stdout__
            execution_output = stdout_buffer.getvalue().strip()
            print("THE OUTPUT IS THIS->",execution_output)
            return {
                "status": "success",
                "code": code,
                "output": execution_output,
                "message": "Analysis completed successfully"
            }
        except Exception as e:
            sys.stdout = sys.__stdout__
            return {
                "status": "error",
                "code": None,
                "output": None,
                "message": f"Error executing analysis: {str(e)}"
            }

    def _generate_analysis_code(self, llm: ChatOpenAI, files_descriptions: Dict[str, str], steps: str, files_data: Dict[str, pd.DataFrame]) -> str:
        """
        Debug version of the code generator with additional data validation checks.
        """
        # Debug print 1: Check if files_data dictionary contains our file
        print("\nDebug - Available files in files_data:")
        print(list(files_data.keys()))

        # Debug print 2: Check the actual DataFrame
        d1 = files_data['div_profit_aug_purchase.csv']
        print("\nDebug - DataFrame info:")
        print(d1.info())
        
        # Debug print 3: Check column names explicitly
        print("\nDebug - Column names in DataFrame:")
        print(d1.columns.tolist())
        
        # Debug print 4: Check first few rows of data
        print("\nDebug - First 5 rows of DataFrame:")
        print(d1.head())
        
        # Debug print 5: Check data types of columns
        print("\nDebug - DataFrame dtypes:")
        print(d1.dtypes)

        # Create a formatted string of file descriptions
        file_schema_info = ""
        for filename, description in files_descriptions.items():
            file_schema_info += f"\nFile: {filename}\nSchema:\n{description}\n"
        # Create prompt for the LLM with explicit indentation instructions
        prompt = f"""You are a Python code generator expert. Given the following analysis steps and file descriptions, 
            generate executable Python code using pandas, matplotlib, and seaborn.
            
            Available files and their schemas:
            {file_schema_info}

            Example of how to access a DataFrame:
            df = files_data['div_profit_aug_purchase.csv']

            Analysis Steps:
            {steps}
            
            Generate complete, executable Python code that:
            1. Uses the correct DataFrame(s) based on the file descriptions and required columns
            2. Always prints the final results using print() statements
            3. If visualizations are specified, creates them appropriately
            4. Includes proper styling and formatting
            5. Uses seaborn for better visual appeal
            6. Includes comments explaining the analysis steps
            7. Avoid using pd.melt() unless data structure specifically requires it
            8. For bar plots, prefer the pattern:
            - Create a new DataFrame with 'Metric' and 'Value' columns
            - Use sns.barplot(data=df, x='Metric', y='Value')
            9. Always ensure the final results are visible to the user 

            Important:
            - The column names in the data are exactly: {', '.join(d1.columns.tolist())}
            - Do not modify these column names in the code
            - Use these exact column names in the visualization code
            - Do not include any import statements - they are already handled
            - Use files_data dictionary to access DataFrames
            - Ensure that you are using correct data types for columns , the data type for item code is string
            """

        # Get code from LLM
        response = llm.invoke(prompt)
        code = response.content
        
        # Clean up code and add wrapper
        code = code.replace('```python', '').replace('```', '')
        code = '\n'.join([line for line in code.splitlines() if not line.strip().startswith('import ')])
        
        return self._wrap_code(code)

    def _format_file_descriptions(self) -> str:
        """Format file descriptions for prompt."""
        return "\n".join([
            f"File: {file}\nSchema:\n{desc}\n"
            for file, desc in self.file_descriptions.items()
        ])

    def _wrap_code(self, code: str) -> str:
        """Wrap generated code with necessary imports and error handling."""
        wrapped_code = """
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

try:
"""
        indented_code = '\n'.join(['    ' + line for line in code.splitlines()])
        wrapped_code += indented_code
        
        wrapped_code += """
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f'Error during analysis: {str(e)}')
    raise
"""
        return wrapped_code

    def _setup_agent(self):
        """Set up the planning agent with tools and prompt."""
        tools = [
            Tool(
                name="GetAvailableFiles",
                func=self.get_available_files,
                description="Get information about available CSV files and their columns."
            ),
            Tool(
                name="AnalyzeSteps",
                func=self.analyze_required_steps,
                description="Analyze what steps are needed to answer the query."
            )
        ]

        # Get tool names
        tool_names = [tool.name for tool in tools]

        # Create the prompt template with all required variables
        prompt = PromptTemplate.from_template(
            """You are an ERP planning assistant capable of sophisticated data analysis.
            Available Files and Their Descriptions:
            {files_info}

            Question: {input}
            
            You have access to the following tools:

            {tools}

            Use the following format:

            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Begin!

            {agent_scratchpad}"""
        )

        # Create base prompt with the known variables
        base_prompt = prompt.partial(
            files_info="\n".join([
                f"File: {file}\nDescription:\n{desc}\n"
                for file, desc in self.file_descriptions.items()
            ]),
            tool_names=", ".join(tool_names),
            agent_scratchpad=""  # Empty string for initial state
        )

        # Create the agent with the ReAct prompt
        agent = create_react_agent(
            llm=self.llm,
            tools=tools,
            prompt=base_prompt
        )

        # Create the agent executor
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True  # Add error handling
        )

    def plan_and_execute(self, question: str) -> Dict[str, Any]:
        """Plan and execute the analysis, returning both plan and results."""
        if self.agent_executor is None:
            raise ValueError("Agent not initialized. Please add file descriptions first.")
        
        result = self.agent_executor.invoke({"input": question})
        execution_result = self.execute_analysis(result["output"])
        
        return {
            "plan": result["output"],
            "execution_result": execution_result
        }

    def get_available_files(self,*args) -> Dict[str, str]:
        """Get list of available CSV files and their column descriptions."""
        return self.file_descriptions

    def analyze_required_steps(self, query: str) -> str:
        """Analyze query and determine required steps."""
        prompt = f"""Given the query: {query}
        And the available files with their descriptions:
        {self._format_file_descriptions()}
        
        Determine the required analysis steps."""
        
        response = self.llm.invoke(prompt)
        return response.content


def call_erp_assistant(state: MessagesState) -> Command[Literal["xyro_employee_base"]]:
    """
    Handler function to process ERP queries and return to xyro_employee_base.
    This function will be called by the main system's state graph.
    """
    try:
        # Initialize the ERP planner
        planner = ERPPlanner(
            excel_folder='xlfiles',
            processed_folder='csvfiles',
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Add file descriptions
        planner.add_file_description(
            "div_profit_aug_purchase.csv",
            """
            Division and supplier profit data with columns:
            - Division: Business division name
            - Brand /Principal: Brand name
            - Supplier Name: supplier name
            - Item Code: Unique item identifier
            - Item Description: What type of item is it 
            - Item Specification: Detailed item specifications
            - Purchase value: Purchase cost value
            - Sale value: Sale price value
            - Profit: Absolute profit (sale_value - purchase_value)
            - Profit %: Profit margin as percentage
            """
        )
        planner.add_file_description(
            "sales_comp_details_aug_Sheet1.csv",
            """
            Sales compensation details with columns:
            - Month: Month and year of the sales data (e.g., Jan-24)
            - FA Target/Order/Invoice: Financial Analytics target, order, and invoice amounts
            - PA Target/Order/Invoice: Product Analytics target, order, and invoice amounts
            - EQ Target/Order/Invoice: Equipment target, order, and invoice amounts
            - HY Target/Order/Invoice: Hybrid target, order, and invoice amounts
            - Gen Target/Order/Invoice: General target, order, and invoice amounts
            - DC-PA Target/Order/Invoice: DC-Product Analytics target, order, and invoice amounts
            - EA Target/Order/Invoice: Enterprise Analytics target, order, and invoice amounts
            - Total Target: Overall target amount
            - Achieved Order %: Percentage of target achieved in orders
            - Achieved Invoice %: Percentage of target achieved in invoices
            """
        )
        planner.add_file_description(
            "sales_details_aug_Sheet1.csv",
            """
            Sales performance details by area/representative with columns:
            - Code: Area or representative code
            - Name: Sales representative or area manager name
            - Target: Sales target value
            - Quote Value: Total quoted amount
            - Order Value: Total order amount received
            - Invoice Value: Total invoiced amount
            - Collection Value: Total amount collected
            - Achieved %: Percentage of target achieved (based on Invoice Value)
            """
        )
        planner.add_file_description(
            "sales_details_aug_Detailed.csv",
            """
            Detailed sales performance breakdown by area/representative and product division with columns:
            - Code: Area or representative code
            - Name: Sales representative or area manager name
            - Target: Sales target value
            - Quote Value: Total quoted amount
            - Order Value: Total order amount received
            - Invoice Value: Total invoiced amount
            - Collection Value: Total amount collected
            - Achieved %: Percentage of target achieved (based on Invoice Value)
            
            Each area's data is further broken down by division categories:
            - DC-PA: DC Product Analytics division
            - EA: Enterprise Analytics division
            - EQ: Equipment division
            - FA: Financial Analytics division
            - Gen: General division
            - HY: Hybrid division
            - PA: Product Analytics division
            
            The file provides both summary rows for each area and detailed breakdowns by division within each area.
            """
        )


        
        # Get query from state
        query = state["messages"][0].content
        
        # Execute analysis
        result = planner.plan_and_execute(query)
        
         # Format result - simplified to just the output
        if result["execution_result"]["status"] == "success":
            response_content = result['execution_result']['output']
            
            # Return command with properly formatted message
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content=response_content,
                            name="erp_assistant"
                        )
                    ]
                },
                goto="xyro_employee_base"
            )
        else:
            error_message = f"Error in analysis: {result['execution_result']['message']}"
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content=error_message,
                            name="erp_assistant"
                        )
                    ]
                },
                goto="xyro_employee_base"
            )
            
    except Exception as e:
        error_message = f"Error in ERP analysis: {str(e)}"
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=error_message,
                        name="erp_assistant"
                    )
                ]
            },
            goto="xyro_employee_base"
        )