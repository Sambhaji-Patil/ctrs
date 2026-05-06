from fastapi import FastAPI, HTTPException
from fastapi import Body
from pydantic import BaseModel
from openai import AsyncOpenAI
import asyncio
import json
import os
import uuid
from dotenv import load_dotenv
from models import ClusterRequest

from utils.filtering import  get_representatives 


os.makedirs("output", exist_ok=True)


# Import the existing pipeline
from agents import run_pipeline

load_dotenv()

app = FastAPI()

# Configure AsyncOpenAI Client with Requesty settings
client = AsyncOpenAI(
    api_key=os.environ.get("REQUESTY_API_KEY", "missing_key"),
    base_url="https://router.requesty.ai/v1",
    default_headers={
        "HTTP-Referer": "https://yourapp.com", 
        "X-Title": "My App",
    }
)

class LogsRequest(BaseModel):
    logs: str

async def get_insights(report: str) -> str:
    response = await client.chat.completions.create(
        model="openai/gpt-4o",
        messages=[
            {"role": "system", "content": "You are an Insights Agent. Analyze the provided user session logs/report. Provide exactly 5-7 short, easy-to-read numbered points containing the key behavioral insights. Ensure these insights are highly specific to the details in the provided story/logs and avoid any generic observations."},
            {"role": "user", "content": f"Here is the report generated from the logs:\n{report}"}
        ]
    )
    return response.choices[0].message.content

async def get_state_flow(report: str) -> str:
    response = await client.chat.completions.create(
        model="openai/gpt-4o",
        messages=[
             {"role": "system", "content": "You are a State Flow Agent. Analyze the report/logs and generate a high-level state flow diagram in Mermaid JS format (flowchart TD). Keep it to a maximum of 10 nodes. IMPORTANT: You must ONLY include the REASON on the edges (arrows) for turning point decisions, moments of friction, or loops (e.g., when the user goes back to adjust the cart because of price constraints). DO NOT annotate standard forward steps (like opening the app, or standard progression) with descriptions on the edges. If the user loops back or returns to a previous state, correctly map the arrow back to the previous node and explicitly state the reason on that edge. Return ONLY the raw Mermaid code string without markdown wrappers (e.g., no ```mermaid)."},
             {"role": "user", "content": f"Here is the report generated from the logs:\n{report}"}
        ]
    )
    return response.choices[0].message.content.strip()

async def get_suggestions(report: str) -> str:
    response = await client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a Suggestion Agent. Review the user progression and provide ONLY 5 actionable business recommendations to improve conversion. Format your response strictly as a numbered list of short, concise points. Make all suggestions highly specific to the provided story and logs; do not include generic advice."},
            {"role": "user", "content": f"Here is the report generated from the logs:\n{report}"}
        ]
    )
    return response.choices[0].message.content

@app.post("/processed-logs")
async def process_logs(request: LogsRequest):
    try:
        # 1. Run the existing pipeline to get the report (CTR)
        # Since it is synchronous and does I/O, we offload to a thread
        output_dir = f"output_{uuid.uuid4().hex}"
        report = await asyncio.to_thread(run_pipeline, request.logs, output_dir)

        # 2. Run all three new agents concurrently using the generated report string
        insights, state_flow, suggestions = await asyncio.gather(
            get_insights(report),
            get_state_flow(report),
            get_suggestions(report)
        )

        combined = f"""
# REPORT:
{report}

# STATE FLOW
```mermaid
{state_flow.replace("\n", """
""")}
```

# INSIGHTS:
{insights}

# SUGESTIONS:
{suggestions}
"""
        os.makedirs("output", exist_ok=True)
        x = len(os.listdir("output"))
        file_name = f"output/output{x}.md"
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(combined)

        # 3. Return the final output
        return {
            "report": report,
            "insights": insights,
            "state_flow": state_flow,
            "suggestions": suggestions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/get_representatives")
def cluster_texts(request: ClusterRequest = Body(...)):
    if not request.texts or len(request.texts) == 0:
        raise HTTPException(status_code=400, detail="texts list cannot be empty")

    try:
        reps = get_representatives(
            request.texts,
            request.eps,
            request.min_samples
        )

        return {
            "input_size": len(request.texts),
            "output_size": len(reps),
            "representatives": reps
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))