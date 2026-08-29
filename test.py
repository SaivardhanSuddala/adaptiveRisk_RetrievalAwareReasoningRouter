
from dotenv import load_dotenv
import os
load_dotenv()
for k in ['GROQ_API_KEY','GEMINI_API_KEY','CEREBRAS_API_KEY']:
    v = os.getenv(k)
    print(k, '-> SET' if v else '-> MISSING')
